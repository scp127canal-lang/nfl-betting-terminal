"""Servidor FastAPI: matchups, props, combinadas y chat con Gemini (function calling).

Ejecutar:  uvicorn main:app --reload
"""

from __future__ import annotations

import logging
import math
import threading
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

import bet_builder as bb
import data_pipeline as dp
import prediction_engine as pe
import scheduler
from config import GEMINI_API_KEY, GEMINI_MODEL, american_to_decimal
from odds_pipeline import MARKET_STAT_MAP

logger = logging.getLogger("nfl_api")

# stat -> mercado representativo (el primero de cada grupo; total_tds = Anytime TD)
STAT_TO_MARKET = {}
for _m, _s in MARKET_STAT_MAP.items():
    STAT_TO_MARKET.setdefault(_s, _m)
STAT_TO_MARKET.pop("first_td", None)  # Primer TD necesita el contexto del partido
ABBR_TO_NAME = {v: k for k, v in bb.TEAM_ABBR.items()}
RISK_LEVELS = ("BAJO", "MODERADO", "ALTO")


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def jsonable(obj):
    """Convierte numpy/pandas/NaN a tipos JSON válidos."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if math.isnan(obj) or math.isinf(obj) else float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    return obj


def _decimal_or_none(american) -> float | None:
    try:
        return None if pd.isna(american) else american_to_decimal(american)
    except ValueError:
        return None


def resolve_team(team: str) -> str:
    team = team.strip()
    if team.upper() in ABBR_TO_NAME:
        return team.upper()
    for name, abbr in bb.TEAM_ABBR.items():
        if team.lower() in name.lower():
            return abbr
    raise ValueError(f"Equipo desconocido: {team}")


def resolve_stat(market: str) -> str:
    if market in MARKET_STAT_MAP and MARKET_STAT_MAP[market] in STAT_TO_MARKET:
        return MARKET_STAT_MAP[market]
    if market in STAT_TO_MARKET:
        return market
    raise ValueError(f"Mercado no soportado: {market}. Usa {sorted(MARKET_STAT_MAP)}")


# --------------------------------------------------------------------------- #
# Estado de la app (datos + combinadas cacheadas por snapshot de cuotas)
# --------------------------------------------------------------------------- #
class AppState:
    def __init__(self):
        self._lock = threading.Lock()
        self._parlays: dict | None = None
        self._parlays_key: str | None = None

    def week(self) -> dict:
        return dp.get_current_nfl_week()

    def seasons(self) -> list[int]:
        season = self.week()["season"]
        return [season - 1, season]

    def weekly(self) -> pd.DataFrame:
        return dp.load_weekly_stats(self.seasons())

    def schedules(self) -> pd.DataFrame:
        return dp.load_schedules(self.seasons())

    def week_games(self, season: int | None = None, week: int | None = None) -> pd.DataFrame:
        info = self.week()
        season, week = season or info["season"], week or info["week"]
        sched = self.schedules()
        return sched[(sched["season"] == season) & (sched["week"] == week)].reset_index(drop=True)

    def game_for_team(self, team: str) -> pd.Series:
        games = self.week_games()
        match = games[(games["home_team"] == team) | (games["away_team"] == team)]
        if match.empty:
            raise ValueError(f"{team} no juega en la semana {self.week()['week']} (bye)")
        return match.iloc[0]

    def invalidate(self, *_):
        with self._lock:
            self._parlays, self._parlays_key = None, None

    def parlays(self, max_legs: int = 6) -> dict:
        snap = scheduler.get_snapshot()
        meta = snap.get("meta")
        if not meta or snap["player_props"] is None:
            raise LookupError("Aún no hay cuotas descargadas. Se actualizan a la hora "
                              "MORNING_REFRESH_HOUR los días con partido (o POST /odds/refresh).")
        key = f"{meta['refreshed_at']}|{max_legs}"
        with self._lock:
            if self._parlays_key == key:
                return self._parlays
        weekly, sched = self.weekly(), self.schedules()
        legs = (bb.build_player_legs(snap["player_props"], weekly, main_only=True)
                + bb.build_team_legs(snap["game_odds"], sched))
        tiers = bb.build_parlays(legs, weekly=weekly, max_legs=max_legs)
        result = {"odds_refreshed_at": meta["refreshed_at"], "legs_evaluated": len(legs),
                  "tiers": tiers}
        with self._lock:
            self._parlays, self._parlays_key = result, key
        return result


state = AppState()
gemini_client: genai.Client | None = None


# --------------------------------------------------------------------------- #
# Lógica de negocio (compartida por REST y por las tools del chat)
# --------------------------------------------------------------------------- #
def matchups_data(season: int | None = None, week: int | None = None) -> dict:
    info = state.week()
    games = state.week_games(season, week)
    sched = state.schedules()
    out = []
    for g in games.itertuples(index=False):
        row = {"game_id": g.game_id, "away_team": g.away_team, "home_team": g.home_team,
               "gameday": g.gameday, "gametime": getattr(g, "gametime", None),
               "home_score": getattr(g, "home_score", None), "away_score": getattr(g, "away_score", None),
               "vegas": {"spread_line_home": getattr(g, "spread_line", None),
                         "total_line": getattr(g, "total_line", None),
                         "home_ml_decimal": _decimal_or_none(getattr(g, "home_moneyline", None)),
                         "away_ml_decimal": _decimal_or_none(getattr(g, "away_moneyline", None))}}
        hl, al = dp.team_points_log(sched, g.home_team), dp.team_points_log(sched, g.away_team)
        if len(hl) >= 3 and len(al) >= 3:
            sim = pe.simulate_game_points(hl["points_for"], hl["points_against"],
                                          al["points_for"], al["points_against"])
            row["monte_carlo"] = {"proj_total": sim["proj_total"],
                                  "proj_margin_home": sim["proj_margin"],
                                  "home_win_pct": sim["home_win_pct"],
                                  "away_win_pct": round(100 - sim["home_win_pct"], 2)}
        out.append(row)
    return jsonable({"season": season or info["season"], "week": week or info["week"],
                     "season_type": info["season_type"], "games": out})


def predict_prop_data(player: str, market: str, line: float, decimal_odds: float,
                      side: str = "over") -> dict:
    stat = resolve_stat(market)
    log = dp.get_player_game_log(state.weekly(), player, pe.REALISM_GAMES)
    if log.empty:
        raise LookupError(f"Sin historial para {player}")
    result = pe.evaluate_manual_prop(line, decimal_odds, log, stat, side=side.lower())
    result.update({"player": log["player_display_name"].iloc[-1], "stat": stat,
                   "team": log["recent_team"].iloc[-1] if "recent_team" in log else None,
                   "last_games": log[["season", "week", stat]].to_dict("records")})
    return jsonable(result)


def parlays_data(risk_level: str = "TODOS", top: int = 5, max_legs: int = 6) -> dict:
    data = state.parlays(max_legs)
    level = risk_level.upper()
    levels = RISK_LEVELS if level == "TODOS" else (level,)
    if any(lv not in RISK_LEVELS for lv in levels):
        raise ValueError(f"Nivel de riesgo inválido: {risk_level}. Usa BAJO, MODERADO, ALTO o TODOS")
    return jsonable({
        "odds_refreshed_at": data["odds_refreshed_at"],
        "legs_evaluated": data["legs_evaluated"],
        "tiers": {lv: {"badge": bb.RISK_TIERS[lv]["badge"], "rule": bb.RISK_TIERS[lv]["rule"],
                       "parlays": data["tiers"][lv][:top]} for lv in levels},
    })


# --------------------------------------------------------------------------- #
# Modelos de request
# --------------------------------------------------------------------------- #
class PropRequest(BaseModel):
    player: str = Field(..., examples=["Joe Burrow"])
    market: str = Field(..., examples=["player_pass_yds_alternate", "passing_yards"])
    line: float = Field(..., gt=0, examples=[283.0])
    decimal_odds: float = Field(..., gt=1, examples=[1.57])
    side: Literal["over", "under"] = "over"


class LegRequest(BaseModel):
    type: Literal["player", "team"]
    market: str = Field(..., description="Mercado de jugador o h2h / spreads / totals")
    decimal_odds: float = Field(..., gt=1)
    side: str = Field("over", description="over/under para props y totals; ignorado en h2h/spreads")
    line: float | None = None
    player: str | None = None
    team: str | None = Field(None, description="Equipo elegido (h2h/spreads) o cualquiera del partido (totals)")


class ParlayRequest(BaseModel):
    legs: list[LegRequest] = Field(..., min_length=2)
    bankroll: float = Field(1000.0, gt=0)


class ChatTurn(BaseModel):
    role: Literal["user", "model"]
    text: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatTurn] = []


# --------------------------------------------------------------------------- #
# Construcción de patas para /parlay/evaluate
# --------------------------------------------------------------------------- #
def _player_leg(req: LegRequest) -> bb.Leg:
    if not req.player or req.line is None:
        raise ValueError("Una pata de jugador necesita 'player' y 'line'")
    stat = resolve_stat(req.market)
    weekly = state.weekly()
    log = dp.get_player_game_log(weekly, req.player, pe.REALISM_GAMES)
    if log.empty:
        raise ValueError(f"Sin historial para {req.player}")
    game = state.game_for_team(log["recent_team"].iloc[-1])
    props = pd.DataFrame([{
        "event_id": game["game_id"], "market": STAT_TO_MARKET[stat], "player": req.player,
        "outcome": req.side.capitalize(), "line": req.line, "decimal_odds": req.decimal_odds}])
    legs = bb.build_player_legs(props, weekly)
    if not legs:
        guard = pe.validate_prop_realism(log, req.line, stat)
        raise ValueError(f"Realism Guard: {guard['reason']}")
    return legs[0]


def _team_leg(req: LegRequest) -> bb.Leg:
    if req.market not in ("h2h", "spreads", "totals") or not req.team:
        raise ValueError("Una pata de equipo necesita market h2h/spreads/totals y 'team'")
    if req.market != "h2h" and req.line is None:
        raise ValueError(f"El mercado {req.market} necesita 'line'")
    team = resolve_team(req.team)
    game = state.game_for_team(team)
    outcome = req.side.capitalize() if req.market == "totals" else ABBR_TO_NAME[team]
    odds = pd.DataFrame([{
        "event_id": game["game_id"], "home_team": ABBR_TO_NAME[game["home_team"]],
        "away_team": ABBR_TO_NAME[game["away_team"]], "market": req.market,
        "outcome": outcome, "line": req.line, "decimal_odds": req.decimal_odds}])
    legs = bb.build_team_legs(odds, state.schedules())
    if not legs:
        raise ValueError(f"Sin historial suficiente para simular {game['away_team']}@{game['home_team']}")
    return legs[0]


# --------------------------------------------------------------------------- #
# Tools del chat (Gemini function calling automático)
# --------------------------------------------------------------------------- #
def _safe(fn, *args, **kwargs) -> dict:
    try:
        return fn(*args, **kwargs)
    except (LookupError, ValueError) as exc:
        return {"error": str(exc)}


def obtener_semana_actual() -> dict:
    """Devuelve la temporada y la semana NFL activas hoy."""
    return jsonable(state.week())


def obtener_partidos(semana: int = 0) -> dict:
    """Partidos de una semana NFL con proyecciones Monte Carlo (total, margen,
    % de victoria) y líneas de Las Vegas en cuota decimal.

    Args:
        semana: número de semana; 0 = semana actual.
    """
    return _safe(matchups_data, None, semana or None)


def obtener_combinadas(nivel_riesgo: str = "TODOS", max_resultados: int = 3) -> dict:
    """Combinadas +EV generadas por el simulador con las cuotas actuales.

    Args:
        nivel_riesgo: BAJO (prob > 50%, cuota 1.80-3.20), MODERADO (prob 30-50%,
            cuota 3.20-7.50), ALTO (prob < 30%, cuota > 7.50) o TODOS.
        max_resultados: cuántas combinadas devolver por nivel.
    """
    return _safe(parlays_data, nivel_riesgo, max(1, min(max_resultados, 10)))


def evaluar_prop(jugador: str, mercado: str, linea: float, cuota_decimal: float,
                 lado: str = "over") -> dict:
    """Evalúa una apuesta de jugador: probabilidad implícita, probabilidad real
    Monte Carlo, % EV y Realism Guard.

    Args:
        jugador: nombre del jugador, ej. "Joe Burrow".
        mercado: passing_yards, rushing_yards, receiving_yards o receptions.
        linea: línea de la apuesta, ej. 283.5.
        cuota_decimal: cuota en formato decimal puro, ej. 1.57.
        lado: "over" o "under".
    """
    return _safe(predict_prop_data, jugador, mercado, linea, cuota_decimal, lado)


CHAT_TOOLS = [obtener_semana_actual, obtener_partidos, obtener_combinadas, evaluar_prop]

SYSTEM_PROMPT = """Eres el analista de apuestas NFL de esta app. Respondes en español, claro y breve.
Reglas:
- Usa SIEMPRE las herramientas para obtener números; nunca inventes probabilidades, cuotas ni stats.
- Las cuotas se muestran SOLO como número decimal puro (ej. 1.91, 4.35). Nunca añadas 'x',
  prefijos, sufijos ni formato americano.
- Al hablar de combinadas, indica el badge del nivel (🛡️ RIESGO BAJO, ⚖️ RIESGO MODERADO,
  🚀 ALTO RIESGO), las selecciones, la cuota decimal, la probabilidad conjunta, el % EV y el
  stake sugerido por cada $1,000 (Kelly 25%).
- Si una herramienta devuelve "error", explícalo al usuario sin inventar datos.
- Si una línea es RECHAZADA por el Realism Guard, dilo y explica el motivo."""


def _tools_used(response) -> list[str]:
    names = []
    for content in response.automatic_function_calling_history or []:
        for part in content.parts or []:
            if part.function_call:
                names.append(part.function_call.name)
    return names


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global gemini_client
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if GEMINI_API_KEY and GEMINI_API_KEY != "tu_api_key_aqui":
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        logger.warning("GEMINI_API_KEY no configurada: /chat deshabilitado")
    scheduler.on_refresh(state.invalidate)
    scheduler.load_snapshot_from_disk()
    scheduler.start_scheduler()
    yield
    scheduler.stop_scheduler()


app = FastAPI(title="NFL +EV API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return jsonable({"status": "ok", "week": state.week(), "gemini": gemini_client is not None,
                     "next_odds_refresh": scheduler.next_run_time(),
                     "odds_snapshot": scheduler.get_snapshot().get("meta")})


@app.get("/matchups")
def matchups(season: int | None = Query(None), week: int | None = Query(None, ge=1, le=22)):
    return matchups_data(season, week)


@app.post("/predict/prop")
def predict_prop(req: PropRequest):
    try:
        return predict_prop_data(req.player, req.market, req.line, req.decimal_odds, req.side)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.post("/parlay/evaluate")
def parlay_evaluate(req: ParlayRequest):
    try:
        legs = [_player_leg(leg) if leg.type == "player" else _team_leg(leg) for leg in req.legs]
        result = bb.evaluate_parlay(legs, weekly=state.weekly(), bankroll=req.bankroll)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return jsonable(result)


@app.get("/parlays")
def parlays(risk: str = Query("TODOS", description="BAJO | MODERADO | ALTO | TODOS"),
            top: int = Query(5, ge=1, le=20)):
    try:
        return parlays_data(risk, top)
    except LookupError as exc:
        raise HTTPException(503, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.post("/odds/refresh")
def odds_refresh(force: bool = Query(False, description="Descargar aunque hoy no haya partido")):
    return jsonable(scheduler.refresh_odds(force=force))


@app.post("/chat")
def chat(req: ChatRequest):
    if gemini_client is None:
        raise HTTPException(503, "GEMINI_API_KEY no configurada")
    contents = [types.Content(role=t.role, parts=[types.Part(text=t.text)]) for t in req.history]
    contents.append(types.Content(role="user", parts=[types.Part(text=req.message)]))
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=CHAT_TOOLS,
                temperature=0.3,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    maximum_remote_calls=8),
            ),
        )
    except Exception as exc:
        logger.exception("Fallo en Gemini")
        raise HTTPException(502, f"Error de Gemini: {exc}")
    return {"reply": response.text or "", "tools_used": _tools_used(response)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
