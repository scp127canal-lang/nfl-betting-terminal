"""Capa de datos y componentes visuales de la Terminal NFL (Streamlit).

Fuentes de cuotas, en orden de prioridad:
  1. Snapshot del scheduler (cache/odds, descargado a las 8:00 los días con partido).
  2. The-Odds-API en vivo (si hay THE_ODDS_API_KEY / ODDS_API_KEY; caché de 30 min).
  3. Líneas de Las Vegas del calendario nflverse (spread, ML y total; sin props).

REGLA: toda cuota se muestra como número decimal puro (1.91), sin 'x' ni sufijos.
"""

from __future__ import annotations

import hmac
import json
import html
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bet_builder as bb
import data_pipeline as dp
import odds_pipeline as op
import prediction_engine as pe
import scheduler
from config import COLOR_NEON_GREEN, COLOR_NEON_PINK, THE_ODDS_API_KEY, american_to_decimal

SEED = 7

# --- Paleta Dark Premium ---
BG_DEEP = "#090C10"
BG_PANEL = "#0D1117"
BG_CARD = "#161B22"
BORDER = "#21262D"
BORDER_HI = "#30363D"
TEXT = "#E6EDF3"
MUTED = "#8B949E"
YELLOW = "#F5C542"
GREEN, PINK = COLOR_NEON_GREEN, COLOR_NEON_PINK

ESPN_LOGO = "https://a.espncdn.com/i/teamlogos/nfl/500/{}.png"
ESPN_ABBR = {"LA": "lar", "WAS": "wsh"}
ABBR_TO_NAME = {v: k for k, v in bb.TEAM_ABBR.items()}
ODDS_API_ENABLED = bool(THE_ODDS_API_KEY) and THE_ODDS_API_KEY != "tu_odds_api_key_aqui"

POSITIONS = ["QB", "RB", "WR", "TE"]
GROUPS = op.MARKET_GROUPS                     # pestañas de mercado (principal + alterno unidos)
GROUP_ORDER = list(GROUPS)
GROUP_LABEL = {g: s["label"] for g, s in GROUPS.items()}
MARKET_LABEL = {m: GROUPS[g]["label"] for m, g in op.MARKET_TO_GROUP.items()}
CATEGORIES = {"td": "⚽ Touchdowns", "pass": "🏈 Pases", "rush": "🏃 Acarreos",
              "rec": "🎯 Recepciones", "all": "🔀 Todos"}
DEFAULT_CATEGORY = {"QB": "pass", "RB": "rush", "WR": "rec", "TE": "rec"}
# Sin props en vivo: mercados de referencia por posición
POSITION_GROUPS = {
    "QB": ["pass_yds", "pass_tds", "pass_att", "pass_cmp", "pass_int", "rush_yds", "anytime_td"],
    "RB": ["rush_yds", "rush_att", "rec_yds", "receptions", "rush_rec_yds", "anytime_td"],
    "WR": ["rec_yds", "receptions", "rush_rec_yds", "anytime_td"],
    "TE": ["rec_yds", "receptions", "anytime_td"],
}
STAT_UNIT = {"passing_yards": ("Yds", "yardas"), "rushing_yards": ("Yds", "yardas"),
             "receiving_yards": ("Yds", "yardas"), "rush_rec_yds": ("Yds", "yardas"),
             "receptions": ("Rec", "recepciones"), "passing_tds": ("TD", "TDs de pase"),
             "total_tds": ("TD", "TDs"), "attempts": ("Int.", "intentos"),
             "completions": ("Comp.", "completos"), "passing_interceptions": ("INT", "intercepciones"),
             "carries": ("Acarreos", "acarreos")}
CATEGORY_METRICS = {
    "td": [("TOTAL TD", "total_tds"), ("RUSH TD", "rushing_tds"), ("REC TD", "receiving_tds"),
           ("PASS TD", "passing_tds")],
    "pass": [("PASS YDS", "passing_yards"), ("PASS TD", "passing_tds"), ("ATT", "attempts"),
             ("CMP", "completions"), ("INT", "passing_interceptions")],
    "rush": [("RUSH YDS", "rushing_yards"), ("CARRIES", "carries"), ("RUSH TD", "rushing_tds"),
             ("YDS C+R", "rush_rec_yds")],
    "rec": [("REC YDS", "receiving_yards"), ("TARGETS", "targets"), ("RECEPTIONS", "receptions"),
            ("TARGET SHARE", "target_share"), ("REC TD", "receiving_tds")],
}
POSITION_METRICS = {
    "QB": [("PASS YDS", "passing_yards"), ("PASS ATT", "attempts"),
           ("PASS TD", "passing_tds"), ("RUSH YDS", "rushing_yards")],
    "RB": [("RUSH YDS", "rushing_yards"), ("CARRIES", "carries"),
           ("REC YDS", "receiving_yards"), ("RECEPTIONS", "receptions")],
    "WR": [("REC YDS", "receiving_yards"), ("TARGETS", "targets"),
           ("RECEPTIONS", "receptions"), ("TARGET SHARE", "target_share")],
}
POSITION_METRICS["TE"] = POSITION_METRICS["WR"]
FALLBACK_DEPTH = {"QB": 1, "RB": 3, "WR": 5, "TE": 2}
TIER_COLORS = {"BAJO": GREEN, "MODERADO": YELLOW, "ALTO": PINK}


# =========================================================================== #
# Formato
# =========================================================================== #
def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def fmt_odds(x) -> str:
    """Cuota decimal pura con 2 decimales: 1.91 (nunca '1.91x')."""
    return "—" if _nan(x) else f"{float(x):.2f}"


def esc(text) -> str:
    return html.escape(str(text))


def logo_url(team: str) -> str:
    return ESPN_LOGO.format(ESPN_ABBR.get(team, team).lower())


def game_tag(event_id) -> str:
    """'2026_12_NO_CIN' -> 'NO@CIN'."""
    parts = str(event_id).split("_")
    return f"{parts[2]}@{parts[3]}" if len(parts) >= 4 and parts[0].isdigit() else ""


def badge_html(tier: str | None) -> str:
    if not tier:
        return '<span class="badge badge-NONE">Sin nivel de riesgo</span>'
    info = bb.RISK_TIERS[tier]
    return f'<span class="badge badge-{tier}">{info["emoji"]} {info["badge"]}</span>'


def ev_badge(ev_pct, fair_odds=None, plus_ev: bool | None = None, status: str | None = None) -> str:
    """🟢 +EV +8.4% · Fair Odds: 1.62  |  🟡 sin ventaja comprobada  |  EV −3.1% · Fair 1.95."""
    if _nan(ev_pct):
        return (f'<span class="evb evb-none">Fair Odds: {fmt_odds(fair_odds)}</span>'
                if not _nan(fair_odds) else "")
    if status == "unvalidated":
        return (f'<span class="evb evb-warn" title="El modelo ve valor, pero en el backtest con cuotas '
                f'históricas este mercado no ganó dinero. Trátalo como EV ≈ 0.">🟡 EV {ev_pct:+.1f}% · '
                f'sin ventaja comprobada</span>')
    plus = pe.MIN_EDGE_PCT <= ev_pct <= pe.MAX_CREDIBLE_EV if plus_ev is None else plus_ev
    if not plus and ev_pct > pe.MAX_CREDIBLE_EV:
        return (f'<span class="evb evb-none" title="EV demasiado alto para ser real: el mercado probablemente '
                f'sabe algo (lesión, rol). Verifica antes de apostar.">⚠️ Revisar · EV {ev_pct:+.0f}%</span>')
    if plus:
        return (f'<span class="evb evb-pos">🟢 +EV {ev_pct:+.1f}% · Fair Odds: '
                f'{fmt_odds(fair_odds)}</span>')
    return f'<span class="evb evb-neg">EV {ev_pct:+.1f}% · Fair {fmt_odds(fair_odds)}</span>'


def chip(text: str, cls: str = "") -> str:
    return f'<span class="chip {cls}">{text}</span>'


# =========================================================================== #
# Datos base
# =========================================================================== #
def context() -> dict:
    info = dp.get_current_nfl_week()
    return {**info, "seasons": [info["season"] - 1, info["season"]]}


def load_weekly(seasons: list[int]) -> pd.DataFrame:
    return dp.load_weekly_stats(seasons)


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    return dp.load_schedules(seasons)


def week_games(sched: pd.DataFrame, season: int, week: int, include_final: bool) -> pd.DataFrame:
    games = sched[(sched["season"] == season) & (sched["week"] == week)]
    if not include_final:
        games = games[games["home_score"].isna()]
    return games.sort_values(["gameday", "gametime"]).reset_index(drop=True)


def snapshot_key() -> str:
    meta = scheduler.get_snapshot().get("meta")
    return meta["refreshed_at"] if meta else "sin-snapshot"


def odds_status() -> str:
    meta = scheduler.get_snapshot().get("meta")
    if meta:
        try:
            from zoneinfo import ZoneInfo
            import datetime as _dt
            ts = _dt.datetime.fromisoformat(meta["refreshed_at"]).astimezone(ZoneInfo("America/New_York"))
            when = ts.strftime("%d/%m %H:%M ET")
        except (KeyError, ValueError):
            when = meta.get("refreshed_at", "")
        slot = meta.get("slot")
        slot_txt = f" · jornada {slot[-5:]} ET" if slot and slot != "manual" else ""
        return f"Cuotas actualizadas {when}{slot_txt}"
    if ODDS_API_ENABLED:
        return "The-Odds-API en vivo · caché 30 min"
    return "Vegas nflverse · sin props en vivo"


# =========================================================================== #
# Cuotas de partido (unificadas por game_id de nflverse)
# =========================================================================== #
def _dec(american) -> float | None:
    try:
        return None if american is None or pd.isna(american) else american_to_decimal(american)
    except ValueError:
        return None


def _api_game_odds() -> pd.DataFrame | None:
    snap = scheduler.get_snapshot()
    if snap.get("game_odds") is not None and not snap["game_odds"].empty:
        return snap["game_odds"]
    if ODDS_API_ENABLED:
        try:
            return op.fetch_game_odds()
        except op.OddsAPIError:
            return None
    return None


def _consensus_rows(df: pd.DataFrame) -> list[dict]:
    """Una cuota por outcome: línea más ofrecida y cuota mediana entre casas."""
    rows = []
    for (market, outcome), g in df.groupby(["market", "outcome"]):
        if market == "h2h":
            line, odds = None, g["decimal_odds"].median()
        else:
            line = g["line"].mode().iloc[0]
            odds = g.loc[g["line"] == line, "decimal_odds"].median()
        rows.append({"market": market, "outcome": outcome, "line": line,
                     "decimal_odds": round(float(odds), 2)})
    return rows


def _schedule_rows(g) -> list[dict]:
    home, away = ABBR_TO_NAME[g.home_team], ABBR_TO_NAME[g.away_team]
    spread = getattr(g, "spread_line", None)  # nflverse: positivo = local favorito
    cand = [
        ("h2h", home, None, _dec(g.home_moneyline)),
        ("h2h", away, None, _dec(g.away_moneyline)),
        ("spreads", home, None if pd.isna(spread) else -spread, _dec(g.home_spread_odds)),
        ("spreads", away, None if pd.isna(spread) else spread, _dec(g.away_spread_odds)),
        ("totals", "Over", g.total_line, _dec(g.over_odds)),
        ("totals", "Under", g.total_line, _dec(g.under_odds)),
    ]
    return [{"market": m, "outcome": o, "line": ln, "decimal_odds": d} for m, o, ln, d in cand
            if d is not None and (m == "h2h" or (ln is not None and not pd.isna(ln)))]


@st.cache_data(ttl=1800, show_spinner=False)
def game_market_board(games: pd.DataFrame, snap_key: str) -> tuple[pd.DataFrame, dict]:
    """Tablero h2h/spreads/totals por partido + mapa game_id -> event_id de la API."""
    api = _api_game_odds()
    rows, event_map = [], {}
    for g in games.itertuples(index=False):
        home, away = ABBR_TO_NAME.get(g.home_team), ABBR_TO_NAME.get(g.away_team)
        if not home or not away:
            continue
        base = {"event_id": g.game_id, "home_team": home, "away_team": away}
        match = (api[(api["home_team"] == home) & (api["away_team"] == away)]
                 if api is not None and not api.empty else pd.DataFrame())
        if not match.empty:
            event_map[g.game_id] = match["event_id"].iloc[0]
            rows += [{**base, **r, "source": "The-Odds-API"} for r in _consensus_rows(match)]
        else:
            rows += [{**base, **r, "source": "Vegas nflverse"} for r in _schedule_rows(g)]
    cols = ["event_id", "home_team", "away_team", "market", "outcome", "line", "decimal_odds", "source"]
    return pd.DataFrame(rows, columns=cols), event_map


@st.cache_resource(ttl=1800, show_spinner=False)
def team_legs(_board: pd.DataFrame, _sched: pd.DataFrame, cache_key: str) -> list[bb.Leg]:
    return bb.build_team_legs(_board, _sched, seed=SEED)


@st.cache_data(ttl=1800, show_spinner=False)
def projections(games: pd.DataFrame, sched: pd.DataFrame) -> dict:
    """Marcador proyectado Monte Carlo por partido."""
    out = {}
    for g in games.itertuples(index=False):
        hl, al = dp.team_points_log(sched, g.home_team), dp.team_points_log(sched, g.away_team)
        if len(hl) < 3 or len(al) < 3:
            continue
        sim = pe.simulate_game_points(hl["points_for"], hl["points_against"],
                                      al["points_for"], al["points_against"], seed=SEED)
        out[g.game_id] = {"home_pts": int(round(sim["home"].mean())),
                          "away_pts": int(round(sim["away"].mean())),
                          "home_win_pct": sim["home_win_pct"], "proj_total": sim["proj_total"]}
    return out


def best_market_legs(legs: list[bb.Leg], game_id: str) -> dict[str, bb.Leg]:
    """Por mercado, el lado con mayor EV según el simulador."""
    best = {}
    for leg in legs:
        if leg.event_id == game_id and (leg.market not in best or leg.ev_pct > best[leg.market].ev_pct):
            best[leg.market] = leg
    return best


# =========================================================================== #
# Props y jugadores
# =========================================================================== #
def game_props(game_id: str, api_event_id: str | None) -> pd.DataFrame:
    """Props del partido (snapshot o API en vivo) con event_id = game_id."""
    props = pd.DataFrame()
    snap = scheduler.get_snapshot().get("player_props")
    if api_event_id and snap is not None and not snap.empty:
        props = snap[snap["event_id"] == api_event_id]
    if props.empty and api_event_id and ODDS_API_ENABLED:
        try:
            props = op.fetch_event_player_props(api_event_id)
        except op.OddsAPIError:
            props = pd.DataFrame()
    return props.assign(event_id=game_id) if not props.empty else props


def slate_props(event_map: dict) -> pd.DataFrame:
    """Props del snapshot para los partidos visibles (sin gastar requests)."""
    snap = scheduler.get_snapshot().get("player_props")
    if snap is None or snap.empty or not event_map:
        return pd.DataFrame()
    reverse = {v: k for k, v in event_map.items()}
    props = snap[snap["event_id"].isin(reverse)]
    return props.assign(event_id=props["event_id"].map(reverse))


@st.cache_resource(ttl=1800, show_spinner=False)
def scan_game_props(_props: pd.DataFrame, _weekly: pd.DataFrame, cache_key: str) -> pd.DataFrame:
    """+EV de TODOS los props del partido (modelo Monte Carlo vs cuota de la casa)."""
    if _props.empty:
        return pd.DataFrame()
    return pe.evaluate_props_board(_props, _weekly, seed=SEED)


def best_ev_by_player(scan: pd.DataFrame) -> dict[str, dict]:
    """Mejor prop +EV de cada jugador: {player: {ev_pct, fair_odds, label}}."""
    if scan.empty or "ev_pct" not in scan:
        return {}
    plus = scan[scan["is_plus_ev"].fillna(False).astype(bool)].sort_values("ev_pct", ascending=False)
    out = {}
    for r in plus.drop_duplicates("player").itertuples(index=False):
        if r.market in op.YES_NO_MARKETS:
            label = f"{MARKET_LABEL.get(r.market, '')} · {'Sí' if r.outcome == 'Over' else 'No'} @ {fmt_odds(r.decimal_odds)}"
        else:
            label = f"{r.outcome} {r.line:g} {MARKET_LABEL.get(r.market, '')}"
        out[r.player] = {"ev_pct": r.ev_pct, "fair_odds": r.fair_odds, "label": label}
    counts = plus.groupby("player").size().to_dict()
    for p in out:
        out[p]["n_plus"] = int(counts.get(p, 0))
    return out


def _latest_by_player(weekly: pd.DataFrame) -> pd.DataFrame:
    return weekly.sort_values(["season", "week"]).groupby("name_key").tail(1).set_index("name_key")


def game_players(props: pd.DataFrame, weekly: pd.DataFrame, season: int,
                 home: str, away: str) -> pd.DataFrame:
    """TODOS los jugadores con props del partido. Sin props: plantilla por uso reciente."""
    latest = _latest_by_player(weekly)
    rows = []
    if not props.empty and props["player"].notna().any():
        for player, g in props.dropna(subset=["player"]).groupby("player"):
            key = dp.normalize_name(player)
            info = latest.loc[key] if key in latest.index else None
            rows.append({
                "player": player, "name_key": key,
                "position": info["position"] if info is not None else "—",
                "team": info["recent_team"] if info is not None else "—",
                "headshot": info.get("headshot_url") if info is not None else None,
                "markets": [grp for grp in GROUP_ORDER
                            if set(GROUPS[grp]["markets"]) & set(g["market"])],
                "has_odds": True})
    else:
        recent = weekly[(weekly["season"] == season) & weekly["recent_team"].isin([home, away])
                        & weekly["position"].isin(POSITIONS)]
        last_weeks = recent.groupby("recent_team")["week"].transform(lambda w: w >= w.max() - 3)
        recent = recent[last_weeks].fillna({"attempts": 0, "carries": 0, "targets": 0})
        usage = recent.assign(usage=np.select(
            [recent["position"] == "QB", recent["position"] == "RB"],
            [recent["attempts"], recent["carries"] + recent["targets"]], recent["targets"]))
        agg = usage.groupby(["name_key", "recent_team", "position"], as_index=False)["usage"].sum()
        agg = agg[agg["name_key"].map(lambda k: latest.loc[k, "recent_team"] if k in latest.index else None)
                  == agg["recent_team"]]
        for (team, pos), g in agg.groupby(["recent_team", "position"]):
            for r in g.nlargest(FALLBACK_DEPTH[pos], "usage").itertuples():
                info = latest.loc[r.name_key]
                rows.append({"player": info["player_display_name"], "name_key": r.name_key,
                             "position": pos, "team": team, "headshot": info.get("headshot_url"),
                             "markets": POSITION_GROUPS[pos], "has_odds": False})
    df = pd.DataFrame(rows, columns=["player", "name_key", "position", "team", "headshot",
                                     "markets", "has_odds"])
    order = {p: i for i, p in enumerate(POSITIONS)}
    return df.assign(_o=df["position"].map(order).fillna(9)).sort_values(["_o", "team", "player"]) \
             .drop(columns="_o").reset_index(drop=True)


def player_lines(props: pd.DataFrame, player: str, group: str) -> pd.DataFrame:
    """Escalera consolidada de un grupo (principal + alterno): line | over | under
    (cuota decimal mediana entre casas)."""
    if props.empty:
        return pd.DataFrame(columns=["line", "over", "under"])
    markets = GROUPS[group]["markets"]
    g = props[(props["player"] == player) & props["market"].isin(markets)].dropna(subset=["line"])
    if g.empty:
        return pd.DataFrame(columns=["line", "over", "under"])
    pivot = g.pivot_table(index="line", columns="outcome", values="decimal_odds", aggfunc="median")
    return (pivot.rename(columns={"Over": "over", "Under": "under"})
            .reindex(columns=["over", "under"]).round(2).reset_index().sort_values("line")
            .reset_index(drop=True))


def reference_lines(values: pd.Series, group: str) -> pd.DataFrame:
    """Sin cuotas en vivo: líneas de referencia alrededor de la mediana L10."""
    spec = GROUPS[group]
    if spec.get("yes_no"):
        lines = [0.5]
    else:
        med = float(np.median(values)) if len(values) else 0
        step = 1 if spec["model"] == "count" else max(5, int(round(med * 0.1 / 5) * 5))
        base = math.floor(med) + 0.5
        lines = sorted({round(base + k * step, 1) for k in range(-3, 4) if base + k * step > 0})
    return pd.DataFrame({"line": lines, "over": np.nan, "under": np.nan})


def default_line(lines: pd.DataFrame, values: pd.Series) -> float:
    if lines["over"].notna().any():
        return float(lines.loc[(lines["over"] - 1.91).abs().idxmin(), "line"])
    med = float(np.median(values)) if len(values) else lines["line"].median()
    return float(lines.loc[(lines["line"] - med).abs().idxmin(), "line"])


@st.cache_data(ttl=1800, show_spinner=False)
def player_sims(name_key: str, stat: str, values: tuple) -> np.ndarray:
    """Simulación Monte Carlo del jugador (misma semilla que el escáner de props)."""
    return pe.simulate_stat(np.asarray(values, dtype=float), stat, seed=SEED)


@st.cache_resource(ttl=1800, show_spinner=False)
def first_td_sims(_weekly: pd.DataFrame, _log: pd.DataFrame, name_key: str,
                  home: str, away: str) -> np.ndarray:
    return pe.first_td_sims(_weekly, _log, home, away, seed=SEED)


def make_player_leg(player: str, market: str, line: float, odds: float, side: str,
                    game_id: str, weekly: pd.DataFrame, market_prob: float | None = None) -> bb.Leg | None:
    props = pd.DataFrame([{"event_id": game_id, "market": market, "player": player,
                           "outcome": side.capitalize(), "line": line, "decimal_odds": odds,
                           "market_prob": np.nan if market_prob is None else market_prob}])
    legs = bb.build_player_legs(props, weekly, seed=SEED)
    return legs[0] if legs else None


@st.cache_resource(ttl=1800, show_spinner=False)
def slate_parlays(_legs: list[bb.Leg], _weekly: pd.DataFrame, cache_key: str, mode: str = "mixed") -> dict:
    return bb.build_parlays(_legs, weekly=_weekly, top_n=4, seed=SEED, mode=mode)


@st.cache_resource(ttl=1800, show_spinner=False)
def player_legs_cached(_props: pd.DataFrame, _weekly: pd.DataFrame, cache_key: str) -> list[bb.Leg]:
    """Patas para las combinadas sugeridas: solo líneas principales/fijas (cuota de consenso)."""
    return bb.build_player_legs(_props, _weekly, seed=SEED, main_only=True)


def slate_props_live(event_map: dict, fetch_missing: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Props de los partidos pendientes SIN gastar créditos por defecto:
    snapshot del scheduler + copias en disco (partidos ya abiertos, < 30 min).

    Con `fetch_missing=True` descarga los que falten (~17 créditos por partido).
    Devuelve (props, game_ids_sin_props_cargados).
    """
    frames, missing = [], []
    snap = slate_props(event_map)
    have = set(snap["event_id"]) if not snap.empty else set()
    if not snap.empty:
        frames.append(snap)
    for game_id, api_id in event_map.items():
        if game_id in have:
            continue
        cached = op.cached_event_props(api_id)
        if cached is None and fetch_missing and ODDS_API_ENABLED:
            cached = game_props(game_id, api_id)
        if cached is None:
            missing.append(game_id)
        elif not cached.empty:
            frames.append(cached.assign(event_id=game_id))
    props = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return props, missing


def require_password() -> None:
    """Pantalla de acceso. Solo se activa si APP_PASSWORD está definido (Secrets / .env).
    Protege tus créditos de The-Odds-API cuando la app es pública."""
    from config import APP_PASSWORD
    if not APP_PASSWORD or st.session_state.get("auth_ok"):
        return
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid, st.container(key="loginbox"):
        st.markdown('<div style="text-align:center;font-size:2.4rem">🏈</div>'
                    '<div style="text-align:center;font-weight:800;font-size:1.35rem">NFL +EV Terminal</div>'
                    '<div class="muted" style="text-align:center;margin-bottom:10px">Acceso privado</div>',
                    unsafe_allow_html=True)
        with st.form("login", border=False):
            pwd = st.text_input("Contraseña", type="password", placeholder="Contraseña de acceso")
            ok = st.form_submit_button("Entrar", type="primary", width="stretch")
        if ok:
            if hmac.compare_digest(pwd.encode(), APP_PASSWORD.encode()):
                st.session_state["auth_ok"] = True
                st.rerun()
            st.error("Contraseña incorrecta.")
    st.stop()


def pool_note(player_pool: list, team_pool: list, props: pd.DataFrame) -> str:
    """Leyenda del pool de las combinadas sugeridas (solo líneas principales fijas)."""
    n_props = len({l.unique_key for l in player_pool})
    n_games = props["event_id"].nunique() if not props.empty else 0
    return (f"Pool: {n_props} props de jugador (Líneas principales fijas Draftea) + {len(team_pool)} líneas "
            f"de equipo · {n_games} partido(s) con props. Sin escaleras alternas: cada prop es la línea "
            "base de la casa con su cuota de consenso. Las combinadas 🔥 SGP usan la cuota teórica: "
            "ajústala en la boleta con la cuota real.")


def quota_text() -> str | None:
    q = op.QUOTA.get("remaining")
    return f"Créditos The-Odds-API: {q}" if q is not None else None


# =========================================================================== #
# Boleta (session_state)
# =========================================================================== #
def slip() -> list[bb.Leg]:
    return st.session_state.setdefault("slip", [])


def add_to_slip(legs: list[bb.Leg]) -> None:
    current = slip()
    for leg in legs:
        if any(l.label == leg.label for l in current):
            continue
        replaced = [l for l in current if l.unique_key == leg.unique_key]
        for old in replaced:
            current.remove(old)
            st.toast(f"Reemplazada: {old.label}", icon="🔁")
        current.append(leg)
    st.toast(f"Boleta: {len(current)} selección(es)", icon="🎟️")


def add_player_selection(player: str, market: str, line: float, odds: float, side: str,
                         game_id: str, weekly: pd.DataFrame, market_prob: float | None = None) -> None:
    """Callback: corre antes del rerender, así la boleta lateral se actualiza al instante."""
    leg = make_player_leg(player, market, line, odds, side, game_id, weekly, market_prob)
    if leg:
        add_to_slip([leg])
    else:
        st.toast("No se pudo construir la selección (Realism Guard o sin historial).", icon="🚫")


def remove_from_slip(label: str) -> None:
    st.session_state["slip"] = [l for l in slip() if l.label != label]


def clear_slip() -> None:
    st.session_state["slip"] = []


def slip_signature(legs: list[bb.Leg]) -> str:
    return str(abs(hash(tuple(l.label for l in legs))))


def evaluate_slip(legs: list[bb.Leg], weekly: pd.DataFrame) -> dict:
    """Probabilidad conjunta Monte Carlo (cópula correlacionada) de la boleta."""
    labels = tuple(l.label for l in legs)
    memo = st.session_state.get("_slip_eval")
    if memo and memo[0] == labels:
        return memo[1]
    if len(legs) == 1:
        res = {"joint_prob_pct": round(legs[0].prob * 100, 2), "independent_prob_pct": None,
               "coherence_issues": []}
    else:
        r = bb.evaluate_parlay(legs, weekly=weekly, seed=SEED)
        res = {k: r[k] for k in ("joint_prob_pct", "independent_prob_pct", "coherence_issues")}
    st.session_state["_slip_eval"] = (labels, res)
    return res


# =========================================================================== #
# Navegación
# =========================================================================== #
def open_game(game_id: str) -> None:
    st.session_state.current_view = "player_detail"
    st.session_state.selected_game = game_id
    st.session_state.selected_player_id = None


def back_to_games() -> None:
    st.session_state.current_view = "games"
    st.session_state.selected_player_id = None


def player_id(name_key: str, team: str) -> str:
    """ID estable del jugador dentro del partido (nombre normalizado + equipo)."""
    return f"{name_key}|{team}"


def select_player(pid: str) -> None:
    """Focus View: guarda el jugador y oculta el resto del catálogo."""
    st.session_state.selected_player_id = pid


def clear_player() -> None:
    """Vuelve al catálogo completo de jugadores del partido."""
    st.session_state.selected_player_id = None


# =========================================================================== #
# Estilos
# =========================================================================== #
GLASS = (f"background: linear-gradient(145deg, rgba(22,27,34,.78), rgba(13,17,23,.62));"
         f"backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);"
         f"border: 1px solid {BORDER}; border-radius: 16px;"
         f"box-shadow: 0 10px 30px rgba(0,0,0,.40), inset 0 1px 0 rgba(255,255,255,.03);")


def inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

.stApp {{
  background:
    radial-gradient(1200px 500px at 15% -10%, rgba(0,255,135,.05), transparent 60%),
    radial-gradient(900px 400px at 95% 0%, rgba(255,46,147,.05), transparent 60%),
    {BG_DEEP};
}}
.stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label, .stApp li,
.stApp [data-testid="stMarkdownContainer"], .stApp button p, .stApp input {{
  font-family: 'Inter', system-ui, sans-serif;
}}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {BG_PANEL}, {BG_DEEP});
  border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
hr {{ border-color: {BORDER} !important; }}
.mono {{ font-family: 'JetBrains Mono', ui-monospace, monospace; }}

/* ---------- Glass cards ---------- */
div[class*="st-key-card_"], div[class*="st-key-parlay_"], div[class*="st-key-terminal"],
div[class*="st-key-pcard_"], div[class*="st-key-pcardsel_"], div[class*="st-key-slipleg_"],
div[class*="st-key-gamehdr"], div[class*="st-key-evalbox"] {{ {GLASS} padding: 14px 16px; }}
div[class*="st-key-card_"]:hover, div[class*="st-key-pcard_"]:hover {{
  border-color: {BORDER_HI}; box-shadow: 0 14px 34px rgba(0,0,0,.5); }}
div[class*="st-key-pcardsel_"] {{ border-color: {GREEN};
  box-shadow: 0 0 0 1px rgba(0,255,135,.35), 0 0 24px rgba(0,255,135,.15); }}
div[class*="st-key-slipleg_"] {{ padding: 8px 10px; border-radius: 12px; }}
div[class*="st-key-slipleg_sgp"] {{ border-left: 3px solid {PINK}; }}

/* ---------- Marcador proyectado ---------- */
div[class*="st-key-score_"] button {{
  background: radial-gradient(120% 140% at 50% 0%, rgba(0,255,135,.10), transparent 60%),
              linear-gradient(180deg, #11161D, {BG_DEEP});
  border: 1px solid rgba(0,255,135,.55); border-radius: 16px; min-height: 88px;
  box-shadow: 0 0 22px rgba(0,255,135,.14), inset 0 1px 0 rgba(255,255,255,.05);
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}}
div[class*="st-key-score_"] button p {{
  font-family: 'JetBrains Mono', monospace !important; font-size: 2.15rem !important;
  font-weight: 700; letter-spacing: 1px; color: #FFFFFF;
}}
div[class*="st-key-score_"] button:hover {{ transform: translateY(-2px); border-color: {GREEN};
  box-shadow: 0 0 30px rgba(0,255,135,.30); }}
@keyframes pulse-ring {{
  0% {{ box-shadow: 0 0 0 0 rgba(0,255,135,.45); }}
  70% {{ box-shadow: 0 0 0 9px rgba(0,255,135,0); }}
  100% {{ box-shadow: 0 0 0 0 rgba(0,255,135,0); }}
}}
.pulse {{ display: inline-block; animation: pulse-ring 2.2s infinite; color: {GREEN};
  background: rgba(0,255,135,.07); border: 1px solid rgba(0,255,135,.45); border-radius: 999px;
  padding: 3px 12px; font-size: .72rem; font-weight: 700; letter-spacing: .2px; }}

/* ---------- Botones de cuota ---------- */
div[class*="st-key-odds_"] button, div[class*="st-key-oddsev_"] button {{
  background: linear-gradient(180deg, #161B22 0%, #0D1117 100%); border: 1px solid {BORDER_HI};
  border-radius: 12px; min-height: 48px; position: relative; overflow: visible;
  transition: border-color .15s ease, box-shadow .15s ease;
}}
div[class*="st-key-odds_"] button p, div[class*="st-key-oddsev_"] button p {{
  font-size: .84rem !important; font-weight: 600; }}
div[class*="st-key-odds_"] button:hover {{ border-color: {MUTED}; }}
div[class*="st-key-oddsev_"] button {{ border-color: rgba(0,255,135,.55);
  background: linear-gradient(180deg, rgba(0,255,135,.08) 0%, #0D1117 100%);
  box-shadow: 0 0 14px rgba(0,255,135,.12); }}
div[class*="st-key-oddsev_"] button:hover {{ border-color: {GREEN}; box-shadow: 0 0 20px rgba(0,255,135,.28); }}
div[class*="st-key-oddsev_"] button::after {{
  content: "+EV"; position: absolute; top: -9px; right: 10px; padding: 1px 8px;
  background: {GREEN}; color: #03140B; font-size: .62rem; font-weight: 800; border-radius: 999px;
  box-shadow: 0 0 12px rgba(0,255,135,.7); letter-spacing: .4px;
}}

/* ---------- Botones de jugador / acciones ---------- */
div[class*="st-key-pl_"] button {{ background: linear-gradient(180deg, #161B22, #0D1117);
  border: 1px solid {BORDER_HI}; border-radius: 10px; min-height: 34px; }}
div[class*="st-key-pl_"] button:hover {{ border-color: {GREEN}; color: {GREEN}; }}
div[class*="st-key-back"] button {{ border-radius: 999px; border: 1px solid {BORDER_HI};
  background: {BG_PANEL}; }}
div[class*="st-key-back"] button:hover {{ border-color: {GREEN}; color: {GREEN}; }}
/* Focus View: botón grande para volver al catálogo de jugadores */
.st-key-playerback {{ position: sticky; top: 3.2rem; z-index: 50; margin-bottom: 4px; }}
.st-key-playerback button {{ min-height: 52px; border-radius: 14px !important; font-weight: 800 !important;
  font-size: 1.02rem !important; color: #04130B !important; border: 1px solid {GREEN} !important;
  background: linear-gradient(135deg, {GREEN}, #00C46A) !important;
  box-shadow: 0 0 22px rgba(0,255,135,.28); }}
.st-key-playerback button:hover {{ box-shadow: 0 0 30px rgba(0,255,135,.45); filter: brightness(1.05); }}

/* ---------- Inputs, pills, métricas ---------- */
[data-testid="stTextInput"] [data-baseweb="input"], [data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
  background: {BG_PANEL}; border: 1px solid {BORDER_HI}; border-radius: 12px; }}
[data-testid="stTextInput"] [data-baseweb="input"]:focus-within {{ border-color: {GREEN};
  box-shadow: 0 0 0 1px rgba(0,255,135,.35); }}
[data-testid="stButtonGroup"] button {{ border-radius: 999px !important; border: 1px solid {BORDER_HI};
  background: {BG_PANEL}; padding: 2px 14px; }}
[data-testid="stButtonGroup"] button[kind*="Active"], [data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[aria-pressed="true"] {{ border-color: {GREEN} !important;
  color: {GREEN} !important; background: rgba(0,255,135,.10) !important; }}
[data-testid="stMetric"] {{ {GLASS} padding: 10px 14px; border-radius: 14px; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: .72rem !important; font-weight: 700;
  letter-spacing: .6px; text-transform: uppercase; }}
[data-testid="stMetricValue"] {{ color: #FFFFFF; font-family: 'JetBrains Mono', monospace; font-weight: 700; }}
[data-testid="stExpander"] details {{ {GLASS} }}
[data-testid="stExpander"] summary p {{ font-weight: 700; }}

/* ---------- Tipos ---------- */
.hero {{ {GLASS} padding: 18px 22px; display: flex; justify-content: space-between;
  align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 14px; }}
.hero h1 {{ margin: 0; font-size: 1.9rem; font-weight: 800; letter-spacing: -.5px;
  background: linear-gradient(90deg, #FFFFFF 0%, {GREEN} 55%, {PINK} 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent; }}
.hero .sub {{ color: {MUTED}; font-size: .82rem; margin-top: 2px; }}
.chips {{ display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end; }}
.chip {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .72rem;
  font-weight: 600; border: 1px solid {BORDER_HI}; color: {TEXT}; background: rgba(13,17,23,.7); }}
.chip a {{ color: {GREEN}; text-decoration: none; }}
.chip-green {{ border-color: rgba(0,255,135,.5); color: {GREEN}; }}
.chip-pink {{ border-color: rgba(255,46,147,.5); color: {PINK}; }}
.chip-live::before {{ content: "●"; color: {GREEN}; margin-right: 5px; }}
.team {{ text-align: center; font-weight: 800; font-size: 1.05rem; color: {TEXT}; }}
.team img {{ width: 64px; height: 64px; display: block; margin: 0 auto 4px;
  filter: drop-shadow(0 4px 10px rgba(0,0,0,.6)); }}
.team .win {{ color: {MUTED}; font-size: .72rem; font-weight: 600; }}
.team .win.fav {{ color: {GREEN}; }}
.muted {{ color: {MUTED}; font-size: .78rem; }}
.center {{ text-align: center; }}
.badge {{ display: inline-block; padding: 3px 11px; border-radius: 999px; font-weight: 700;
  font-size: .76rem; border: 1px solid; }}
.badge-BAJO {{ color: {GREEN}; background: rgba(0,255,135,.10); }}
.badge-MODERADO {{ color: {YELLOW}; background: rgba(245,197,66,.10); }}
.badge-ALTO {{ color: {PINK}; background: rgba(255,46,147,.10); }}
.badge-NONE {{ color: {MUTED}; background: transparent; border-color: {BORDER_HI}; }}
.evb {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: .72rem;
  font-weight: 700; white-space: nowrap; }}
.evb-pos {{ color: {GREEN}; background: rgba(0,255,135,.10); border: 1px solid rgba(0,255,135,.55);
  box-shadow: 0 0 12px rgba(0,255,135,.20); }}
.evb-neg {{ color: {MUTED}; border: 1px solid {BORDER_HI}; }}
.st-key-btmetrics [data-testid="stMetric"], .st-key-perfmetrics [data-testid="stMetric"] {{
  background: #161B22 !important; border: 1px solid #21262D !important; border-radius: 8px !important;
  padding: 10px 12px !important; min-height: 92px; text-align: center; }}
.st-key-btmetrics [data-testid="stMetricLabel"] p, .st-key-perfmetrics [data-testid="stMetricLabel"] p {{
  font-size: .72rem !important; white-space: nowrap; }}
.evb-warn {{ color: #FFB020; background: rgba(255,176,32,.08); border: 1px solid rgba(255,176,32,.45); }}
.evb-none {{ color: {MUTED}; border: 1px dashed {BORDER_HI}; }}
.ev-cap {{ text-align: center; font-size: .7rem; margin-top: -4px; }}
.ev-cap.pos {{ color: {GREEN}; }} .ev-cap.neg {{ color: {MUTED}; }}
.leg-line {{ font-size: .86rem; margin: 3px 0; line-height: 1.35; }}
.leg-line b {{ color: {GREEN}; font-family: 'JetBrains Mono', monospace; }}
.pc {{ display: flex; gap: 10px; align-items: center; margin-bottom: 8px; }}
.pc img {{ width: 46px; height: 46px; border-radius: 50%; object-fit: cover; background: {BG_PANEL};
  border: 1px solid {BORDER_HI}; }}
.pc .nm {{ font-weight: 800; font-size: .92rem; line-height: 1.15; }}
.pc .meta {{ color: {MUTED}; font-size: .72rem; font-weight: 600; }}
.term-head {{ display: flex; gap: 14px; align-items: center; }}
.term-head img {{ width: 76px; height: 76px; border-radius: 50%; object-fit: cover;
  border: 2px solid rgba(0,255,135,.5); background: {BG_PANEL}; }}
.term-title {{ font-size: 1.7rem; font-weight: 800; margin: 0; letter-spacing: -.4px; }}
.lines {{ width: 100%; border-collapse: separate; border-spacing: 0 4px; font-size: .84rem; }}
.lines th {{ color: {MUTED}; font-size: .68rem; text-transform: uppercase; letter-spacing: .6px;
  text-align: left; padding: 2px 10px; font-weight: 700; }}
.lines td {{ background: rgba(13,17,23,.75); padding: 7px 10px; border-top: 1px solid {BORDER};
  border-bottom: 1px solid {BORDER}; vertical-align: middle; }}
.lines td:first-child {{ border-left: 1px solid {BORDER}; border-radius: 10px 0 0 10px;
  font-family: 'JetBrains Mono', monospace; font-weight: 700; }}
.lines td:last-child {{ border-right: 1px solid {BORDER}; border-radius: 0 10px 10px 0; }}
.lines tr.active td {{ border-color: rgba(0,255,135,.6); background: rgba(0,255,135,.06); }}
.lines .odd {{ font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #FFF; }}
.lines .mc {{ color: {MUTED}; font-size: .74rem; margin: 0 6px; }}
.lines tr.bad td {{ opacity: .45; }}
.big-pick {{ font-size: 1.25rem; font-weight: 800; }}
.big-pick .mono {{ color: {GREEN}; }}
.sgp-alert {{ border: 1px solid rgba(255,46,147,.6); background: linear-gradient(135deg,
  rgba(255,46,147,.16), rgba(255,46,147,.04)); color: #FFD3E8; border-radius: 12px;
  padding: 9px 12px; font-weight: 700; font-size: .85rem; box-shadow: 0 0 18px rgba(255,46,147,.18); }}
.sgp-alert .muted {{ color: #F5A3C9; font-weight: 500; }}
.theo {{ {GLASS} padding: 10px 12px; border-radius: 12px; font-size: .8rem; }}
.theo .formula {{ font-family: 'JetBrains Mono', monospace; color: {TEXT}; font-size: .9rem; }}
.payout {{ {GLASS} padding: 12px; border-radius: 12px; text-align: center; }}
.payout .big {{ font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; }}
.payout .big .arrow {{ color: {MUTED}; margin: 0 6px; }}
.payout .big .win {{ color: {GREEN}; }}
.verdict-pos {{ color: {GREEN}; font-weight: 800; }}
.verdict-neg {{ color: {PINK}; font-weight: 800; }}
.tier-h {{ margin: 0 0 2px 0; font-weight: 800; }}
.slip-title {{ font-size: 1.25rem; font-weight: 800; margin-bottom: 4px; }}

/* ---------- Edición manual arriba + escaleras compactas ---------- */
div[class*="st-key-editbox"] {{ {GLASS} padding: 12px 16px; border-color: rgba(0,255,135,.35); }}
.edit-title {{ font-weight: 800; font-size: .95rem; margin-bottom: 2px; color: {GREEN}; }}
.edit-hint {{ padding-bottom: 10px; font-size: .78rem; }}

/* ---------- Ficha del jugador: espaciado y grids simétricos ---------- */
.st-key-catpills, .st-key-grppills {{ margin-bottom: 12px !important; }}
.st-key-catpills [data-testid="stButtonGroup"], .st-key-grppills [data-testid="stButtonGroup"] {{
  display: flex; flex-wrap: wrap; gap: 6px; }}
.st-key-l10grid, .st-key-evalmetrics {{ margin-bottom: 12px !important; }}
.st-key-l10grid [data-testid="stHorizontalBlock"],
.st-key-evalmetrics [data-testid="stHorizontalBlock"] {{ align-items: stretch !important; gap: 10px !important; }}
.st-key-l10grid [data-testid="stColumn"], .st-key-evalmetrics [data-testid="stColumn"] {{
  display: flex; flex-direction: column; }}
.st-key-l10grid [data-testid="stColumn"] > div, .st-key-evalmetrics [data-testid="stColumn"] > div {{
  flex: 1; }}
.st-key-l10grid [data-testid="stMetric"], .st-key-evalmetrics [data-testid="stMetric"] {{
  background: {BG_CARD} !important; backdrop-filter: none; box-shadow: none !important;
  border: 1px solid #21262D !important; border-radius: 8px !important; padding: 10px 12px !important;
  height: 100%; min-height: 92px; display: flex; flex-direction: column; justify-content: center; }}
.st-key-evalmetrics [data-testid="stMetric"] {{ align-items: center; text-align: center; }}
.st-key-evalmetrics [data-testid="stMetric"] > div {{ width: 100%; justify-content: center; }}
.st-key-evalmetrics [data-testid="stMetricLabel"], .st-key-evalmetrics [data-testid="stMetricDelta"] {{
  justify-content: center; }}
.st-key-l10grid [data-testid="stMetricValue"], .st-key-evalmetrics [data-testid="stMetricValue"] {{
  font-size: clamp(1.05rem, 1.6vw, 1.45rem) !important; white-space: nowrap; overflow: visible; }}
.st-key-l10grid [data-testid="stMetricLabel"] p, .st-key-evalmetrics [data-testid="stMetricLabel"] p {{
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: .72rem !important; }}
div[class*="st-key-evalbox"] .big-pick {{ margin-bottom: 12px; }}
.verdict-row {{ margin: 0 0 12px; line-height: 1.5; }}
.st-key-addrow {{ display: flex; justify-content: flex-start; padding-top: 2px; }}
.st-key-addrow button {{ min-width: 220px; border-radius: 10px !important; }}

/* ---------- 🎯 Línea Más Probable (moda Monte Carlo) ---------- */
.modecard {{ display: flex; align-items: center; gap: 18px; flex-wrap: wrap; padding: 12px 16px;
  margin: 0 0 12px; border: 1px solid #21262D; border-left: 3px solid {GREEN}; border-radius: 8px;
  background: linear-gradient(135deg, rgba(0,255,135,.06), rgba(13,17,23,.6)); }}
.modecard .mc-left {{ min-width: 220px; }}
.modecard .k {{ color: {MUTED}; font-size: .72rem; font-weight: 700; letter-spacing: .5px; text-transform: uppercase; }}
.modecard .v {{ font-family: 'JetBrains Mono', monospace; font-size: 1.7rem; font-weight: 800; color: #FFF;
  white-space: nowrap; line-height: 1.2; }}
.modecard .v .u {{ font-size: .95rem; color: {MUTED}; font-weight: 600; }}
.modecard .mc-right {{ flex: 1 1 260px; display: flex; align-items: center; gap: 14px;
  border-left: 1px solid #21262D; padding-left: 16px; }}
.modecard .delta {{ font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 800;
  white-space: nowrap; }}
.ladder {{ max-height: 250px; overflow-y: auto; padding-right: 4px; }}
.ladder::-webkit-scrollbar {{ width: 6px; }}
.ladder::-webkit-scrollbar-thumb {{ background: {BORDER_HI}; border-radius: 6px; }}
.ladder thead th {{ position: sticky; top: 0; background: {BG_PANEL}; z-index: 1; }}

/* ---------- Tablero de anotador (Anytime / Primer TD) ---------- */
.tdboard {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 6px 0 10px; }}
.tdcell {{ {GLASS} flex: 1 1 220px; padding: 10px 14px; border-radius: 14px; }}
.tdcell .k {{ color: {MUTED}; font-size: .7rem; font-weight: 700; letter-spacing: .5px; text-transform: uppercase; }}
.tdcell .v {{ font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #FFF;
  white-space: nowrap; }}

/* ---------- Métricas de tarjetas de combinadas: sin truncar ---------- */
.pstats {{ display: flex; gap: 6px; margin: 8px 0 4px; }}
.pstat {{ flex: 1 1 0; min-width: 0; padding: 6px 4px; text-align: center; background: rgba(13,17,23,.8);
  border: 1px solid {BORDER}; border-radius: 10px; }}
.pstat .k {{ color: {MUTED}; font-size: .66rem; font-weight: 700; letter-spacing: .6px;
  text-transform: uppercase; }}
.pstat .v {{ font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #FFFFFF;
  font-size: clamp(0.9rem, 1.2vw, 1.2rem); white-space: nowrap; overflow: visible; text-overflow: clip; }}
.pstat .v.win {{ color: {GREEN}; }}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {{ font-size: clamp(1rem, 1.4vw, 1.35rem); }}
[data-testid="stMetricValue"] > div {{ overflow: visible; text-overflow: clip; white-space: nowrap; }}

/* ========== Boleta (sidebar): ancho garantizado y bloques uniformes ========== */
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ padding-left: 4px; padding-right: 4px; }}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding: 1rem 10px 2rem; }}
/* Bloques: Bankroll, selección, métricas, calculadora, vaciar */
.st-key-slipbank, div[class*="st-key-slipleg_"], .st-key-slipmetrics, .st-key-slipadj,
.st-key-slippay, .st-key-slipclear {{
  background: {BG_CARD} !important; backdrop-filter: none !important; box-shadow: none !important;
  border: 1px solid #21262D !important; border-radius: 8px !important; padding: 10px !important;
  margin-bottom: 0; }}
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {{ gap: 10px !important; }}
div[class*="st-key-slipleg_sgp"] {{ border-left: 3px solid {PINK} !important; }}
[data-testid="stSidebar"] .theo, [data-testid="stSidebar"] .payout {{
  background: transparent; backdrop-filter: none; box-shadow: none; border: none; padding: 0; }}
/* Fila de selección: texto + botón ❌ cuadrado fijo */
div[class*="st-key-slipleg_"] [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap !important;
  gap: 8px !important; align-items: center !important; }}
div[class*="st-key-slipleg_"] [data-testid="stColumn"]:first-child {{ min-width: 0 !important; flex: 1 1 auto !important; }}
div[class*="st-key-slipleg_"] [data-testid="stColumn"]:last-child {{
  flex: 0 0 36px !important; width: 36px !important; min-width: 36px !important; max-width: 36px !important;
  display: flex; justify-content: center; }}
div[class*="st-key-slipleg_"] [data-testid="stColumn"]:last-child button {{
  width: 34px !important; height: 34px !important; min-width: 34px !important; min-height: 34px !important;
  padding: 0 !important; border-radius: 8px !important; border: 1px solid #30363D !important;
  background: rgba(255,46,147,.08) !important; display: flex; align-items: center; justify-content: center; }}
div[class*="st-key-slipleg_"] [data-testid="stColumn"]:last-child button p {{
  font-size: .8rem !important; line-height: 1 !important; margin: 0 !important;
  white-space: nowrap !important; overflow: visible !important; text-overflow: clip !important; }}
div[class*="st-key-slipleg_"] [data-testid="stColumn"]:last-child button:hover {{
  border-color: {PINK} !important; background: rgba(255,46,147,.18) !important; }}
.leg-line {{ overflow-wrap: anywhere; }}
/* Métricas de la boleta: títulos completos, sin "PROB. CONJ..." */
.st-key-slipmetrics {{ padding: 10px 6px !important; }}
.st-key-slipmetrics [data-testid="stHorizontalBlock"] {{ gap: 6px !important; flex-wrap: nowrap !important; }}
.st-key-slipmetrics [data-testid="stColumn"] {{ min-width: 0 !important; }}
.st-key-slipmetrics [data-testid="stMetric"] {{
  background: rgba(13,17,23,.85) !important; backdrop-filter: none; box-shadow: none !important;
  border: 1px solid #21262D !important; border-radius: 8px !important; padding: 10px 3px !important;
  min-height: 82px; display: flex; flex-direction: column; justify-content: center; align-items: center;
  text-align: center; }}
.st-key-slipmetrics [data-testid="stMetric"] > div, .st-key-slipmetrics [data-testid="stMetricLabel"],
.st-key-slipmetrics [data-testid="stMetricValue"], .st-key-slipmetrics [data-testid="stMetricDelta"] {{
  justify-content: center; width: 100%; }}
.st-key-slipmetrics [data-testid="stMetricLabel"], .st-key-slipmetrics [data-testid="stMetricLabel"] > div,
.st-key-slipmetrics [data-testid="stMetricLabel"] p {{
  font-size: 0.72rem !important; letter-spacing: 0.4px !important; font-weight: 600 !important;
  white-space: nowrap !important;
  overflow: visible !important; text-overflow: clip !important; max-width: none !important; }}
.st-key-slipmetrics [data-testid="stMetricLabel"], .st-key-evalmetrics [data-testid="stMetricLabel"] {{
  display: flex !important; justify-content: center !important; width: 100% !important; }}
.st-key-slipmetrics [data-testid="stMetricLabel"] > div, .st-key-evalmetrics [data-testid="stMetricLabel"] > div,
.st-key-slipmetrics [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"],
.st-key-evalmetrics [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] {{
  width: auto !important; justify-content: center !important; text-align: center !important; }}
.st-key-slipmetrics [data-testid="stMetricValue"] {{ font-size: 1.15rem !important; }}
.st-key-slipmetrics [data-testid="stMetricDelta"] {{ font-size: .7rem !important; }}
.st-key-slipmetrics .slipbadges {{ margin-top: 8px; }}
.st-key-slipclear button {{ border-radius: 8px !important; }}
.st-key-loginbox {{ {GLASS} padding: 24px 22px; margin-top: 12vh; border-radius: 16px; }}
</style>""", unsafe_allow_html=True)


# =========================================================================== #
# Cabecera
# =========================================================================== #
def render_header(ctx: dict, api_url: str | None) -> None:
    live = "chip-green chip-live" if (ODDS_API_ENABLED or snapshot_key() != "sin-snapshot") else ""
    chips = [chip(f"Temporada {ctx['season']}"), chip(f"Semana {ctx['week']} · {ctx['season_type']}"),
             chip(esc(odds_status()), live)]
    if quota_text():
        chip_cls = "chip-pink" if int(op.QUOTA["remaining"] or 0) < 100 else ""
        chips.append(chip(esc(quota_text()), chip_cls))
    if api_url:
        chips.append(chip(f'API <a href="{api_url}/docs" target="_blank">{esc(api_url)}</a>'))
    st.markdown(f"""<div class="hero"><div><h1>🏈 NFL +EV Terminal</h1>
      <div class="sub">Monte Carlo 10,000 iteraciones · Realism Guard · cópula SGP · Kelly 25%</div></div>
      <div class="chips">{''.join(chips)}</div></div>""", unsafe_allow_html=True)


def render_api_diagnostics() -> None:
    """Explica por qué no llegan cuotas en vivo, en vez de fallar en silencio."""
    import config
    if not ODDS_API_ENABLED:
        detail = f" Error al leer Secrets: `{config.SECRETS_ERROR}`." if config.SECRETS_ERROR else ""
        st.warning("🔑 No se detectó la clave de The-Odds-API, así que se usan las líneas de Vegas de nflverse "
                   "(sin props de jugadores). En local: `ODDS_API_KEY=...` en `.env`. En Streamlit Cloud: "
                   "⋮ → Settings → Secrets → `ODDS_API_KEY = \"tu_clave\"` (con comillas) y reinicia la app."
                   + detail)
    elif op.QUOTA.get("last_error"):
        st.error(f"⚠️ The-Odds-API respondió con un error: {op.QUOTA['last_error']}. "
                 "401 = clave inválida · 429 / OUT_OF_USAGE_CREDITS = sin créditos este mes.")


# =========================================================================== #
# 📈 Rendimiento del modelo (resultados reales)
# =========================================================================== #
@st.cache_data(show_spinner=False)
def _tracking_data(mtime: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    import tracking
    picks = tracking.load_picks()
    return picks, tracking.summarize(picks)


STATUS_LABEL = {"won": "✅ Ganada", "lost": "❌ Perdida", "void": "➖ Anulada", "pending": "⏳ Pendiente"}
BACKTEST_REPORT = Path(__file__).parent / "backtest" / "report.json"
FAMILY_LABEL = {"receptions": "Recepciones", "receiving_yards": "Yds recepción", "rushing_yards": "Yds carrera",
                "passing_yards": "Yds pase", "total_tds": "Anytime TD", "h2h": "Moneyline", "spreads": "Spread",
                "totals": "Total partido", "passing_tds": "TD pase", "first_td": "Primer TD"}
VARIANT_LABEL = {"completo": "Modelo + mercado", "sesgo": "Mercado + corrección de sesgo", "mercado": "Solo mercado"}


@st.cache_data(show_spinner=False)
def _backtest_report(mtime: float) -> dict:
    return json.loads(BACKTEST_REPORT.read_text(encoding="utf-8"))


def prob_method_text(res: dict, family: str | None, mkt_txt: str) -> str:
    """Cómo se obtuvo la probabilidad ajustada, en palabras (según la calibración del backtest)."""
    cal = (pe.CALIBRATION.get("families") or {}).get(family) if res.get("calibrated") else None
    mc = f"Monte Carlo {res['mc_prob_pct']:.1f}%"
    if not cal:
        w = res.get("model_weight") or 0
        return f"{w:.0%} {mc} + {1 - w:.0%} {mkt_txt} (mezcla fija: mercado sin backtest)"
    variant = cal.get("variant")
    if variant == "mercado":
        return f"{mkt_txt} (backtest: el modelo no mejora al mercado aquí; {mc} solo como referencia)"
    if variant == "sesgo":
        return f"{mkt_txt} + corrección de sesgo medido en el backtest ({mc} solo como referencia)"
    return f"{mkt_txt} + {mc} con peso {res.get('model_weight') or 0:.0%} calibrado en el backtest"


def calibration_chart(report: dict) -> go.Figure:
    """Prob. esperada vs frecuencia real (fuera de muestra). La diagonal = calibración perfecta."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Calibración perfecta",
                             line=dict(color="#30363D", width=1), hoverinfo="skip"))
    series = (("calibracion_mc", "Monte Carlo puro", "#8B949E", "dash", "diamond"),
              ("calibracion_ajustada", "Prob. ajustada (la que usa la app)", COLOR_NEON_GREEN, "solid", "circle"))
    for key, name, color, dash, symbol in series:
        rows = report.get(key) or []
        fig.add_trace(go.Scatter(
            x=[r["esperado_pct"] for r in rows], y=[r["real_pct"] for r in rows], name=name,
            mode="lines+markers", line=dict(color=color, width=2, dash=dash),
            marker=dict(size=9, symbol=symbol, color=color, line=dict(color="#0E1117", width=2)),
            customdata=[r["n"] for r in rows],
            hovertemplate=f"<b>{name}</b><br>Esperado %{{x:.1f}}% · Real %{{y:.1f}}%"
                          "<br>%{customdata} líneas<extra></extra>"))
    fig.update_layout(height=340, margin=dict(l=40, r=20, t=10, b=40), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#C9D1D9"), hovermode="closest",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=11)))
    axis = dict(range=[0, 100], gridcolor="#21262D", zeroline=False, ticksuffix="%")
    fig.update_xaxes(title_text="Probabilidad que dijo el modelo", **axis)
    fig.update_yaxes(title_text="Frecuencia real", **axis)
    return fig


def render_backtest() -> None:
    """Resultados del backtest con cuotas históricas (fuera de muestra)."""
    try:
        rep = _backtest_report(BACKTEST_REPORT.stat().st_mtime)
    except (OSError, ValueError):
        st.caption("Sin backtest todavía: `python backtest.py all --budget 8000`.")
        return
    cal = rep.get("calibracion", {})
    single = rep.get("sencillas", {})
    res, val = single.get("resultado", {}), rep.get("validacion", {})
    comp = rep.get("comparacion", {})
    weeks = rep.get("weeks", [])
    st.caption(f"Temporada 2026, semanas {min(weeks)}–{max(weeks)} · {rep.get('n_lineas', 0):,} líneas "
               "principales con la cuota de ~1 h antes del kickoff · cada semana se evaluó con un modelo que "
               "NO la vio (validación fuera de muestra).")
    with st.container(key="btmetrics"):
        c1, c2, c3, c4 = st.columns(4, gap="small")
        c1.metric("Sencillas +EV", f"{res.get('n', 0):,}", f"EV ≥ {single.get('umbral_elegido', 0):g}%",
                  delta_color="off")
        c2.metric("Acierto real", f"{res.get('acierto_pct') or 0:.1f}%",
                  f"esperado {res.get('prob_media_pct') or 0:.1f}%", delta_color="off")
        c3.metric("ROI fuera de muestra", f"{res.get('roi_pct') or 0:+.1f}%",
                  f"{res.get('unidades', 0):+.1f} u", delta_color="off")
        test = val.get("prueba_2a_mitad", {})
        c4.metric("ROI prueba ciega", f"{test.get('roi_pct') or 0:+.1f}%",
                  f"{test.get('n', 0)} apuestas · {val.get('prueba_semanas', '')}", delta_color="off")
    mkt, adj = comp.get("mercado", {}), comp.get("calibrada", {})
    mc = comp.get("monte_carlo", {})
    st.markdown(
        f"<div class='muted'>Precisión (log-loss, menor = mejor): Monte Carlo puro <b>{mc.get('log_loss', 0):.4f}</b>"
        f" · mercado sin vig <b>{mkt.get('log_loss', 0):.4f}</b> · prob. ajustada <b>{adj.get('log_loss', 0):.4f}</b>. "
        "El Monte Carlo solo está sobreconfiado; por eso la app ancla la probabilidad al mercado y solo usa el modelo "
        "donde demostró aportar.</div>", unsafe_allow_html=True)
    st.plotly_chart(calibration_chart(rep), width="stretch", config={"displayModeBar": False})

    validated = set(cal.get("validated_families") or [])
    roi_by_fam = {r["familia"]: r for r in single.get("por_familia", [])}
    rows = []
    for fam, f in (cal.get("families") or {}).items():
        r = roi_by_fam.get(fam, {})
        rows.append({"Mercado": FAMILY_LABEL.get(fam, fam), "Método": VARIANT_LABEL.get(f.get("variant"), ""),
                     "Peso modelo": f"{f.get('model_weight', 0):.0%}", "Líneas": f.get("n"),
                     "Apuestas +EV": r.get("n", 0), "ROI %": r.get("roi_pct"),
                     "Estado": "🟢 Validado" if fam in validated else "⚪ Sin ventaja comprobada"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    side = {r["lado"]: r for r in single.get("por_lado", [])}
    if side:
        st.caption(" · ".join(f"{'Over' if k == 'over' else 'Under' if k == 'under' else 'Equipo'}: "
                              f"{v['n']} apuestas, ROI {v['roi_pct']:+.1f}%" for k, v in side.items()))
    parl = rep.get("combinadas") or {}
    chosen = parl.get("variante_elegida")
    if chosen:
        names = {"BAJO": "🟢 Bajo", "MODERADO": "🟡 Moderado", "ALTO": "🔴 Alto", "TOTAL": "Total"}
        prow = [{"Nivel": names.get(r["nivel"], r["nivel"]), "Combinadas": r["n"], "Acierto %": r["acierto_pct"],
                 "Prob. esperada %": r["prob_media_pct"], "ROI %": r["roi_pct"],
                 "ROI con recorte SGP 15% %": r["roi_recorte_sgp_pct"]}
                for r in parl.get("variantes", {}).get(chosen, [])]
        st.markdown("**Combinadas sugeridas en el backtest** "
                    f"<span class='muted'>(regla elegida: {chosen.replace('_', ' ')})</span>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(prow), hide_index=True, width="stretch")
        off = (cal.get("parlay") or {}).get("disabled_tiers") or []
        if off:
            st.caption("Niveles desactivados por perder en el backtest: " + ", ".join(names.get(t, t) for t in off))
    st.caption("Muestra de una sola temporada: los ROI tienen margen de error de varios puntos. El seguimiento en "
               "vivo y el CLV dirán si la ventaja se mantiene.")


def render_performance() -> None:
    """¿El +EV se materializa? Selecciones registradas 1 h antes de cada jornada y su resultado."""
    import tracking
    try:
        mtime = tracking.PICKS_FILE.stat().st_mtime
    except OSError:
        mtime = None
    with st.expander("📈 Rendimiento del modelo", expanded=False):
        t1, t2 = st.tabs(["🧪 Backtest (cuotas históricas)", "📡 Seguimiento en vivo"])
        with t1:
            render_backtest()
        with t2:
            _render_live(mtime)


def _render_live(mtime: float | None) -> None:
    if mtime is None:
        st.caption("Aún no hay selecciones registradas. Cada jornada, ~1 h antes del kickoff, GitHub Actions "
                   "guarda las sencillas +EV y la mejor combinada de cada riesgo; al día siguiente se "
                   "califican con los resultados reales de nflverse.")
        return
    picks, summary = _tracking_data(mtime)
    total = summary[summary["grupo"] == "TOTAL"]
    if total.empty:
        st.caption("Sin selecciones registradas.")
        return
    t = total.iloc[0]
    settled = int(t["ganadas"] + t["perdidas"])
    with st.container(key="perfmetrics"):
        c1, c2, c3, c4 = st.columns(4, gap="small")
        c1.metric("Calificadas", f"{settled}", f"{int(t['pendientes'])} pendientes", delta_color="off")
        c2.metric("Acierto real", "—" if t["acierto_pct"] is None or pd.isna(t["acierto_pct"])
                  else f"{t['acierto_pct']:.1f}%",
                  None if pd.isna(t["prob_esperada_pct"]) else f"esperado {t['prob_esperada_pct']:.1f}%",
                  delta_color="off")
        c3.metric("Unidades", f"{t['unidades']:+.2f} u")
        c4.metric("ROI", "—" if pd.isna(t["roi_pct"]) else f"{t['roi_pct']:+.1f}%",
                  None if pd.isna(t["ev_esperado_pct"]) else f"EV esperado {t['ev_esperado_pct']:+.1f}%",
                  delta_color="off")
    if settled < 50:
        st.caption(f"⚠️ Muestra pequeña ({settled} calificadas): con menos de ~100–200 selecciones el ROI "
                   "es sobre todo varianza. Lo importante es que el acierto real se acerque al esperado.")
    names = {"SENCILLA": "Sencillas", "BAJO": "🟢 Bajo", "MODERADO": "🟡 Moderado", "ALTO": "🔴 Alto",
             "TOTAL": "Total"}
    st.dataframe(summary.assign(grupo=summary["grupo"].map(names)).rename(columns={
        "grupo": "Grupo", "selecciones": "Sel.", "pendientes": "Pend.", "ganadas": "G", "perdidas": "P",
        "anuladas": "Anul.", "acierto_pct": "Acierto %", "prob_esperada_pct": "Prob. esperada %",
        "ev_esperado_pct": "EV esperado %", "unidades": "Unidades", "roi_pct": "ROI %",
        "clv_medio_pct": "CLV medio %", "clv_positivo_pct": "CLV > 0 %"}),
        hide_index=True, width="stretch")
    recent = picks.sort_values("logged_at", ascending=False).head(25)
    st.dataframe(pd.DataFrame({
        "Fecha": recent["game_date"], "Tipo": recent["tier"].map(names).fillna(recent["tier"]),
        "Selección": recent["label"], "Cuota": pd.to_numeric(recent["decimal_odds"], errors="coerce"),
        "Prob. %": pd.to_numeric(recent["prob_pct"], errors="coerce"),
        "EV %": pd.to_numeric(recent["ev_pct"], errors="coerce"),
        "Resultado": recent["status"].map(STATUS_LABEL).fillna("⏳ Pendiente"),
        "Unidades": pd.to_numeric(recent["profit"], errors="coerce"),
        "CLV %": pd.to_numeric(recent["clv_pct"], errors="coerce")}),
        hide_index=True, width="stretch")
    st.caption("CLV = EV de la cuota tomada medido contra la línea de cierre SIN margen (~10 min antes del "
               "kickoff). Si la cuota no se mueve sale ≈ −3 a −6% (el margen de la casa); por encima de 0% le "
               "ganaste al mercado. Es la señal más rápida: un CLV medio creciente en 50+ selecciones indica "
               "ventaja real aunque el ROI todavía sea ruido.")
    st.caption("Stake plano de 1 unidad por selección, a la cuota registrada ~1 h antes del partido "
               "(consenso de casas; en combinadas del mismo partido la casa real suele pagar menos).")


# =========================================================================== #
# Tarjetas de partidos
# =========================================================================== #
def _team_block(team: str, win_pct: float | None = None) -> str:
    win = ""
    if win_pct is not None:
        win = f'<div class="win {"fav" if win_pct >= 50 else ""}">{win_pct:.0f}% win</div>'
    return f'<div class="team"><img src="{logo_url(team)}" alt="{team}">{team}{win}</div>'


def _market_button_label(leg: bb.Leg) -> str:
    if leg.market == "spreads":
        return f"Spread · {leg.team} {leg.line:+g} · {fmt_odds(leg.decimal_odds)}"
    if leg.market == "h2h":
        return f"Moneyline · {leg.team} gana · {fmt_odds(leg.decimal_odds)}"
    return f"Total · {leg.side.capitalize()} {leg.line:g} · {fmt_odds(leg.decimal_odds)}"


def render_game_card(game, proj: dict | None, best: dict[str, bb.Leg], source: str) -> None:
    gid = game.game_id
    final = not pd.isna(game.home_score)
    with st.container(key=f"card_{gid}"):
        chips = [chip(f"{esc(game.gameday)} · {esc(game.gametime)} ET")]
        if final:
            chips.append(chip("FINAL", "chip-pink"))
        chips.append(chip(esc(source), "chip-green" if source == "The-Odds-API" else ""))
        st.markdown(f'<div class="chips" style="justify-content:space-between">{"".join(chips)}</div>',
                    unsafe_allow_html=True)

        c_away, c_score, c_home = st.columns([1, 3, 1], vertical_alignment="center")
        win_home = proj["home_win_pct"] if proj else None
        c_away.markdown(_team_block(game.away_team, 100 - win_home if proj else None), unsafe_allow_html=True)
        c_home.markdown(_team_block(game.home_team, win_home), unsafe_allow_html=True)
        with c_score:
            label = (f"{game.away_team} {proj['away_pts']} - {game.home_team} {proj['home_pts']}"
                     if proj else f"{game.away_team} @ {game.home_team}")
            st.button(label, key=f"score_{gid}", width="stretch", on_click=open_game, args=(gid,),
                      help="Abrir la analítica de jugadores de este partido")
            sub = '<span class="pulse">👇 Toca el marcador para abrir analítica de jugadores</span>'
            if final:
                sub += (f'<div class="muted" style="margin-top:4px">Real: {game.away_team} '
                        f'{int(game.away_score)} - {game.home_team} {int(game.home_score)}</div>')
            elif proj:
                sub += f'<div class="muted" style="margin-top:4px">Total proyectado {proj["proj_total"]:g}</div>'
            st.markdown(f'<div class="center">{sub}</div>', unsafe_allow_html=True)

        cols = st.columns(3)
        for col, market in zip(cols, ("spreads", "h2h", "totals")):
            leg = best.get(market)
            with col:
                if leg is None:
                    st.button("Sin línea", key=f"odds_{gid}_{market}", disabled=True, width="stretch")
                    continue
                plus = leg.plus_ev
                st.button(_market_button_label(leg), key=f"{'oddsev_' if plus else 'odds_'}{gid}_{market}",
                          width="stretch", on_click=add_to_slip, args=([leg],),
                          help=f"Prob. ajustada {leg.prob:.1%} (Monte Carlo {leg.model_prob:.1%} + mercado) · "
                               f"implícita {leg.implied_prob_pct:.1f}% · añadir a la boleta")
                st.markdown(f'<div class="ev-cap {"pos" if plus else "neg"}">P {leg.prob:.0%} · '
                            f'EV {leg.ev_pct:+.1f}%</div>', unsafe_allow_html=True)


# =========================================================================== #
# Sección de jugadores
# =========================================================================== #
def render_game_header(game: pd.Series, proj: dict | None, source: str, n_players: int,
                       has_props: bool) -> None:
    away, home = game["away_team"], game["home_team"]
    with st.container(key="gamehdr"):
        c_away, c_mid, c_home = st.columns([1, 3, 1], vertical_alignment="center")
        win_home = proj["home_win_pct"] if proj else None
        c_away.markdown(_team_block(away, 100 - win_home if proj else None), unsafe_allow_html=True)
        c_home.markdown(_team_block(home, win_home), unsafe_allow_html=True)
        score = f"{away} {proj['away_pts']} - {home} {proj['home_pts']}" if proj else f"{away} @ {home}"
        props_chip = (chip(f"{n_players} jugadores con props", "chip-green") if has_props
                      else chip("Sin props en vivo · líneas de referencia", "chip-pink"))
        when = chip(f"{esc(game['gameday'])} · {esc(game['gametime'])} ET")
        week = chip(f"Semana {game['week']}")
        c_mid.markdown(
            f'<div class="center"><div class="mono" style="font-size:2.4rem;font-weight:700">{score}</div>'
            f'<div class="chips" style="justify-content:center;margin-top:6px">'
            f'{when}{week}{chip(esc(source))}{props_chip}</div></div>', unsafe_allow_html=True)


def render_player_cards(players: pd.DataFrame, best_ev: dict, per_row: int = 4) -> None:
    """Catálogo de jugadores del partido. Clic en una tarjeta -> Focus View."""
    for start in range(0, len(players), per_row):
        cols = st.columns(per_row)
        for col, row in zip(cols, players.iloc[start:start + per_row].itertuples()):
            sel = False
            key = f"{row.name_key}_{row.team}".replace(" ", "_")
            img = row.headshot if isinstance(row.headshot, str) else logo_url(row.team)
            best = best_ev.get(row.player)
            if best:
                tag = (f'{ev_badge(best["ev_pct"], best["fair_odds"], True)}'
                       f'<div class="muted" style="margin-top:3px">{esc(best["label"])} · '
                       f'{best["n_plus"]} prop(s) +EV</div>')
            elif row.has_odds:
                tag = '<span class="evb evb-neg">Sin props +EV</span>'
            else:
                tag = '<span class="evb evb-none">Líneas de referencia</span>'
            with col:
                with st.container(key=f"{'pcardsel_' if sel else 'pcard_'}{key}"):
                    st.markdown(f'<div class="pc"><img src="{esc(img)}"><div><div class="nm">{esc(row.player)}'
                                f'</div><div class="meta">{row.position} · {row.team}</div></div></div>{tag}',
                                unsafe_allow_html=True)
                    st.button("📊 Analizar", key=f"pl_{key}", width="stretch",
                              on_click=select_player, args=(player_id(row.name_key, row.team),))


def render_player_back_button(n_players: int) -> None:
    """Botón destacado arriba de la ficha (Focus View) para volver al catálogo."""
    with st.container(key="playerback"):
        st.button(f"⬅️ Volver a todos los jugadores del partido ({n_players})", key="back_players",
                  type="primary", width="stretch", on_click=clear_player)


LINE_COLOR = "#FFE600"   # amarillo neón: contrasta con barras verdes y rosas


def _game_labels(log: pd.DataFrame) -> list[str]:
    """'W12<br>@ CLE' (visitante) / 'W3<br>vs TB' (local). game_id = AÑO_SEM_VISITA_LOCAL.
    Dos renglones: se leen completos sin inclinar el texto."""
    labels, seasons = [], log["season"] if "season" in log else pd.Series([None] * len(log))
    last_season = seasons.max() if seasons.notna().any() else None
    for i, (w, opp) in enumerate(zip(log["week"], log.get("opponent_team", pd.Series([""] * len(log))))):
        gid = str(log["game_id"].iloc[i]) if "game_id" in log else ""
        parts = gid.split("_")
        if len(parts) >= 4 and opp:
            venue = "@" if parts[3] == opp else "vs"
        else:
            venue = "vs"
        yr = f" '{str(seasons.iloc[i])[-2:]}" if last_season is not None and seasons.iloc[i] != last_season else ""
        labels.append(f"W{int(w)}{yr}<br>{venue} {opp or '—'}")
    return labels


def prop_chart(log: pd.DataFrame, stat: str, line: float) -> go.Figure:
    vals = pd.to_numeric(log[stat], errors="coerce").fillna(0)
    x = _game_labels(log)
    colors = [GREEN if v > line else PINK for v in vals]
    fmt = (lambda v: f"{v:.0f}") if (vals % 1 == 0).all() else (lambda v: f"{v:.1f}")
    fig = go.Figure(go.Bar(
        x=x, y=vals, marker=dict(color=colors, line=dict(width=0), cornerradius=6),
        text=[f"<b>{fmt(v)}</b>" for v in vals], textposition="outside", cliponaxis=False,
        textfont=dict(color="#FFFFFF", size=13, family="JetBrains Mono"),
        hovertemplate="%{x}<br>%{y:.1f}<extra></extra>"))
    # Línea activa: amarillo neón, discontinua y por encima de las barras
    fig.add_hline(y=line, line_dash="dash", line_color=LINE_COLOR, line_width=2, layer="above")
    # Etiqueta arriba a la izquierda, en la franja superior: nunca tapa barras ni sus valores
    fig.add_annotation(xref="paper", yref="paper", x=0, y=1.0, xanchor="left", yanchor="bottom",
                       yshift=8, showarrow=False, text=f"<b>╌╌ Línea {line:g}</b>",
                       font=dict(color=LINE_COLOR, size=13, family="JetBrains Mono"))
    top = max(float(vals.max()), line) or 1
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380,
        margin=dict(l=30, r=30, t=40, b=70), showlegend=False, bargap=.28,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        yaxis=dict(gridcolor=BORDER, zeroline=False, tickfont=dict(color=MUTED),
                   range=[0, top * 1.28]))
    fig.update_xaxes(showgrid=False, type="category", tickangle=0, automargin=True,
                     tickfont=dict(color="#FFFFFF", size=11, family="Inter, sans-serif"))
    return fig


def _side_cell(row, side: str) -> str:
    odds = getattr(row, side)
    prob = getattr(row, f"{side}_prob_pct")
    badge = ev_badge(getattr(row, f"{side}_ev_pct"), getattr(row, f"{side}_fair"),
                     getattr(row, f"{side}_plus_ev"), getattr(row, f"{side}_status", None))
    mc = getattr(row, f"{side}_mc_pct", None)
    tip = f' title="Monte Carlo puro {mc:.1f}% · ajustada con el mercado {prob:.1f}%"' if mc is not None else ""
    return (f'<span class="odd">{fmt_odds(odds)}</span><span class="mc"{tip}>P {prob:.1f}%</span>{badge}')


def render_lines_table(ev_lines: pd.DataFrame, active_line: float, reference: bool) -> None:
    title = "Líneas de referencia (sin cuotas en vivo)" if reference else "Líneas oficiales y alternas"
    rows = []
    for r in ev_lines.itertuples(index=False):
        cls = "active" if math.isclose(r.line, active_line) else ""
        cls += "" if r.realism_ok else " bad"
        guard = "" if r.realism_ok else ' <span class="evb evb-neg">🚫 Realism Guard</span>'
        rows.append(f'<tr class="{cls}"><td>{r.line:g}{guard}</td><td>{_side_cell(r, "over")}</td>'
                    f'<td>{_side_cell(r, "under")}</td></tr>')
    st.markdown(f'<div class="muted" style="margin:0 0 4px">{title} · probabilidad real Monte Carlo '
                f'junto a cada cuota</div><div class="ladder"><table class="lines"><thead><tr><th>Línea</th>'
                f'<th>Over</th><th>Under</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>',
                unsafe_allow_html=True)


def render_mode_card(mode: dict, stat: str, line: float, line_src: str, reference: bool) -> None:
    """🎯 Línea Más Probable (Moda 10k Sims) y su diferencia frente a la línea activa."""
    if mode.get("mode_val") is None:
        return
    short, long_ = STAT_UNIT.get(stat, ("", ""))
    mv = mode["mode_val"]
    val = f"{mv:.0f}" if mode["discrete"] else f"{mv:.1f}"
    rng = ("" if mode["discrete"] else
           f' · rango {mode["range_low"]:g}–{mode["range_high"]:g}')
    diff = mv - line
    tone = GREEN if diff > 0 else PINK if diff < 0 else MUTED
    src = "de referencia" if reference else f"de la {line_src}" if line_src == "casa" else f"de {line_src}"
    side_word = "por encima" if diff > 0 else "por debajo" if diff < 0 else "igual"
    st.markdown(
        f'<div class="modecard"><div class="mc-left"><div class="k">🎯 Línea Más Probable (Moda 10k Sims)</div>'
        f'<div class="v">{val} <span class="u">{short}</span></div>'
        f'<div class="muted">Concentración: <b class="mono">{mode["concentration_pct"]:.1f}%</b>{rng}</div></div>'
        f'<div class="mc-right"><div class="delta" style="color:{tone}">{diff:+.1f}</div>'
        f'<div class="muted">El modelo proyecta que el resultado más repetido es <b>{val} {long_}</b> vs '
        f'<b class="mono">{line:g}</b> {src} ({side_word} de la línea).</div></div></div>',
        unsafe_allow_html=True)


def _render_td_board(props: pd.DataFrame, name: str, groups: list[str], weekly: pd.DataFrame,
                     log: pd.DataFrame, name_key: str, home: str | None, away: str | None,
                     has_odds: bool) -> None:
    """Tablero de anotador: Anytime TD y Primer TD con cuota decimal, prob. MC y badge +EV."""
    if "total_tds" not in log:
        return
    hist = pd.to_numeric(log["total_tds"], errors="coerce").dropna()
    cells = []
    for g in ("anytime_td", "first_td"):
        if g not in groups:
            continue
        sims = (first_td_sims(weekly, log, name_key, home, away) if g == "first_td"
                else player_sims(name_key, "total_tds", tuple(hist.tail(pe.REALISM_GAMES).tolist())))
        mc = pe.probability_from_sims(sims, 0.5, "over")
        w = pe.model_weight(len(hist.tail(pe.REALISM_GAMES)),
                            pe.MODEL_WEIGHT_FIRST_TD if g == "first_td" else pe.MODEL_WEIGHT)
        lines = player_lines(props, name, g) if has_odds else pd.DataFrame()
        row = lines.loc[lines["line"] == 0.5] if not lines.empty else lines
        odds = float(row["over"].iloc[0]) if not row.empty and not pd.isna(row["over"].iloc[0]) else None
        if odds:
            e = pe.prop_edge(mc, odds, pe.no_vig_prob(odds, row["under"].iloc[0]), w,
                             family="first_td" if g == "first_td" else "total_tds", side="over",
                             n_games=len(hist.tail(pe.REALISM_GAMES)))
            p = e["model_prob_pct"] / 100
            badge = ev_badge(e["ev_pct"], e["fair_odds"], e["is_plus_ev"], e["status"])
        else:
            p = mc
            badge = ev_badge(None, round(1 / p, 2) if p > 0 else None)
        cells.append(f'<div class="tdcell"><div class="k">{GROUP_LABEL[g]} · Sí</div>'
                     f'<div class="v">{fmt_odds(odds)}</div><div class="muted">P {p:.1%} · MC {mc:.1%} · '
                     f'implícita {"—" if not odds else f"{100 / odds:.1f}%"}</div>{badge}</div>')
    if cells:
        st.markdown(f'<div class="tdboard">{"".join(cells)}</div>', unsafe_allow_html=True)


def render_player_terminal(player: pd.Series, props: pd.DataFrame, weekly: pd.DataFrame,
                           game_id: str) -> None:
    name, pos = player["player"], player["position"]
    log = dp.get_player_game_log(weekly, name, pe.REALISM_GAMES)
    with st.container(key="terminal"):
        img = player["headshot"] if isinstance(player.get("headshot"), str) else logo_url(player["team"])
        src = ("cuotas The-Odds-API", "chip-green") if player["has_odds"] else \
              ("líneas de referencia · cuota manual", "chip-pink")
        st.markdown(f'<div class="term-head"><img src="{esc(img)}"><div><p class="term-title">{esc(name)}</p>'
                    f'<div class="chips" style="justify-content:flex-start">{chip(pos)}{chip(player["team"])}'
                    f'{chip(src[0], src[1])}{chip(f"L{len(log)} juegos")}</div></div></div>',
                    unsafe_allow_html=True)

        if log.empty:
            st.warning("Sin historial reciente para este jugador: no se puede simular.")
            return

        k = f"{game_id}|{player['name_key']}"
        groups = list(player["markets"] or POSITION_GROUPS.get(pos, []))
        parts = str(game_id).split("_")
        away_t, home_t = (parts[2], parts[3]) if len(parts) >= 4 else (None, None)

        # --- Selector de categoría ---
        cat_key = f"cat_{k}"
        if cat_key not in st.session_state:
            st.session_state[cat_key] = DEFAULT_CATEGORY.get(pos, "all")
        with st.container(key="catpills"):
            cat = st.pills("Categoría de mercado", list(CATEGORIES), format_func=CATEGORIES.get, key=cat_key,
                           required=True, label_visibility="collapsed") or "all"
        cat_groups = groups if cat == "all" else [g for g in groups if GROUPS[g]["cat"] == cat]

        # --- Indicadores L10 de la categoría (grid simétrico) ---
        metrics = CATEGORY_METRICS.get(cat) or POSITION_METRICS.get(pos, POSITION_METRICS["WR"])
        if cat == "td" and pos != "QB":
            metrics = [m for m in metrics if m[1] != "passing_tds"]
        metrics = [m for m in metrics if m[1] in log]
        if metrics:
            with st.container(key="l10grid"):
                cols = st.columns(len(metrics), gap="small")
                for col, (label, mstat) in zip(cols, metrics):
                    v = pd.to_numeric(log[mstat], errors="coerce").fillna(0)
                    l10, l3 = v.mean(), v.tail(3).mean()
                    if mstat == "target_share":
                        col.metric(f"{label} · L10", f"{l10:.1%}", f"{(l3 - l10) * 100:+.1f} pts L3")
                    elif l10 < 10:
                        col.metric(f"{label} · L10", f"{l10:.2f}", f"{l3 - l10:+.2f} L3")
                    else:
                        col.metric(f"{label} · L10", f"{l10:.1f}", f"{l3 - l10:+.1f} L3")

        if not cat_groups:
            where = "en The-Odds-API para este partido" if player["has_odds"] else "de referencia"
            st.info(f"{name} no tiene mercados de {CATEGORIES[cat]} {where}. Prueba 🔀 Todos.")
            return

        # --- Cápsulas de mercado dentro de la categoría ---
        grp_key = f"grp_{k}_{cat}"
        if st.session_state.get(grp_key) not in cat_groups:
            st.session_state[grp_key] = cat_groups[0]
        with st.container(key="grppills"):
            group = st.pills("Mercado", cat_groups, format_func=GROUP_LABEL.get, key=grp_key, required=True)
        spec = GROUPS[group]
        stat, yes_no = spec["stat"], bool(spec.get("yes_no"))
        hist_stat = "total_tds" if stat == "first_td" else stat
        if hist_stat not in log:
            st.info("Sin historial de esta estadística para el jugador.")
            return
        hist = pd.to_numeric(log[hist_stat], errors="coerce").dropna()
        if stat == "first_td":
            sims = first_td_sims(weekly, log, player["name_key"], home_t, away_t)
        else:
            sims = player_sims(player["name_key"], stat, tuple(hist.tail(pe.REALISM_GAMES).tolist()))

        lines = player_lines(props, name, group) if player["has_odds"] else pd.DataFrame()
        reference = lines.empty
        if reference:
            lines = reference_lines(hist, group)
        ev_lines = pe.evaluate_prop_lines(sims, lines, hist.to_numpy())
        market = spec["markets"][0]
        side_fmt = (lambda s: "Sí" if s == "Over" else "No") if yes_no else (lambda s: s)

        if cat == "td":
            _render_td_board(props, name, groups, weekly, log, player["name_key"], home_t, away_t,
                             player["has_odds"])

        km = f"{k}|{group}"
        keys = {n: f"{n}_{km}" for n in ("side", "sel", "manual", "mline", "modds")}
        d_line = default_line(lines, hist)
        st.session_state.setdefault(keys["side"], "Over")
        st.session_state.setdefault(keys["sel"], d_line)
        st.session_state.setdefault(keys["manual"], False)
        st.session_state.setdefault(keys["mline"], d_line)
        st.session_state.setdefault(keys["modds"], 1.91)

        by_line = ev_lines.set_index("line")

        def fmt_line(v: float) -> str:
            r = by_line.loc[v]
            parts = [] if yes_no else [f"{v:g}"]
            for s in ("over", "under"):
                odd = r[s]
                parts.append(f"{side_fmt(s.capitalize())} {fmt_odds(odd) if not pd.isna(odd) else '—'} "
                             f"(P {r[f'{s}_prob_pct']:.1f}%)")
            return "  ·  ".join(parts)

        # --- ✍️ Edición manual: ARRIBA, sin scroll ---
        with st.container(key="editbox"):
            st.markdown('<div class="edit-title">✍️ Editar Línea y Cuota Manual</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1.2, 3, 1.3], vertical_alignment="bottom")
            c1.pills("¿Anota?" if yes_no else "Lado", ["Over", "Under"], key=keys["side"], required=True,
                     format_func=side_fmt)
            c2.selectbox("Cuota decimal y probabilidad ajustada" if yes_no else
                         "Línea alterna · cuota decimal y probabilidad ajustada",
                         lines["line"].tolist(), key=keys["sel"], format_func=fmt_line)
            c3.toggle("Usar línea manual", key=keys["manual"], disabled=yes_no,
                      help="En mercados Sí/No la línea es fija; solo cambia la cuota" if yes_no else None)
            side = st.session_state[keys["side"]]
            c4, c5, c6 = st.columns([1, 1, 2.2], vertical_alignment="bottom")
            c4.number_input("Línea manual", key=keys["mline"], min_value=0.5, step=0.5, format="%.1f",
                            disabled=yes_no, help="Ej. 283.0 · se usa cuando 'Usar línea manual' está activo")
            c5.number_input("Cuota decimal", key=keys["modds"], min_value=1.01, step=0.01, format="%.2f",
                            help="Número decimal puro, ej. 1.57")
            # Sin resumen aquí: la evaluación única vive en el bloque de abajo
            hint = ("Sin cuotas en vivo: se usa la 'Cuota decimal'." if reference else
                    "Activa 'Usar línea manual' para evaluar tu propia línea y cuota." if not yes_no else
                    "Línea fija 0.5 (anota / no anota): edita solo la cuota.")
            c6.markdown(f'<div class="muted edit-hint">{hint}</div>', unsafe_allow_html=True)

        manual = st.session_state[keys["manual"]] and not yes_no
        mkt_prob = None
        if manual:
            line, odds, odds_src = st.session_state[keys["mline"]], st.session_state[keys["modds"]], "manual"
        else:
            line = st.session_state[keys["sel"]]
            row = lines.loc[lines["line"] == line]
            book = row[side.lower()].iloc[0] if not row.empty else np.nan
            if pd.isna(book):
                odds, odds_src = st.session_state[keys["modds"]], "manual"
            else:
                odds, odds_src = float(book), "casa"
                other = row["under" if side == "Over" else "over"].iloc[0]
                mkt_prob = pe.no_vig_prob(odds, other)       # mercado sin vig con los dos lados
        line, odds = float(line), float(odds)
        weight = pe.model_weight(len(hist.tail(pe.REALISM_GAMES)),
                                 pe.MODEL_WEIGHT_FIRST_TD if stat == "first_td" else pe.MODEL_WEIGHT)

        # --- 🎯 Línea Más Probable (moda Monte Carlo) ---
        mode_sims = sims if stat != "first_td" else player_sims(
            player["name_key"], "total_tds", tuple(hist.tail(pe.REALISM_GAMES).tolist()))
        render_mode_card(pe.most_probable_line(mode_sims, hist_stat), hist_stat, line,
                         "casa" if odds_src == "casa" else "tu línea", reference)

        # --- Evaluación +EV de la selección activa (resumen único) ---
        res = pe.evaluate_manual_prop(line, odds, log, hist_stat, side=side.lower(), sims=sims,
                                      market_prob=mkt_prob, weight=weight, family=stat)
        with st.container(key="evalbox"):
            if res["verdict"] == "RECHAZADA":
                st.error(f"🚫 Realism Guard: {res['realism']['reason']}")
            else:
                pick = (f'{GROUP_LABEL[group]} · {side_fmt(side)}' if yes_no
                        else f'{GROUP_LABEL[group]} · {side} <span class="mono">{line:g}</span>')
                st.markdown(f'<div class="big-pick">{pick} @ <span class="mono">{fmt_odds(odds)}</span> '
                            f'<span class="muted">(cuota {odds_src})</span> '
                            f'{ev_badge(res["ev_pct"], res["fair_odds"], res["is_plus_ev"], res.get("status"))}</div>',
                            unsafe_allow_html=True)
                with st.container(key="evalmetrics"):
                    e1, e2, e3, e4 = st.columns(4, gap="small")
                    e1.metric("Prob. implícita", f"{res['implied_prob_pct']:.2f}%")
                    e2.metric("Prob. ajustada", f"{res['real_prob_pct']:.2f}%", f"{res['edge_pct']:+.2f} pts")
                    e3.metric("% EV", f"{res['ev_pct']:+.2f}%")
                    e4.metric("Fair Odds", fmt_odds(res["fair_odds"]))
                css = "verdict-pos" if res["is_plus_ev"] else "verdict-neg"
                mkt_txt = (f"mercado sin vig {res['market_prob_pct']:.1f}%" if res.get("market_prob_pct") is not None
                           else "mercado estimado")
                method = prob_method_text(res, stat, mkt_txt)
                st.markdown(f"<div class='verdict-row'><span class='{css}'>{res['verdict']}</span> "
                            f"<span class='muted'>· Prob. ajustada {res['real_prob_pct']:.1f}% = {method} · "
                            f"EV = {res['real_prob_pct'] / 100:.4f} * {fmt_odds(odds)} - 1 · +EV desde "
                            f"{pe.MIN_EDGE_PCT:g}% · media {res['projection_mean']:g} · mediana "
                            f"{res['projection_median']:g} · Realism Guard OK (tope {res['realism']['threshold']:g})"
                            f"</span></div>", unsafe_allow_html=True)
                with st.container(key="addrow"):
                    st.button("➕ Añadir a la Boleta", key=f"add_{km}", type="primary",
                              on_click=add_player_selection,
                              args=(name, market, line, odds, side, game_id, weekly, mkt_prob),
                              disabled=stat == "first_td",
                              help="Primer TD depende de todo el partido: no se combina en la boleta"
                              if stat == "first_td" else None)

        # --- Gráfico L10 (hit-rate verde/rosa) ---
        values = pd.to_numeric(log[hist_stat], errors="coerce").fillna(0)
        hits = int((values > line).sum())
        what = "TDs por juego · anotó" if hist_stat == "total_tds" and line <= 0.5 else "superó la línea"
        st.markdown(f"**{GROUP_LABEL[group]}** · línea activa <span class='mono'>{line:g}</span> · "
                    f"{what} en **{hits}/{len(values)}** juegos", unsafe_allow_html=True)
        st.plotly_chart(prop_chart(log, hist_stat, line), theme=None, key=f"chart_{km}",
                        config={"displayModeBar": False})

        # --- Escaleras alternas: colapsadas y con scroll interno ---
        n_plus = int(ev_lines["over_plus_ev"].sum() + ev_lines["under_plus_ev"].sum())
        title = ("📊 Ver tabla completa de escaleras alternas (The-Odds-API)" if not reference
                 else "📊 Ver tabla de líneas de referencia")
        with st.expander(f"{title} · {len(ev_lines)} líneas · {n_plus} con +EV", expanded=False):
            render_lines_table(ev_lines, -1 if manual else line, reference)


# =========================================================================== #
# Combinadas sugeridas
# =========================================================================== #
def render_parlay_card(parlay: dict, legs_by_label: dict, bankroll: float, key: str) -> None:
    with st.container(key=f"parlay_{key}"):
        extra = chip("🔥 SGP", "chip-pink") if parlay.get("sgp") else ""
        mix = chip(f"👤 {parlay.get('n_player_legs', 0)} · 🏟️ {parlay.get('n_team_legs', 0)}")
        st.markdown(f"{badge_html(parlay['tier'])} {extra} {mix} <span class='muted'>· {parlay['n_legs']} "
                    f"patas</span>", unsafe_allow_html=True)
        for leg in parlay["legs"]:
            st.markdown(f"<div class='leg-line'>• {esc(leg['label'])} <b>{fmt_odds(leg['decimal_odds'])}</b> "
                        f"<span class='muted'>P {leg['prob_pct']:.0f}%</span></div>", unsafe_allow_html=True)
        stake = parlay["stake"]["stake_per_1000"] * bankroll / 1000
        st.markdown(
            '<div class="pstats">'
            f'<div class="pstat"><div class="k">Cuota</div><div class="v">{fmt_odds(parlay["decimal_odds"])}</div></div>'
            f'<div class="pstat"><div class="k">Prob.</div><div class="v">{parlay["joint_prob_pct"]:.1f}%</div></div>'
            f'<div class="pstat"><div class="k">Stake</div><div class="v win">${stake:,.2f}</div></div>'
            '</div>', unsafe_allow_html=True)
        p = parlay["joint_prob_pct"] / 100
        be = f" · +EV mientras la cuota real sea ≥ {fmt_odds(1 / p)}" if p > 0 else ""
        st.markdown(f"<div class='muted'>EV {parlay['ev_pct']:+.1f}% con cuota teórica{be}</div>",
                    unsafe_allow_html=True)
        legs = [legs_by_label[l] for l in parlay["labels"] if l in legs_by_label]
        st.button("➕ Añadir Parlay a la Boleta", key=f"addp_{key}", width="stretch",
                  on_click=add_to_slip, args=(legs,))


PARLAY_MODES = {
    "mixed": "🔀 Mixtos (Equipos + Jugadores)",
    "players": "👤 Solo Jugadores",
    "teams": "🏟️ Solo Equipos",
}


def render_parlay_mode_selector() -> str:
    st.markdown("<h3 style='margin-bottom:0'>🔥 Combinadas Sugeridas por Riesgo</h3>", unsafe_allow_html=True)
    st.session_state.setdefault("parlay_mode", "mixed")
    mode = st.segmented_control("Tipo de parlay", list(PARLAY_MODES), format_func=PARLAY_MODES.get,
                                key="parlay_mode", required=True, label_visibility="collapsed")
    return mode or "mixed"


def render_parlay_module(tiers: dict, legs_by_label: dict, bankroll: float, note: str = "") -> None:
    if note:
        st.caption(note)
    validated = pe.CALIBRATION.get("validated_families")
    if validated is not None and bb.parlay_param("validated_only"):
        names = ", ".join(FAMILY_LABEL.get(f, f) for f in validated) or "ninguno"
        st.caption(f"🧪 Solo se combinan mercados con ventaja comprobada en el backtest: **{names}**. "
                   "La probabilidad conjunta está validada (esperado vs real dentro de ±1 punto); el resto de "
                   "mercados se puede evaluar en la boleta, pero sin garantía de valor.")
        if not any(tiers.values()):
            st.info("Hoy no hay combinadas que cumplan las reglas validadas por el backtest en este modo. "
                    "Prueba 👤 Solo Jugadores; si tampoco hay, es preferible no apostar a forzar una combinada "
                    "sin ventaja.")
    cols = st.columns(3)
    for col, tier in zip(cols, ("BAJO", "MODERADO", "ALTO")):
        info = bb.RISK_TIERS[tier]
        with col:
            st.markdown(f"<h4 class='tier-h' style='color:{TIER_COLORS[tier]}'>{info['emoji']} "
                        f"{info['badge']}</h4><div class='muted'>{info['rule']}</div>", unsafe_allow_html=True)
            parlays = tiers.get(tier, [])
            if not parlays:
                st.info("Sin combinadas +EV que cumplan este nivel con las cuotas actuales.")
            for i, p in enumerate(parlays):
                render_parlay_card(p, legs_by_label, bankroll, f"{tier}_{i}")


# =========================================================================== #
# Boleta lateral
# =========================================================================== #
def render_bet_slip(weekly: pd.DataFrame) -> None:
    st.markdown('<div class="slip-title">🎟️ Boleta de Apuestas (Bet Slip)</div>', unsafe_allow_html=True)
    st.session_state.setdefault("bankroll", 1000.0)
    st.session_state.setdefault("stake", 100.0)
    with st.container(key="slipbank"):
        st.number_input("Bankroll ($)", key="bankroll", min_value=10.0, step=50.0, format="%.2f")
    legs = slip()
    if not legs:
        st.info("Toca una cuota de partido, añade una prop desde la analítica de jugadores "
                "o un parlay sugerido.")
        return

    sgp = bb.detect_sgp(legs)
    sgp_events = set(sgp["games"])
    for leg in legs:
        in_sgp = leg.event_id in sgp_events
        key = f"slipleg_{'sgp' if in_sgp else 'x'}_{abs(hash(leg.label))}"
        with st.container(key=key):
            c1, c2 = st.columns([0.82, 0.18], gap="small", vertical_alignment="center")
            tag = game_tag(leg.event_id)
            tag_html = chip(("🔥 " if in_sgp else "") + tag, "chip-pink" if in_sgp else "") if tag else ""
            c1.markdown(f"<div class='leg-line'>{tag_html} {esc(leg.label)}<br><b>{fmt_odds(leg.decimal_odds)}</b> "
                        f"<span class='muted'>· P {leg.prob:.1%} (MC {leg.model_prob:.0%})</span></div>", unsafe_allow_html=True)
            c2.button("❌", key=f"rm_{abs(hash(leg.label))}", on_click=remove_from_slip, args=(leg.label,),
                      help="Quitar de la boleta")

    if sgp["is_sgp"]:
        games = " · ".join(f"{game_tag(e) or e}: {len(l)} selecciones" for e, l in sgp["games"].items())
        st.markdown(f'<div class="sgp-alert">{bb.SGP_ALERT}<div class="muted">{esc(games)} · la probabilidad '
                    f'conjunta ya incluye la correlación simulada del partido</div></div>',
                    unsafe_allow_html=True)

    res = evaluate_slip(legs, weekly)
    theo = bb.theoretical_odds(legs)
    with st.container(key="slipadj"):
        if len(legs) >= 2:
            factors = " * ".join(fmt_odds(l.decimal_odds) for l in legs)
            formula = f"{factors} = {fmt_odds(theo)}" if len(legs) <= 5 else fmt_odds(theo)
            st.markdown(f'<div class="theo"><div class="muted">Cuota Multiplicada Teórica</div>'
                        f'<div class="formula">{formula}</div></div>', unsafe_allow_html=True)
            adj_key = f"adj_{slip_signature(legs)}"
            st.session_state.setdefault(adj_key, float(theo))
            st.number_input("🎯 Cuota Real Ajustada por Draftea", key=adj_key, min_value=1.01, step=0.01,
                            format="%.2f", help="Escribe la cuota que te ofrece la casa para esta combinada.")
            if sgp["is_sgp"]:
                st.warning(bb.SGP_NOTE)
            else:
                st.caption(bb.SGP_NOTE)
            adjusted = float(st.session_state[adj_key])
        else:
            adjusted = theo
        st.number_input("Monto a apostar ($)", key="stake", min_value=0.0, step=10.0, format="%.2f")
    rp = bb.reprice_parlay(res["joint_prob_pct"], adjusted, st.session_state["stake"],
                           st.session_state["bankroll"], theoretical=theo)

    with st.container(key="slipmetrics"):
        c1, c2 = st.columns(2, gap="small")
        c1.metric("Prob. conjunta", f"{rp['joint_prob_pct']:.2f}%")
        c2.metric("Prob. implícita", f"{rp['implied_prob_pct']:.2f}%")
        c3, c4 = st.columns(2, gap="small")
        c3.metric("Cuota usada", fmt_odds(rp["adjusted_odds"]),
                  f"{-rp['house_haircut_pct']:+.1f}% vs teórica" if rp["house_haircut_pct"] else None,
                  delta_color="normal")
        c4.metric("% EV", f"{rp['ev_pct']:+.2f}%")
        st.markdown(f"<div class='slipbadges'>{badge_html(rp['tier'])} "
                    f"{ev_badge(rp['ev_pct'], rp['break_even_odds'], rp['is_plus_ev'])}</div>",
                    unsafe_allow_html=True)

    with st.container(key="slippay"):
        st.markdown(f'<div class="payout"><div class="muted">Calculadora de pago</div><div class="big">'
                    f'${rp["stake"]:,.2f}<span class="arrow">➔</span><span class="win">${rp["payout"]:,.2f}</span>'
                    f'</div><div class="muted">cobrados · neto ${rp["net_profit"]:+,.2f} · '
                    f'$EV {rp["ev_cash"]:+,.2f}</div></div>', unsafe_allow_html=True)
        st.caption(f"La combinada es +EV mientras la cuota real sea ≥ {fmt_odds(rp['break_even_odds'])} · "
                   f"Kelly 25% sugiere ${rp['kelly']['stake']:,.2f} con tu bankroll.")
        if res.get("independent_prob_pct") is not None:
            st.caption(f"Si las patas fueran independientes: {res['independent_prob_pct']:.2f}% "
                       f"(la diferencia es la correlación).")
        for issue in res.get("coherence_issues") or []:
            st.caption(f"⚠️ {issue}")
    with st.container(key="slipclear"):
        st.button("🗑️ Vaciar boleta", key="clear_slip", on_click=clear_slip, width="stretch")
