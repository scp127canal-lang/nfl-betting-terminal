"""Ingesta de cuotas desde The-Odds-API (v4): mercados de partido y TODOS los props de jugador.

Todas las cuotas se piden en formato americano y se convierten a decimal puro
con `config.american_to_decimal` (float con 2 decimales, sin 'x' ni sufijos).

Coste: The-Odds-API cobra 1 crédito por mercado devuelto y región. Con el
catálogo completo un partido cuesta ~17 créditos, por eso los props por
partido se cachean 30 min en memoria y en disco (cache/odds/props/).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from config import THE_ODDS_API_KEY, american_to_decimal
from data_pipeline import normalize_name

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
ODDS_TTL = 1800  # 30 min
PROPS_CACHE_DIR = Path(__file__).parent / "cache" / "odds" / "props"

GAME_MARKETS = ("h2h", "spreads", "totals")

# --------------------------------------------------------------------------- #
# Catálogo de mercados de jugador
#   group  -> pestaña/cápsula del panel del jugador (une principal + _alternate)
#   stat   -> columna de stats semanales que se simula
#   model  -> "yards" (Gamma cero-inflada) | "count" (Poisson/Binomial negativa)
#             | "first_td" (reparto Poisson del primer TD del partido)
#   cat    -> td | pass | rush | rec
# --------------------------------------------------------------------------- #
MARKET_GROUPS = {
    "anytime_td":   {"label": "⚽ Anytime TD", "cat": "td", "stat": "total_tds", "model": "count",
                     "markets": ["player_anytime_td"], "yes_no": True},
    "first_td":     {"label": "🥇 Primer TD", "cat": "td", "stat": "first_td", "model": "first_td",
                     "markets": ["player_1st_td"], "yes_no": True},
    "rush_rec_tds": {"label": "TDs carrera+rec", "cat": "td", "stat": "total_tds", "model": "count",
                     "markets": ["player_rush_reception_tds"]},
    "pass_tds":     {"label": "TD pase", "cat": "pass", "stat": "passing_tds", "model": "count",
                     "markets": ["player_pass_tds", "player_pass_tds_alternate"]},
    "pass_yds":     {"label": "Yds pase", "cat": "pass", "stat": "passing_yards", "model": "yards",
                     "markets": ["player_pass_yds", "player_pass_yds_alternate"]},
    "pass_att":     {"label": "Intentos pase", "cat": "pass", "stat": "attempts", "model": "count",
                     "markets": ["player_pass_attempts"]},
    "pass_cmp":     {"label": "Completos", "cat": "pass", "stat": "completions", "model": "count",
                     "markets": ["player_pass_completions"]},
    "pass_int":     {"label": "Intercepciones", "cat": "pass", "stat": "passing_interceptions",
                     "model": "count", "markets": ["player_pass_interceptions"]},
    "rush_yds":     {"label": "Yds carrera", "cat": "rush", "stat": "rushing_yards", "model": "yards",
                     "markets": ["player_rush_yds", "player_rush_yds_alternate"]},
    "rush_att":     {"label": "Acarreos", "cat": "rush", "stat": "carries", "model": "count",
                     "markets": ["player_rush_attempts"]},
    "rec_yds":      {"label": "Yds recepción", "cat": "rec", "stat": "receiving_yards", "model": "yards",
                     "markets": ["player_reception_yds", "player_reception_yds_alternate"]},
    "receptions":   {"label": "Recepciones", "cat": "rec", "stat": "receptions", "model": "count",
                     "markets": ["player_receptions", "player_receptions_alternate"]},
    "rush_rec_yds": {"label": "Yds carrera+rec", "cat": "rec", "stat": "rush_rec_yds", "model": "yards",
                     "markets": ["player_rush_reception_yds"]},
}
MARKET_TO_GROUP = {m: g for g, spec in MARKET_GROUPS.items() for m in spec["markets"]}
# Mercado de The-Odds-API -> columna de stats que se simula
MARKET_STAT_MAP = {m: MARKET_GROUPS[g]["stat"] for m, g in MARKET_TO_GROUP.items()}
YES_NO_MARKETS = {m for g, s in MARKET_GROUPS.items() if s.get("yes_no") for m in s["markets"]}

ALL_PLAYER_MARKETS = tuple(MARKET_TO_GROUP)
CORE_PLAYER_MARKETS = ("player_pass_yds_alternate", "player_pass_tds", "player_rush_yds_alternate",
                       "player_reception_yds_alternate", "player_receptions_alternate",
                       "player_anytime_td")
# ODDS_PLAYER_MARKETS=core reduce el coste a ~6 créditos por partido
PLAYER_MARKETS = (CORE_PLAYER_MARKETS if os.getenv("ODDS_PLAYER_MARKETS", "all").lower() == "core"
                  else ALL_PLAYER_MARKETS)

ODDS_COLUMNS = [
    "event_id", "commence_time", "home_team", "away_team", "bookmaker",
    "market", "outcome", "player", "player_key", "line", "decimal_odds", "last_update",
]

QUOTA = {"remaining": None, "used": None, "last_cost": None}


class OddsAPIError(RuntimeError):
    pass


def _get(path: str, **params) -> list | dict:
    if not THE_ODDS_API_KEY or THE_ODDS_API_KEY == "tu_odds_api_key_aqui":
        raise OddsAPIError("Falta THE_ODDS_API_KEY / ODDS_API_KEY en el archivo .env")
    resp = requests.get(f"{BASE_URL}{path}",
                        params={"apiKey": THE_ODDS_API_KEY, **params}, timeout=30)
    if resp.status_code != 200:
        raise OddsAPIError(f"The-Odds-API {resp.status_code}: {resp.text[:200]}")
    QUOTA.update(remaining=resp.headers.get("x-requests-remaining"),
                 used=resp.headers.get("x-requests-used"),
                 last_cost=resp.headers.get("x-requests-last"))
    logger.info("The-Odds-API %s · coste %s · restantes %s", path, QUOTA["last_cost"], QUOTA["remaining"])
    return resp.json()


def _to_decimal(price) -> float | None:
    try:
        return american_to_decimal(price)
    except (TypeError, ValueError):
        logger.warning("Cuota no convertible: %r", price)
        return None


def flatten_event(event: dict) -> list[dict]:
    """Aplana un evento a una fila por outcome: casa x mercado x outcome.

    No filtra nada: cada jugador (`description`) y cada línea alternativa
    (`point`) que venga en el JSON queda como fila propia.
    Mercados Sí/No (Anytime TD, Primer TD) se normalizan a Over/Under 0.5
    para que el motor los trate como "anota al menos 1".
    """
    rows = []
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            key = market.get("key")
            yes_no = key in YES_NO_MARKETS
            for out in market.get("outcomes", []):
                player = out.get("description")
                outcome, line = out.get("name"), out.get("point")
                if yes_no:
                    outcome = {"Yes": "Over", "No": "Under"}.get(outcome, outcome)
                    line = 0.5 if line is None else line
                rows.append({
                    "event_id": event.get("id"),
                    "commence_time": event.get("commence_time"),
                    "home_team": event.get("home_team"),
                    "away_team": event.get("away_team"),
                    "bookmaker": book.get("key"),
                    "market": key,
                    "outcome": outcome,                  # equipo, 'Over' o 'Under'
                    "player": player,
                    "player_key": normalize_name(player) if player else None,
                    "line": line,
                    "decimal_odds": _to_decimal(out.get("price")),
                    "last_update": market.get("last_update"),
                })
    return rows


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=ODDS_COLUMNS)
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True)
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df["decimal_odds"] = pd.to_numeric(df["decimal_odds"], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# Caché en disco de props por partido (sobrevive reinicios: no repite créditos)
# --------------------------------------------------------------------------- #
def _props_path(event_id: str) -> Path:
    return PROPS_CACHE_DIR / f"{event_id}.parquet"


def cached_event_props(event_id: str, max_age: float = ODDS_TTL) -> pd.DataFrame | None:
    """Props ya descargados (sin red). None si no hay copia fresca."""
    path = _props_path(event_id)
    try:
        if path.exists() and time.time() - path.stat().st_mtime < max_age:
            return pd.read_parquet(path)
    except (OSError, ValueError):
        return None
    return None


def _save_event_props(event_id: str, df: pd.DataFrame) -> None:
    try:
        PROPS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_props_path(event_id), index=False)
    except OSError as exc:
        logger.warning("No se pudo cachear props de %s: %s", event_id, exc)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=ODDS_TTL, show_spinner=False)
def fetch_events() -> pd.DataFrame:
    """Lista de partidos próximos (este endpoint no consume cuota)."""
    data = _get(f"/sports/{SPORT}/events")
    return pd.DataFrame(data, columns=["id", "commence_time", "home_team", "away_team"])


@st.cache_data(ttl=ODDS_TTL, show_spinner=False)
def fetch_game_odds(regions: str = "us") -> pd.DataFrame:
    """Moneyline (h2h), spreads y totals de todos los partidos próximos (~3 créditos)."""
    data = _get(f"/sports/{SPORT}/odds", regions=regions,
                markets=",".join(GAME_MARKETS), oddsFormat="american")
    return _frame([row for event in data for row in flatten_event(event)])


@st.cache_data(ttl=ODDS_TTL, show_spinner=False)
def fetch_event_player_props(event_id: str, regions: str = "us") -> pd.DataFrame:
    """TODOS los props de jugador de un partido (principales + alternos + TDs).

    Usa la copia en disco si tiene menos de 30 min; si no, llama a la API
    (~17 créditos con el catálogo completo, ~6 con ODDS_PLAYER_MARKETS=core).
    """
    cached = cached_event_props(event_id)
    if cached is not None:
        return cached
    data = _get(f"/sports/{SPORT}/events/{event_id}/odds", regions=regions,
                markets=",".join(PLAYER_MARKETS), oddsFormat="american")
    df = _frame(flatten_event(data))
    _save_event_props(event_id, df)
    return df


@st.cache_data(ttl=ODDS_TTL, show_spinner=False)
def fetch_all_player_props(regions: str = "us") -> pd.DataFrame:
    """Props de TODOS los partidos y TODOS los jugadores disponibles.

    Ojo: cada partido es una petición que gasta cuota de la API.
    """
    frames = []
    for event_id in fetch_events()["id"]:
        try:
            frames.append(fetch_event_player_props(event_id, regions))
        except OddsAPIError as exc:
            logger.warning("Props de %s no disponibles: %s", event_id, exc)
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else _frame([])


def is_main_market(market: str) -> bool:
    """Mercado de línea principal/fija (no escalera alterna)."""
    return market in MARKET_STAT_MAP and not market.endswith("_alternate")


def main_lines(props: pd.DataFrame) -> pd.DataFrame:
    """Solo la Línea Principal / Fija de cada jugador y mercado (la que se puede
    meter tal cual en Draftea). Descarta todas las escaleras `_alternate`.

    Por (partido, jugador, mercado) se elige la línea que ofrecen más casas;
    en empate, la más equilibrada (Over y Under más parecidos). Devuelve una fila
    por outcome con la cuota de consenso (mediana entre casas), redondeada a 2 decimales.
    """
    cols = list(props.columns) if not props.empty else ODDS_COLUMNS
    if props.empty:
        return pd.DataFrame(columns=cols)
    df = props[props["market"].map(is_main_market) & props["player"].notna()
               & props["line"].notna() & props["decimal_odds"].notna()
               & props["outcome"].isin(["Over", "Under"])]
    if df.empty:
        return pd.DataFrame(columns=cols)
    keys = ["event_id", "player", "market"]
    per_line = (df.groupby(keys + ["line", "outcome"])
                .agg(books=("bookmaker", "nunique"), odds=("decimal_odds", "median"))
                .reset_index())
    wide = per_line.pivot_table(index=keys + ["line"], columns="outcome",
                                values=["books", "odds"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["n_books"] = wide.filter(like="books_").max(axis=1)
    over, under = wide.get("odds_Over"), wide.get("odds_Under")
    wide["imbalance"] = ((over - under).abs() if over is not None and under is not None
                         else pd.Series(0.0, index=wide.index)).fillna(9.0)
    chosen = (wide.sort_values(["n_books", "imbalance"], ascending=[False, True])
              .drop_duplicates(keys)[keys + ["line"]])
    out = (df.merge(chosen, on=keys + ["line"])
           .groupby(keys + ["line", "outcome"], as_index=False)
           .agg(decimal_odds=("decimal_odds", "median"),
                **{c: (c, "first") for c in ("commence_time", "home_team", "away_team",
                                             "player_key", "last_update") if c in df.columns}))
    out["decimal_odds"] = out["decimal_odds"].round(2)
    out["bookmaker"] = "consenso"
    return out.reindex(columns=cols)


def list_players(props: pd.DataFrame) -> pd.DataFrame:
    """Todos los jugadores con props y los mercados en que aparecen."""
    if props.empty:
        return pd.DataFrame(columns=["player", "player_key", "markets"])
    return (props.dropna(subset=["player"])
            .groupby(["player", "player_key"])["market"]
            .agg(lambda m: sorted(set(m)))
            .reset_index(name="markets"))
