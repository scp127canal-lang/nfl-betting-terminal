"""Ingesta de datos NFL con nfl_data_py: stats semanales, play-by-play y calendario."""

from __future__ import annotations

import datetime as dt
import logging
import re
import unicodedata

import nfl_data_py as nfl
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

REGULAR_SEASON_WEEKS = 18
LAST_POSTSEASON_WEEK = 22  # Super Bowl en la numeración de nflverse
DATA_TTL = 6 * 3600  # los datos históricos cambian poco: 6 h de caché

PBP_COLUMNS = [
    "game_id", "season", "week", "season_type", "home_team", "away_team",
    "posteam", "defteam", "play_type", "pass_attempt", "rush_attempt",
    "complete_pass", "yards_gained", "air_yards",
    "passer_player_id", "passer_player_name",
    "receiver_player_id", "receiver_player_name",
    "rusher_player_id", "rusher_player_name",
]

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
OFFENSE_POSITIONS = {"QB", "RB", "FB", "WR", "TE"}

# The-Odds-API usa nombres completos; nflverse usa abreviaturas.
TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def normalize_name(name: str) -> str:
    """Normaliza nombres para cruzar The-Odds-API con nflverse.

    "Ja'Marr Chase" -> "jamarr chase" | "Kenneth Walker III" -> "kenneth walker"
    """
    if not isinstance(name, str):
        return ""
    txt = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    txt = re.sub(r"[^a-z\s]", "", txt.lower().replace("-", " "))
    parts = [p for p in txt.split() if p not in _SUFFIXES]
    return " ".join(parts)


def default_seasons(n: int = 2, today: dt.date | None = None) -> list[int]:
    """Últimas `n` temporadas, incluida la activa."""
    current = get_current_nfl_week(today, use_schedule=False)["season"]
    return list(range(current - n + 1, current + 1))


# --------------------------------------------------------------------------- #
# Semana activa
# --------------------------------------------------------------------------- #
def _labor_day(year: int) -> dt.date:
    first = dt.date(year, 9, 1)
    return first + dt.timedelta(days=(0 - first.weekday()) % 7)


def _calendar_week(today: dt.date) -> dict:
    """Cálculo por calendario: la temporada arranca el jueves tras Labor Day
    y cada semana NFL va de martes a lunes."""
    season = today.year - 1 if today.month <= 2 else today.year
    week1_start = _labor_day(season) + dt.timedelta(days=1)  # martes previo al kickoff

    if today < week1_start:
        return {"season": season, "week": 1, "season_type": "PRE", "source": "calendar"}

    week = (today - week1_start).days // 7 + 1
    if week <= REGULAR_SEASON_WEEKS:
        season_type = "REG"
    elif week <= LAST_POSTSEASON_WEEK + 1:  # +1: semana de descanso antes del SB
        season_type, week = "POST", min(week, LAST_POSTSEASON_WEEK)
    else:
        season_type, week = "OFF", LAST_POSTSEASON_WEEK
    return {"season": season, "week": week, "season_type": season_type, "source": "calendar"}


def get_current_nfl_week(today: dt.date | None = None, use_schedule: bool = True) -> dict:
    """Detecta la semana NFL activa para la fecha de hoy.

    Intenta primero con el calendario oficial (nfl_data_py.import_schedules):
    la semana activa es la primera cuyo último partido aún no ha pasado.
    Si no hay red o calendario, usa el cálculo por fechas.

    Devuelve: {"season": int, "week": int, "season_type": "PRE|REG|POST|OFF", "source": str}
    """
    today = today or dt.date.today()
    cal = _calendar_week(today)
    if not use_schedule:
        return cal

    try:
        sched = load_schedules([cal["season"]])
        if sched.empty:
            return cal
        sched = sched.assign(gameday=pd.to_datetime(sched["gameday"]).dt.date)
        by_week = sched.groupby("week").agg(last_day=("gameday", "max"),
                                            game_type=("game_type", "first"))
        pending = by_week[by_week["last_day"] >= today]
        if pending.empty:
            return {**cal, "season_type": "OFF"}
        week = int(pending.index.min())
        season_type = "REG" if pending.loc[week, "game_type"] == "REG" else "POST"
        return {"season": cal["season"], "week": week,
                "season_type": season_type, "source": "schedule"}
    except Exception as exc:  # sin red / temporada aún no publicada
        logger.warning("Calendario no disponible (%s); uso cálculo por fechas.", exc)
        return cal


# --------------------------------------------------------------------------- #
# Ingesta
# --------------------------------------------------------------------------- #
NFLVERSE_RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"


def _weekly_fallback(seasons: list[int], **_) -> pd.DataFrame:
    """nfl_data_py está archivado y da 404 desde 2025: leemos el nuevo
    `stats_player_week_<año>.parquet` de nflverse (mismo esquema de columnas)."""
    df = pd.read_parquet(f"{NFLVERSE_RELEASES}/stats_player/stats_player_week_{seasons[0]}.parquet")
    return df.rename(columns={"team": "recent_team"})


def _pbp_fallback(seasons: list[int], columns=None, **_) -> pd.DataFrame:
    return pd.read_parquet(f"{NFLVERSE_RELEASES}/pbp/play_by_play_{seasons[0]}.parquet",
                           columns=columns)


def _per_season(loader, seasons, label: str, fallback=None, **kwargs) -> pd.DataFrame:
    """Carga temporada a temporada para que un año sin publicar no tumbe el resto.
    Si nfl_data_py falla, intenta la fuente directa de nflverse."""
    frames = []
    for season in seasons:
        for fn in filter(None, (loader, fallback)):
            try:
                frames.append(fn([season], **kwargs))
                break
            except Exception as exc:
                logger.warning("%s %s vía %s no disponible: %s", label, season, fn.__name__, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(ttl=DATA_TTL, show_spinner=False)
def load_schedules(seasons: list[int]) -> pd.DataFrame:
    return _per_season(nfl.import_schedules, seasons, "Calendario")


@st.cache_data(ttl=DATA_TTL, show_spinner=False)
def load_weekly_stats(seasons: list[int]) -> pd.DataFrame:
    """Estadísticas semanales por jugador (yardas, recepciones, target_share,
    air_yards_share, etc.). Añade `name_key` para cruzar con las cuotas."""
    df = _per_season(nfl.import_weekly_data, seasons, "Weekly",
                     fallback=_weekly_fallback, downcast=True)
    if df.empty:
        return df
    # Solo ataque: el nuevo archivo de nflverse incluye defensas y hay homónimos
    # (Josh Allen QB vs Josh Allen LB) que mezclarían historiales.
    if "position" in df.columns:
        df = df[df["position"].isin(OFFENSE_POSITIONS)].copy()
    # nfl_data_py (<=2024) llama "interceptions" a las INT lanzadas
    if "passing_interceptions" not in df.columns and "interceptions" in df.columns:
        df["passing_interceptions"] = df["interceptions"]
    # Estadísticas combinadas para mercados de props
    for col in ("rushing_yards", "receiving_yards", "rushing_tds", "receiving_tds"):
        if col not in df.columns:
            df[col] = 0
    df["rush_rec_yds"] = df["rushing_yards"].fillna(0) + df["receiving_yards"].fillna(0)
    df["total_tds"] = df["rushing_tds"].fillna(0) + df["receiving_tds"].fillna(0)
    df["name_key"] = df["player_display_name"].map(normalize_name)
    return df.sort_values(["season", "week"]).reset_index(drop=True)


@st.cache_data(ttl=DATA_TTL, show_spinner=False)
def load_pbp(seasons: list[int], columns: list[str] | None = None) -> pd.DataFrame:
    """Play-by-play (por defecto solo las columnas necesarias para el motor)."""
    return _per_season(nfl.import_pbp_data, seasons, "PBP", fallback=_pbp_fallback,
                       columns=columns or PBP_COLUMNS, downcast=True, cache=False)


# --------------------------------------------------------------------------- #
# Vistas derivadas
# --------------------------------------------------------------------------- #
def get_player_game_log(weekly: pd.DataFrame, player_name: str, n: int = 10) -> pd.DataFrame:
    """Últimos `n` partidos del jugador, en orden cronológico."""
    if weekly.empty:
        return weekly
    log = weekly[weekly["name_key"] == normalize_name(player_name)]
    return log.sort_values(["season", "week"]).tail(n).reset_index(drop=True)


def team_pass_volume(pbp: pd.DataFrame) -> pd.DataFrame:
    """Intentos de pase y air yards totales por equipo y partido (desde PBP)."""
    plays = pbp[pbp["pass_attempt"] == 1]
    return (plays.groupby(["season", "week", "posteam"])
            .agg(pass_attempts=("pass_attempt", "sum"),
                 team_air_yards=("air_yards", "sum"))
            .reset_index()
            .rename(columns={"posteam": "team"}))


def team_td_log(weekly: pd.DataFrame, team: str, n: int = 10) -> pd.Series:
    """TDs ofensivos (carrera + recepción) por partido del equipo, últimos `n`."""
    if weekly.empty or "total_tds" not in weekly:
        return pd.Series(dtype=float)
    games = (weekly[weekly["recent_team"] == team]
             .groupby(["season", "week"])["total_tds"].sum().sort_index())
    return games.tail(n).astype(float)


def team_points_log(schedules: pd.DataFrame, team: str, n: int = 10) -> pd.DataFrame:
    """Puntos anotados y permitidos por `team` en sus últimos `n` partidos jugados."""
    played = schedules.dropna(subset=["home_score", "away_score"])
    home = played[played["home_team"] == team].assign(
        points_for=lambda d: d["home_score"], points_against=lambda d: d["away_score"])
    away = played[played["away_team"] == team].assign(
        points_for=lambda d: d["away_score"], points_against=lambda d: d["home_score"])
    log = pd.concat([home, away]).sort_values(["season", "week"]).tail(n)
    return log[["season", "week", "points_for", "points_against"]].reset_index(drop=True)
