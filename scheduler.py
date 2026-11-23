"""Refresco diario de cuotas con APScheduler.

Cada día a las MORNING_REFRESH_HOUR:00 (hora local):
  - Si HOY hay partidos NFL: limpia el caché de cuotas y descarga cuotas frescas
    (líneas de todos los partidos próximos + props de los partidos de hoy).
  - Si NO hay partidos: no toca The-Odds-API (no gasta requests).

La última descarga se guarda en memoria y en disco (cache/odds/*.parquet)
para que sobreviva reinicios del servidor.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import threading
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

import odds_pipeline as op
from config import MORNING_REFRESH_HOUR, NFL_TIMEZONE
from data_pipeline import get_current_nfl_week, load_schedules

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = Path(__file__).parent / "cache" / "odds"          # descarga local (FastAPI 8 AM)
REPO_SNAPSHOT_DIR = Path(__file__).parent / "snapshots"           # GitHub Actions (antes de cada jornada)
SNAPSHOT_MAX_AGE = dt.timedelta(hours=14)  # más viejo que esto: se ignora y se usa la API en vivo
JOB_ID = "morning_odds_refresh"

_lock = threading.RLock()
_snapshot: dict = {"game_odds": None, "player_props": None, "meta": None}
_loaded_sig: tuple | None = None
_listeners: list[Callable[[dict], None]] = []
_scheduler: BackgroundScheduler | None = None


# --------------------------------------------------------------------------- #
# ¿Hay partido hoy?
# --------------------------------------------------------------------------- #
def nfl_today() -> dt.date:
    return dt.datetime.now(ZoneInfo(NFL_TIMEZONE)).date()


def games_today(today: dt.date | None = None) -> pd.DataFrame:
    """Partidos NFL programados para hoy (fecha en hora del Este)."""
    today = today or nfl_today()
    season = get_current_nfl_week(today, use_schedule=False)["season"]
    load_schedules.clear()  # el calendario puede cambiar (flex, aplazamientos)
    sched = load_schedules([season])
    if sched.empty:
        return sched
    days = pd.to_datetime(sched["gameday"]).dt.date
    return sched[days == today].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Snapshot de cuotas
# --------------------------------------------------------------------------- #
def clear_odds_cache() -> None:
    for fn in (op.fetch_events, op.fetch_game_odds,
               op.fetch_event_player_props, op.fetch_all_player_props):
        fn.clear()


def _fetch_props_for_today(today: dt.date) -> pd.DataFrame:
    """Props solo de los partidos de hoy: un request por partido."""
    events = op.fetch_events()
    if events.empty:
        return op._frame([])
    start = pd.to_datetime(events["commence_time"], utc=True).dt.tz_convert(NFL_TIMEZONE)
    todays = events[start.dt.date == today]
    frames = []
    for event_id in todays["id"]:
        try:
            frames.append(op.fetch_event_player_props(event_id))
        except op.OddsAPIError as exc:
            logger.warning("Props de %s no disponibles: %s", event_id, exc)
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else op._frame([])


def _save_snapshot(game_odds: pd.DataFrame, props: pd.DataFrame, meta: dict) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    game_odds.to_parquet(SNAPSHOT_DIR / "game_odds.parquet", index=False)
    props.to_parquet(SNAPSHOT_DIR / "player_props.parquet", index=False)
    (SNAPSHOT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _disk_signature() -> tuple:
    """mtime de cada meta.json: si cambia (nuevo commit de Actions), se recarga."""
    sig = []
    for d in (SNAPSHOT_DIR, REPO_SNAPSHOT_DIR):
        try:
            sig.append((str(d), (d / "meta.json").stat().st_mtime))
        except OSError:
            sig.append((str(d), None))
    return tuple(sig)


def load_snapshot_from_disk() -> bool:
    """Carga el snapshot más reciente entre cache/odds (local) y snapshots/ (Actions)."""
    global _loaded_sig
    best = None
    for d in (SNAPSHOT_DIR, REPO_SNAPSHOT_DIR):
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if best is None or meta.get("refreshed_at", "") > best[1].get("refreshed_at", ""):
            best = (d, meta)
    _loaded_sig = _disk_signature()
    if best is None:
        return False
    d, meta = best
    try:
        game_odds = pd.read_parquet(d / "game_odds.parquet")
        props = pd.read_parquet(d / "player_props.parquet")
    except (OSError, ValueError):
        return False
    with _lock:
        _snapshot.update(game_odds=game_odds, player_props=props, meta=meta)
    return True


def _is_fresh(meta: dict | None) -> bool:
    try:
        ts = dt.datetime.fromisoformat(meta["refreshed_at"])
    except (TypeError, KeyError, ValueError):
        return False
    return dt.datetime.now(dt.timezone.utc) - ts <= SNAPSHOT_MAX_AGE


def get_snapshot() -> dict:
    """Última descarga vigente: {"game_odds", "player_props", "meta"} (None si no hay o caducó).

    Se recarga sola cuando cambia un meta.json en disco; un snapshot de más de
    SNAPSHOT_MAX_AGE se ignora para que la app use la API en vivo.
    """
    with _lock:
        if _snapshot["meta"] is None or _disk_signature() != _loaded_sig:
            load_snapshot_from_disk()
        if not _is_fresh(_snapshot["meta"]):
            return {"game_odds": None, "player_props": None, "meta": None}
        return dict(_snapshot)


def on_refresh(callback: Callable[[dict], None]) -> None:
    """Registra una función que se llama tras cada descarga exitosa."""
    _listeners.append(callback)


def refresh_odds(force: bool = False, today: dt.date | None = None) -> dict:
    """Job diario. Devuelve un resumen de lo que hizo."""
    today = today or nfl_today()
    try:
        games = games_today(today)
    except Exception as exc:
        logger.error("No se pudo leer el calendario: %s", exc)
        games = pd.DataFrame()

    if games.empty and not force:
        result = {"status": "skipped", "date": today.isoformat(),
                  "reason": "Hoy no hay partidos NFL: no se consulta The-Odds-API"}
        logger.info(result["reason"])
        return result

    clear_odds_cache()
    try:
        game_odds = op.fetch_game_odds()
        props = _fetch_props_for_today(today)
    except op.OddsAPIError as exc:
        logger.error("Fallo al descargar cuotas: %s", exc)
        return {"status": "error", "date": today.isoformat(), "reason": str(exc)}

    meta = {
        "refreshed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "date": today.isoformat(),
        "games_today": games["game_id"].tolist() if "game_id" in games else [],
        "game_odds_rows": int(len(game_odds)),
        "player_props_rows": int(len(props)),
        "players": int(props["player"].nunique()) if not props.empty else 0,
        "forced": force,
    }
    with _lock:
        _snapshot.update(game_odds=game_odds, player_props=props, meta=meta)
    try:
        _save_snapshot(game_odds, props, meta)
    except OSError as exc:
        logger.warning("No se pudo guardar el snapshot en disco: %s", exc)

    for callback in _listeners:
        try:
            callback(meta)
        except Exception as exc:
            logger.warning("Listener de refresco falló: %s", exc)

    logger.info("Cuotas actualizadas: %s", meta)
    return {"status": "refreshed", **meta}


# --------------------------------------------------------------------------- #
# APScheduler
# --------------------------------------------------------------------------- #
def start_scheduler(hour: int = MORNING_REFRESH_HOUR) -> BackgroundScheduler:
    global _scheduler
    if _scheduler and _scheduler.running:
        return _scheduler
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(refresh_odds, CronTrigger(hour=hour, minute=0), id=JOB_ID,
                       replace_existing=True, coalesce=True, max_instances=1,
                       misfire_grace_time=3600)
    _scheduler.start()
    logger.info("Scheduler activo: refresco diario a las %02d:00", hour)
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
    _scheduler = None


def next_run_time() -> str | None:
    job = _scheduler.get_job(JOB_ID) if _scheduler else None
    return job.next_run_time.isoformat() if job and job.next_run_time else None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(refresh_odds())
