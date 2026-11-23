"""Actualiza las cuotas ~1 hora antes de cada jornada (bloque de kickoffs) del día.

Pensado para GitHub Actions (cada 15 min en horario de partidos), pero también
se puede correr a mano:

    python refresh_snapshot.py             # decide según el calendario (modo Actions)
    python refresh_snapshot.py --dry-run   # muestra las jornadas de hoy y qué haría
    python refresh_snapshot.py --force     # descarga ya, haya o no jornada

Jornada = partidos cuyo kickoff cae dentro de 45 min del primero del bloque
(ej. domingo: 13:00 · 16:05/16:25 · 20:20, hora del Este).

En cada jornada descarga:
  - Líneas de partido (h2h, spreads, totals) de todos los partidos próximos (~3 créditos).
  - Props de jugador de TODOS los partidos de hoy que aún no empezaron (~17 créditos c/u),
    así las jornadas posteriores también quedan con líneas frescas.
El resultado se guarda en snapshots/ (versionado en git): Streamlit Cloud lo lee sin
gastar créditos por visitante.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")   # sin avisos "No runtime found" fuera de la app
import odds_pipeline as op  # noqa: E402
from config import NFL_TIMEZONE  # noqa: E402
from data_pipeline import get_current_nfl_week, load_schedules  # noqa: E402

ET = ZoneInfo(NFL_TIMEZONE)
OUT_DIR = Path(__file__).parent / "snapshots"
SLOT_GAP = dt.timedelta(minutes=45)        # kickoffs a <45 min del primero = misma jornada
LEAD = dt.timedelta(minutes=60)            # objetivo: 1 hora antes
WINDOW_EARLY = dt.timedelta(minutes=80)    # se acepta correr entre 80 y 15 min antes
WINDOW_LATE = dt.timedelta(minutes=15)     # (los cron de GitHub pueden retrasarse)
KEEP_DAYS = 2                              # props de eventos más viejos se eliminan

log = logging.getLogger("refresh_snapshot")


# --------------------------------------------------------------------------- #
# Jornadas de hoy
# --------------------------------------------------------------------------- #
def todays_slots(now: dt.datetime) -> list[dict]:
    """Bloques de kickoff de hoy (hora del Este) con sus partidos."""
    today = now.astimezone(ET).date()
    season = get_current_nfl_week(today, use_schedule=False)["season"]
    sched = load_schedules([season])
    if sched.empty:
        return []
    games = sched[pd.to_datetime(sched["gameday"]).dt.date == today].dropna(subset=["gametime"])
    if games.empty:
        return []
    kick = pd.to_datetime(games["gameday"].astype(str) + " " + games["gametime"].astype(str))
    games = games.assign(kickoff=kick.dt.tz_localize(ET)).sort_values("kickoff")
    slots: list[dict] = []
    for g in games.itertuples():
        if slots and g.kickoff - slots[-1]["kickoff"] <= SLOT_GAP:
            slots[-1]["games"].append(g.game_id)
        else:
            slots.append({"kickoff": g.kickoff, "games": [g.game_id]})
    for s in slots:
        s["id"] = s["kickoff"].strftime("%Y-%m-%d %H:%M")
        s["refresh_at"] = s["kickoff"] - LEAD
    return slots


def _read_meta() -> dict:
    try:
        return json.loads((OUT_DIR / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def due_slot(now: dt.datetime, slots: list[dict], meta: dict) -> dict | None:
    """Jornada que toca actualizar ahora (ventana 80–15 min antes y aún no hecha)."""
    done = set(meta.get("slots_done", [])) if meta.get("date") == now.astimezone(ET).date().isoformat() else set()
    for s in slots:
        if s["id"] in done:
            continue
        if s["kickoff"] - WINDOW_EARLY <= now <= s["kickoff"] - WINDOW_LATE:
            return s
    return None


# --------------------------------------------------------------------------- #
# Descarga y guardado
# --------------------------------------------------------------------------- #
def _upcoming_today_events(now: dt.datetime) -> pd.DataFrame:
    events = op.fetch_events()                     # gratis
    if events.empty:
        return events
    start = pd.to_datetime(events["commence_time"], utc=True)
    local = start.dt.tz_convert(ET)
    return events[(local.dt.date == now.astimezone(ET).date()) & (start > now)]


def refresh(now: dt.datetime, slot: dict | None) -> dict:
    game_odds = op.fetch_game_odds()
    events = _upcoming_today_events(now)
    frames, failed = [], []
    for event_id in events["id"]:
        try:
            frames.append(op.fetch_event_player_props(event_id))
        except op.OddsAPIError as exc:
            failed.append(event_id)
            log.warning("Props de %s no disponibles: %s", event_id, exc)
    new_props = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else op._frame([])

    # Conserva los props de partidos ya iniciados (del snapshot anterior) y descarta los viejos
    old_path = OUT_DIR / "player_props.parquet"
    if old_path.exists():
        old = pd.read_parquet(old_path)
        cutoff = now - dt.timedelta(days=KEEP_DAYS)
        old = old[~old["event_id"].isin(events["id"])
                  & (pd.to_datetime(old["commence_time"], utc=True) >= cutoff)]
        props = pd.concat([old, new_props], ignore_index=True) if not old.empty else new_props
    else:
        props = new_props

    prev = _read_meta()
    today = now.astimezone(ET).date().isoformat()
    slots_done = prev.get("slots_done", []) if prev.get("date") == today else []
    if slot and slot["id"] not in slots_done:
        slots_done.append(slot["id"])
    meta = {
        "refreshed_at": now.astimezone(dt.timezone.utc).isoformat(timespec="seconds"),
        "date": today,
        "source": "github-actions" if os.getenv("GITHUB_ACTIONS") else "manual",
        "slot": slot["id"] if slot else "manual",
        "slot_games": slot["games"] if slot else [],
        "slots_done": slots_done,
        "events_refreshed": int(len(events)),
        "events_failed": failed,
        "game_odds_rows": int(len(game_odds)),
        "player_props_rows": int(len(props)),
        "players": int(props["player"].nunique()) if not props.empty else 0,
        "credits_remaining": op.QUOTA.get("remaining"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    game_odds.to_parquet(OUT_DIR / "game_odds.parquet", index=False)
    props.to_parquet(OUT_DIR / "player_props.parquet", index=False)
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def _gh_output(**kv) -> None:
    path = os.getenv("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            for k, v in kv.items():
                fh.write(f"{k}={v}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="descargar ahora aunque no toque jornada")
    ap.add_argument("--dry-run", action="store_true", help="solo mostrar jornadas y decisión")
    ap.add_argument("--now", help="simular otra hora (ISO, ej. 2026-11-29T12:05-05:00)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("streamlit", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    now = dt.datetime.fromisoformat(args.now) if args.now else dt.datetime.now(dt.timezone.utc)
    slots = todays_slots(now)
    meta = _read_meta()
    print(f"Ahora (ET): {now.astimezone(ET):%Y-%m-%d %H:%M}")
    if slots:
        done = set(meta.get("slots_done", [])) if meta.get("date") == now.astimezone(ET).date().isoformat() else set()
        for s in slots:
            mark = "✅ hecha" if s["id"] in done else f"actualiza ~{s['refresh_at']:%H:%M}"
            print(f"  Jornada {s['kickoff']:%H:%M} ET · {len(s['games'])} partido(s) · {mark}")
    else:
        print("  Hoy no hay partidos NFL.")

    slot = due_slot(now, slots, meta)
    if not (slot or args.force):
        print("Nada que hacer ahora (0 créditos).")
        _gh_output(changed="false")
        return 0
    if args.dry_run:
        print(f"Tocaría actualizar: {slot['id'] if slot else 'forzado'}")
        _gh_output(changed="false")
        return 0

    try:
        result = refresh(now, slot)
    except op.OddsAPIError as exc:
        print(f"ERROR The-Odds-API: {exc}")
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    # Seguimiento: guarda las selecciones del modelo para esta jornada (tracking/picks.csv)
    try:
        import tracking
        n = tracking.log_slot_picks(now, result["slot"], pd.read_parquet(OUT_DIR / "game_odds.parquet"),
                                    pd.read_parquet(OUT_DIR / "player_props.parquet"))
        print(f"Seguimiento: {n} selección(es) nuevas registradas en tracking/picks.csv")
    except Exception as exc:                      # nunca bloquea la actualización de cuotas
        log.warning("No se pudieron registrar las selecciones: %s", exc)
    _gh_output(changed="true", slot=result["slot"].replace(" ", "_"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
