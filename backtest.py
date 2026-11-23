"""Backtest del modelo con cuotas históricas reales de The-Odds-API.

Objetivo: medir con cientos de partidos ya jugados si las probabilidades del
modelo están bien calibradas y si las selecciones +EV ganan dinero, y ajustar
los parámetros del modelo con esos datos (model_calibration.json).

    python backtest.py fetch --budget 8000   # descarga (se guarda en backtest/raw, nunca se paga dos veces)
    python backtest.py run                   # evalúa el modelo partido a partido SIN mirar el futuro
    python backtest.py fit                   # calibra, valida fuera de muestra y escribe model_calibration.json
    python backtest.py all --budget 8000

Punto en el tiempo: para cada partido de la semana W solo se usan estadísticas
de semanas anteriores y la cuota de ~1 h antes del kickoff, igual que en vivo.
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

import numpy as np
import pandas as pd

os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
import odds_pipeline as op  # noqa: E402
from config import NFL_TIMEZONE  # noqa: E402

ET = ZoneInfo(NFL_TIMEZONE)
ROOT = Path(__file__).parent
BT_DIR = ROOT / "backtest"
RAW_DIR = BT_DIR / "raw"
LEAD = dt.timedelta(minutes=60)
SLOT_GAP = dt.timedelta(minutes=45)
SEASON = 2026
TEAM_WEEKS = range(1, 12)
PROP_WEEKS = range(11, 2, -1)          # de la más reciente hacia atrás hasta agotar presupuesto
PROP_MARKETS = ("player_pass_yds", "player_rush_yds", "player_reception_yds",
                "player_receptions", "player_anytime_td")
MAX_CALL_COST = 10 * len(PROP_MARKETS)

log = logging.getLogger("backtest")


# --------------------------------------------------------------------------- #
# Descarga histórica (con presupuesto y caché en disco)
# --------------------------------------------------------------------------- #
class Budget:
    def __init__(self, credits: int):
        self.limit = credits
        self.start: int | None = None
        self.spent = 0

    def _remaining(self) -> int | None:
        try:
            return int(float(op.QUOTA["remaining"]))
        except (TypeError, ValueError):
            return None

    def can_spend(self, cost: int) -> bool:
        return self.spent + cost <= self.limit

    def charge(self) -> None:
        rem = self._remaining()
        if rem is None:
            return
        if self.start is None:
            self.start = rem + int(float(op.QUOTA.get("last_cost") or 0))
        self.spent = self.start - rem


def _cached_get(name: str, path: str, budget: Budget, est_cost: int, **params) -> dict | None:
    f = RAW_DIR / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    if not budget.can_spend(est_cost):
        return None
    data = op._get(path, **params)
    budget.charge()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data), encoding="utf-8")
    return data


def season_slots(sched: pd.DataFrame, weeks) -> list[dict]:
    """Jornadas (bloques de kickoff) por semana, con su hora de snapshot (1 h antes)."""
    games = sched[(sched["season"] == SEASON) & sched["week"].isin(list(weeks))
                  & sched["gametime"].notna() & sched["home_score"].notna()].copy()
    kick = pd.to_datetime(games["gameday"].astype(str) + " " + games["gametime"].astype(str))
    games["kickoff"] = kick.dt.tz_localize(ET)
    slots = []
    for week, g in games.sort_values("kickoff").groupby("week"):
        for row in g.itertuples():
            if slots and slots[-1]["week"] == week and row.kickoff - slots[-1]["kickoff"] <= SLOT_GAP:
                slots[-1]["games"].append(row.game_id)
            else:
                slots.append({"week": int(week), "kickoff": row.kickoff, "games": [row.game_id]})
    for s in slots:
        snap = (s["kickoff"] - LEAD).astimezone(dt.timezone.utc)
        s["snapshot"] = snap.strftime("%Y-%m-%dT%H:%M:%SZ")
        s["id"] = s["kickoff"].strftime("%Y%m%d_%H%M")
    return slots


def _slot_events(slot: dict, budget: Budget) -> list[dict]:
    data = _cached_get(f"events_{slot['id']}", f"/historical/sports/{op.SPORT}/events", budget, 1,
                       date=slot["snapshot"])
    if not data:
        return []
    kick = slot["kickoff"].astimezone(dt.timezone.utc)
    out = []
    for e in data.get("data", []):
        start = pd.Timestamp(e["commence_time"])
        if abs(start - kick) <= pd.Timedelta(SLOT_GAP):
            out.append(e)
    return out


def fetch(budget_credits: int) -> dict:
    from data_pipeline import load_schedules
    budget = Budget(budget_credits)
    sched = load_schedules([SEASON])
    summary = {"team_slots": 0, "prop_games": 0, "skipped_budget": 0}

    # 1) Líneas de partido 1 h antes de cada jornada (30 créditos por jornada)
    for slot in season_slots(sched, TEAM_WEEKS):
        if not _slot_events(slot, budget):
            continue
        data = _cached_get(f"game_{slot['id']}", f"/historical/sports/{op.SPORT}/odds", budget, 30,
                           date=slot["snapshot"], regions="us", markets=",".join(op.GAME_MARKETS),
                           oddsFormat="american")
        if data is None:
            summary["skipped_budget"] += 1
            continue
        summary["team_slots"] += 1
        log.info("Líneas de partido %s · gastado %s", slot["id"], budget.spent)

    # 2) Props (5 mercados principales) partido por partido, semana más reciente primero
    for week in PROP_WEEKS:
        for slot in [s for s in season_slots(sched, [week])]:
            for e in _slot_events(slot, budget):
                data = _cached_get(f"props_{e['id']}_{slot['id']}",
                                   f"/historical/sports/{op.SPORT}/events/{e['id']}/odds", budget,
                                   MAX_CALL_COST, date=slot["snapshot"], regions="us",
                                   markets=",".join(PROP_MARKETS), oddsFormat="american")
                if data is None:
                    summary["skipped_budget"] += 1
                    continue
                summary["prop_games"] += 1
            log.info("Props semana %s jornada %s · gastado %s", week, slot["id"], budget.spent)
    summary.update(spent=budget.spent, remaining=op.QUOTA.get("remaining"))
    return summary


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(game_odds, props) históricos aplanados, con la jornada y su hora de snapshot."""
    games, props = [], []
    for f in sorted(RAW_DIR.glob("*.json")):
        name = f.stem
        if not name.startswith(("game_", "props_")):
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        slot_id = name.split("_", 1)[1] if name.startswith("game_") else "_".join(name.split("_")[-2:])
        events = data["data"] if isinstance(data.get("data"), list) else [data.get("data") or {}]
        rows = [r for ev in events for r in op.flatten_event(ev)]
        df = op._frame(rows).assign(slot=slot_id, snapshot=data.get("timestamp"))
        (games if name.startswith("game_") else props).append(df)
    g = pd.concat(games, ignore_index=True) if games else op._frame([])
    p = pd.concat(props, ignore_index=True) if props else op._frame([])
    return g, p


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["fetch", "run", "fit", "all"])
    ap.add_argument("--budget", type=int, default=0, help="créditos máximos para descargar")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("streamlit", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    if args.action in ("fetch", "all"):
        print(json.dumps(fetch(args.budget)))
    if args.action in ("run", "all"):
        import backtest_eval
        print(backtest_eval.run())
    if args.action in ("fit", "all"):
        import backtest_eval
        print(json.dumps(backtest_eval.fit(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
