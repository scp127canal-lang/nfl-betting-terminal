"""Seguimiento de resultados: ¿el +EV del modelo se convierte en ganancias reales?

1. `log_slot_picks` (lo llama refresh_snapshot.py ~1 h antes de cada jornada):
   guarda las selecciones que el modelo recomienda para los partidos de esa
   jornada, con la cuota de ese momento:
     - Sencillas +EV creíbles (EV ajustado entre MIN_EDGE_PCT y MAX_CREDIBLE_EV),
       solo líneas principales.
     - La mejor combinada de cada nivel de riesgo (🟢 BAJO · 🟡 MODERADO · 🔴 ALTO).
2. `grade_picks` (workflow diario): califica las pendientes con los resultados
   reales de nflverse (marcador final y estadísticas de cada jugador).

Todo se guarda en tracking/picks.csv (versionado en git), con stake plano de
1 unidad por selección para medir ROI y acierto real vs probabilidad estimada.

    python tracking.py grade      # califica las pendientes
    python tracking.py summary    # resumen en consola
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
import bet_builder as bb  # noqa: E402
import prediction_engine as pe  # noqa: E402
from config import NFL_TIMEZONE  # noqa: E402
from odds_pipeline import is_main_market  # noqa: E402
from data_pipeline import (TEAM_ABBR, get_current_nfl_week, load_schedules,  # noqa: E402
                           load_weekly_stats, normalize_name)

ET = ZoneInfo(NFL_TIMEZONE)
TRACK_DIR = Path(__file__).parent / "tracking"
PICKS_FILE = TRACK_DIR / "picks.csv"
SLOT_HORIZON = dt.timedelta(minutes=100)   # partidos de la jornada: empiezan en < 100 min
MAX_SINGLES = 12                           # sencillas por jornada (las de mayor EV)
SINGLE_MAX_ODDS = 2.50                     # backtest: sin ventaja por encima de 2.50
VOID_AFTER = dt.timedelta(days=6)          # sin datos tras 6 días -> anulada (no jugó / no publicado)

COLUMNS = ["pick_id", "logged_at", "slot", "kind", "tier", "game_date", "events", "label", "n_legs",
           "decimal_odds", "prob_pct", "mc_prob_pct", "implied_prob_pct", "ev_pct", "legs",
           "status", "legs_result", "graded_at", "profit", "closing_odds", "clv_pct", "closing_at"]

log = logging.getLogger("tracking")


# --------------------------------------------------------------------------- #
# Archivo
# --------------------------------------------------------------------------- #
def load_picks(path: Path | None = None) -> pd.DataFrame:
    path = path or PICKS_FILE
    try:
        df = pd.read_csv(path, dtype={"pick_id": str})
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=COLUMNS).astype(object)
    df = df.reindex(columns=COLUMNS)
    text = ["status", "legs_result", "graded_at", "closing_at", "slot", "tier", "kind", "events", "label", "legs"]
    df[text] = df[text].astype(object)                 # columnas vacías no deben quedar como float
    return df


def save_picks(df: pd.DataFrame, path: Path | None = None) -> None:
    path = path or PICKS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reindex(columns=COLUMNS).to_csv(path, index=False)


def _pick_id(kind: str, labels: list[str], game_date: str) -> str:
    raw = f"{kind}|{game_date}|" + "|".join(sorted(labels))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Registro de selecciones (antes de la jornada)
# --------------------------------------------------------------------------- #
def _event_info(game_odds: pd.DataFrame, props: pd.DataFrame) -> dict[str, dict]:
    cols = ["event_id", "commence_time", "home_team", "away_team"]
    frames = [df[cols] for df in (game_odds, props) if df is not None and not df.empty and set(cols) <= set(df)]
    if not frames:
        return {}
    ev = pd.concat(frames).drop_duplicates("event_id")
    out = {}
    for r in ev.itertuples(index=False):
        start = pd.Timestamp(r.commence_time)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        out[r.event_id] = {"start": start, "date": start.tz_convert(ET).date().isoformat(),
                           "home": TEAM_ABBR.get(r.home_team), "away": TEAM_ABBR.get(r.away_team)}
    return out


def _consensus_game_odds(game_odds: pd.DataFrame) -> pd.DataFrame:
    """Una cuota por outcome y partido: línea más ofrecida y cuota mediana entre casas."""
    rows = []
    base_cols = ["event_id", "commence_time", "home_team", "away_team"]
    for (event_id, market, outcome), g in game_odds[game_odds["market"].isin(["h2h", "spreads", "totals"])
                                                    ].groupby(["event_id", "market", "outcome"]):
        if market == "h2h":
            line, odds = np.nan, g["decimal_odds"].median()
        else:
            if g["line"].dropna().empty:
                continue
            line = g["line"].mode().iloc[0]
            odds = g.loc[g["line"] == line, "decimal_odds"].median()
        rows.append({**g.iloc[0][base_cols].to_dict(), "market": market, "outcome": outcome,
                     "line": line, "decimal_odds": round(float(odds), 2)})
    return pd.DataFrame(rows)


def _leg_record(leg: bb.Leg, info: dict[str, dict]) -> dict:
    ev = info.get(leg.event_id, {})
    return {"label": leg.label, "kind": leg.kind, "market": leg.market, "side": leg.side,
            "line": leg.line, "player": leg.player, "team": leg.team, "stat": leg.stat,
            "decimal_odds": leg.decimal_odds, "prob_pct": round(leg.prob * 100, 2),
            "event_id": leg.event_id, "home": ev.get("home"), "away": ev.get("away"),
            "date": ev.get("date")}


def select_picks(legs: list[bb.Leg], weekly: pd.DataFrame, info: dict[str, dict],
                 slot: str, now: dt.datetime) -> list[dict]:
    """Sencillas +EV creíbles + la mejor combinada de cada nivel de riesgo."""
    logged_at = now.astimezone(dt.timezone.utc).isoformat(timespec="seconds")
    picks = []

    best: dict[tuple, bb.Leg] = {}
    for leg in legs:
        if leg.kind == "player" and not is_main_market(leg.market):
            continue
        if not leg.plus_ev or leg.decimal_odds > SINGLE_MAX_ODDS:
            continue
        key = leg.unique_key
        if key not in best or leg.ev_pct > best[key].ev_pct:
            best[key] = leg
    for leg in sorted(best.values(), key=lambda l: l.ev_pct, reverse=True)[:MAX_SINGLES]:
        rec = _leg_record(leg, info)
        picks.append({"pick_id": _pick_id("single", [leg.label], rec["date"] or ""), "logged_at": logged_at,
                      "slot": slot, "kind": "single", "tier": "SENCILLA", "game_date": rec["date"],
                      "events": leg.event_id, "label": leg.label, "n_legs": 1,
                      "decimal_odds": leg.decimal_odds, "prob_pct": round(leg.prob * 100, 2),
                      "mc_prob_pct": round(leg.model_prob * 100, 2),
                      "implied_prob_pct": leg.implied_prob_pct, "ev_pct": leg.ev_pct,
                      "legs": json.dumps([rec], ensure_ascii=False)})

    # 🎯 Alta probabilidad: mismas sencillas y combinadas seguras que muestra la app
    for leg in bb.high_prob_legs(legs, MAX_SINGLES):
        rec = _leg_record(leg, info)
        picks.append({"pick_id": _pick_id("single_hp", [leg.label], rec["date"] or ""), "logged_at": logged_at,
                      "slot": slot, "kind": "single", "tier": "ALTA_PROB", "game_date": rec["date"],
                      "events": leg.event_id, "label": leg.label, "n_legs": 1,
                      "decimal_odds": leg.decimal_odds, "prob_pct": round(leg.prob * 100, 2),
                      "mc_prob_pct": round(leg.model_prob * 100, 2),
                      "implied_prob_pct": leg.implied_prob_pct, "ev_pct": leg.ev_pct,
                      "legs": json.dumps([rec], ensure_ascii=False)})
    by_label_all = {leg.label: leg for leg in legs}
    for size, items in bb.build_safe_parlays(legs).items():
        for p in items:
            chosen = [by_label_all[lb] for lb in p["labels"] if lb in by_label_all]
            recs = [_leg_record(leg, info) for leg in chosen]
            date = min((r["date"] for r in recs if r["date"]), default="")
            picks.append({"pick_id": _pick_id("safe", p["labels"], date), "logged_at": logged_at,
                          "slot": slot, "kind": "parlay", "tier": "SEGURA", "game_date": date,
                          "events": ";".join(sorted({r["event_id"] for r in recs})),
                          "label": " + ".join(p["labels"]), "n_legs": p["n_legs"],
                          "decimal_odds": p["decimal_odds"], "prob_pct": p["joint_prob_pct"],
                          "mc_prob_pct": None, "implied_prob_pct": p["implied_prob_pct"],
                          "ev_pct": p["ev_pct"], "legs": json.dumps(recs, ensure_ascii=False)})

    if len(legs) >= 2:
        tiers = bb.build_parlays(legs, weekly=weekly, top_n=1, seed=7, mode="mixed")
        by_label = {leg.label: leg for leg in legs}
        for tier, items in tiers.items():
            for p in items:
                chosen = [by_label[lb] for lb in p["labels"] if lb in by_label]
                recs = [_leg_record(leg, info) for leg in chosen]
                date = min((r["date"] for r in recs if r["date"]), default="")
                picks.append({"pick_id": _pick_id("parlay", p["labels"], date), "logged_at": logged_at,
                              "slot": slot, "kind": "parlay", "tier": tier, "game_date": date,
                              "events": ";".join(sorted({r["event_id"] for r in recs})),
                              "label": " + ".join(p["labels"]), "n_legs": p["n_legs"],
                              "decimal_odds": p["decimal_odds"], "prob_pct": p["joint_prob_pct"],
                              "mc_prob_pct": None, "implied_prob_pct": p["implied_prob_pct"],
                              "ev_pct": p["ev_pct"], "legs": json.dumps(recs, ensure_ascii=False)})
    return picks


def log_slot_picks(now: dt.datetime, slot: str, game_odds: pd.DataFrame, props: pd.DataFrame,
                   weekly: pd.DataFrame | None = None, schedules: pd.DataFrame | None = None,
                   horizon: dt.timedelta = SLOT_HORIZON) -> int:
    """Guarda las selecciones del modelo para los partidos que empiezan en < `horizon`.
    Devuelve cuántas selecciones nuevas se añadieron (dedupe por pick_id)."""
    info = _event_info(game_odds, props)
    now_ts = pd.Timestamp(now).tz_convert("UTC")
    slot_events = {e for e, v in info.items() if now_ts < v["start"] <= now_ts + horizon}
    if not slot_events:
        return 0
    if weekly is None or schedules is None:
        season = get_current_nfl_week(now.astimezone(ET).date(), use_schedule=False)["season"]
        weekly = load_weekly_stats([season - 1, season]) if weekly is None else weekly
        schedules = load_schedules([season - 1, season]) if schedules is None else schedules

    legs: list[bb.Leg] = []
    if props is not None and not props.empty:
        legs += bb.build_player_legs(props[props["event_id"].isin(slot_events)], weekly, seed=7, main_only=True)
    if game_odds is not None and not game_odds.empty:
        board = _consensus_game_odds(game_odds[game_odds["event_id"].isin(slot_events)])
        if not board.empty:
            legs += bb.build_team_legs(board, schedules, seed=7)

    new = pd.DataFrame(select_picks(legs, weekly, info, slot, now), columns=COLUMNS)
    if new.empty:
        return 0
    old = load_picks()
    new = new[~new["pick_id"].isin(old["pick_id"])].assign(status="pending")
    if new.empty:
        return 0
    save_picks(pd.concat([old, new], ignore_index=True) if not old.empty else new)
    return len(new)


# --------------------------------------------------------------------------- #
# CLV: cuota tomada vs línea de cierre (~10 min antes del kickoff)
# --------------------------------------------------------------------------- #
def closing_markets(event_ids: set[str]) -> dict[str, set[str]]:
    """Mercados de jugador que hacen falta al cierre, por evento (solo los de selecciones registradas)."""
    picks = load_picks()
    need: dict[str, set[str]] = {}
    for r in picks[picks["closing_at"].isna()].itertuples(index=False):
        for leg in json.loads(r.legs):
            if leg.get("event_id") in event_ids and leg.get("kind") == "player":
                need.setdefault(leg["event_id"], set()).add(leg["market"])
    return need


def _close_prob(leg: dict, game_odds: pd.DataFrame, props: pd.DataFrame) -> tuple[float | None, float | None]:
    """(prob. sin vig de cierre del lado apostado, cuota mediana de cierre) a la MISMA línea."""
    if leg["kind"] == "team":
        df = game_odds[(game_odds["event_id"] == leg["event_id"]) & (game_odds["market"] == leg["market"])]
        if df.empty:
            return None, None
        if leg["market"] == "totals":
            df = df[df["line"] == leg["line"]]
            mine = df[df["outcome"].str.lower() == leg["side"]]
            other = df[df["outcome"].str.lower() != leg["side"]]
        else:
            df = df.assign(abbr=df["outcome"].map(TEAM_ABBR))
            mine = df[df["abbr"] == leg["side"]]
            if leg["market"] == "spreads":
                mine = mine[mine["line"] == leg["line"]]
                other = df[(df["abbr"] != leg["side"]) & (df["line"] == -leg["line"])]
            else:
                other = df[df["abbr"] != leg["side"]]
    else:
        df = props[(props["event_id"] == leg["event_id"]) & (props["market"] == leg["market"])
                   & (props["player"] == leg["player"]) & (props["line"] == leg["line"])]
        mine = df[df["outcome"].str.lower() == leg["side"]]
        other = df[df["outcome"].str.lower() != leg["side"]]
    if mine.empty:
        return None, None
    o_m = float(mine["decimal_odds"].median())
    o_o = float(other["decimal_odds"].median()) if not other.empty else None
    return pe.no_vig_prob(o_m, o_o), o_m


def record_closing(game_odds: pd.DataFrame, props: pd.DataFrame, event_ids: set[str],
                   now: dt.datetime | None = None) -> int:
    """Guarda la cuota de cierre y el CLV de cada selección de esos partidos.

    CLV % = p_cierre_sin_vig × cuota_tomada − 1: el EV de tu apuesta medido contra
    la línea final del mercado (la mejor estimación de la probabilidad real). En
    combinadas se multiplican las patas. Positivo de forma sostenida = ventaja real.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    picks = load_picks()
    todo = picks[picks["closing_at"].isna() & picks["events"].fillna("").apply(
        lambda e: bool(set(str(e).split(";")) & event_ids))]
    done = 0
    for idx, r in todo.iterrows():
        legs = json.loads(r["legs"])
        factors, close_odds = [], []
        for leg in legs:
            p, o = _close_prob(leg, game_odds, props)
            if p is None:
                factors = None
                break
            factors.append(p * float(leg["decimal_odds"]))
            close_odds.append(o)
        if not factors:
            continue
        picks.loc[idx, ["closing_odds", "clv_pct", "closing_at"]] = [
            round(float(np.prod(close_odds)), 3), round((float(np.prod(factors)) - 1) * 100, 2),
            now.isoformat(timespec="seconds")]
        done += 1
    if done:
        save_picks(picks)
    return done


# --------------------------------------------------------------------------- #
# Calificación (después de los partidos)
# --------------------------------------------------------------------------- #
def _find_game(sched: pd.DataFrame, home: str, away: str, date: str) -> pd.Series | None:
    g = sched[(sched["home_team"] == home) & (sched["away_team"] == away)]
    if g.empty or not date:
        return None
    days = (pd.to_datetime(g["gameday"]) - pd.Timestamp(date)).abs()
    g = g[days <= pd.Timedelta(days=1)]
    return None if g.empty else g.iloc[0]


def grade_leg(leg: dict, sched: pd.DataFrame, weekly: pd.DataFrame, now: dt.datetime) -> str:
    """'won' · 'lost' · 'void' (push o no jugó) · 'pending' (sin datos aún)."""
    game = _find_game(sched, leg.get("home"), leg.get("away"), leg.get("date"))
    if game is None or pd.isna(game.get("home_score")) or pd.isna(game.get("away_score")):
        return "pending"
    hs, as_ = float(game["home_score"]), float(game["away_score"])
    line = leg.get("line")
    line = None if line is None or (isinstance(line, float) and np.isnan(line)) else float(line)

    if leg["kind"] == "team":
        if leg["market"] == "totals":
            diff = (hs + as_ - line) * (1 if leg["side"] == "over" else -1)
        else:
            margin = hs - as_ if leg["side"] == leg.get("home") else as_ - hs
            diff = margin + (line if leg["market"] == "spreads" else 0.0)
        return "won" if diff > 0 else "lost" if diff < 0 else "void"

    stat = leg.get("stat")
    wk = weekly[(weekly["season"] == game["season"]) & (weekly["week"] == game["week"])]
    teams = {game["home_team"], game["away_team"]}
    team_col = "recent_team" if "recent_team" in wk else "team" if "team" in wk else None
    published = wk[wk[team_col].isin(teams)] if team_col else wk
    played = now - pd.Timestamp(leg["date"]).tz_localize(ET).to_pydatetime() if leg.get("date") else dt.timedelta(0)
    if published.empty or stat not in wk:
        return "void" if played > VOID_AFTER else "pending"
    row = published[published["name_key"] == normalize_name(leg["player"])]
    if row.empty:
        return "void"                                  # no jugó: la casa anula la selección
    value = float(pd.to_numeric(row[stat], errors="coerce").fillna(0).sum())
    diff = (value - line) * (1 if leg["side"] == "over" else -1)
    return "won" if diff > 0 else "lost" if diff < 0 else "void"


def grade_pick(pick: pd.Series, sched: pd.DataFrame, weekly: pd.DataFrame, now: dt.datetime) -> tuple[str, list, float | None]:
    legs = json.loads(pick["legs"])
    results = [grade_leg(leg, sched, weekly, now) for leg in legs]
    if "lost" in results:
        return "lost", results, -1.0
    if "pending" in results:
        return "pending", results, None
    won = [leg for leg, r in zip(legs, results) if r == "won"]
    if not won:
        return "void", results, 0.0
    if len(won) == len(legs):
        odds = float(pick["decimal_odds"])             # cuota registrada (combinada completa)
    else:
        odds = float(np.prod([leg["decimal_odds"] for leg in won]))   # patas anuladas: se recalcula
    return "won", results, round(odds - 1, 4)


def grade_picks(now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    picks = load_picks()
    pending = picks[picks["status"].fillna("pending") == "pending"]
    if pending.empty:
        return {"graded": 0, "pending": 0}
    seasons = sorted({int(str(d)[:4]) for d in pending["game_date"].dropna()} or
                     {get_current_nfl_week(use_schedule=False)["season"]})
    seasons = sorted({s for y in seasons for s in (y - 1, y)})
    sched = load_schedules(seasons)
    weekly = load_weekly_stats(seasons)
    graded = 0
    for idx, pick in pending.iterrows():
        status, results, profit = grade_pick(pick, sched, weekly, now)
        if status == "pending":
            continue
        picks.loc[idx, ["status", "legs_result", "graded_at", "profit"]] = [
            status, json.dumps(results), now.isoformat(timespec="seconds"), profit]
        graded += 1
    save_picks(picks)
    return {"graded": graded, "pending": int((picks["status"] == "pending").sum())}


# --------------------------------------------------------------------------- #
# Resumen de rendimiento
# --------------------------------------------------------------------------- #
GROUP_ORDER = ["SENCILLA", "ALTA_PROB", "SEGURA", "BAJO", "MODERADO", "ALTO"]


def summarize(picks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Por grupo (sencillas y cada nivel de combinadas) y total: récord, acierto, ROI."""
    picks = load_picks() if picks is None else picks
    if picks.empty:
        return pd.DataFrame()
    picks = picks.assign(profit=pd.to_numeric(picks["profit"], errors="coerce"),
                         prob_pct=pd.to_numeric(picks["prob_pct"], errors="coerce"),
                         ev_pct=pd.to_numeric(picks["ev_pct"], errors="coerce"),
                         clv_pct=pd.to_numeric(picks["clv_pct"], errors="coerce"))
    rows = []
    for name, g in [*((t, picks[picks["tier"] == t]) for t in GROUP_ORDER), ("TOTAL", picks)]:
        if g.empty:
            continue
        settled = g[g["status"].isin(["won", "lost"])]
        n = len(settled)
        won = int((settled["status"] == "won").sum())
        profit = float(settled["profit"].sum()) if n else 0.0
        rows.append({"grupo": name, "selecciones": len(g), "pendientes": int((g["status"] == "pending").sum()),
                     "ganadas": won, "perdidas": n - won,
                     "anuladas": int((g["status"] == "void").sum()),
                     "acierto_pct": round(won / n * 100, 1) if n else None,
                     "prob_esperada_pct": round(float(settled["prob_pct"].mean()), 1) if n else None,
                     "ev_esperado_pct": round(float(settled["ev_pct"].mean()), 1) if n else None,
                     "unidades": round(profit, 2),
                     "roi_pct": round(profit / n * 100, 1) if n else None,
                     "clv_medio_pct": round(float(g["clv_pct"].mean()), 2) if g["clv_pct"].notna().any() else None,
                     "clv_positivo_pct": round(float((g["clv_pct"] > 0).mean() * 100), 1)
                     if g["clv_pct"].notna().any() else None})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["grade", "summary"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("streamlit", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    if args.action == "grade":
        res = grade_picks()
        print(json.dumps(res))
        out = os.getenv("GITHUB_OUTPUT")
        if out:
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(f"changed={'true' if res['graded'] else 'false'}\n")
    summary = summarize()
    print(summary.to_string(index=False) if not summary.empty else "Sin selecciones registradas todavía.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
