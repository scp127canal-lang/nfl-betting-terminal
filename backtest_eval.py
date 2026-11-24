"""Evaluación y calibración del modelo con las cuotas históricas de backtest.py.

run(): para cada partido ya jugado reconstruye las selecciones EXACTAMENTE como
       la app (líneas principales, Realism Guard, mismas simulaciones) usando solo
       datos anteriores a ese partido, y las califica con el resultado real.
fit(): ajusta la calibración por familia de mercado, la valida "dejando fuera una
       semana" (cada semana se predice con un modelo que nunca la vio), elige
       umbrales, prueba las combinadas y escribe model_calibration.json +
       backtest/report.json.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import backtest as bt
import bet_builder as bb
import prediction_engine as pe
import tracking
from config import NFL_TIMEZONE
from data_pipeline import TEAM_ABBR, load_schedules, load_weekly_stats, normalize_name

log = logging.getLogger("backtest_eval")
ET = ZoneInfo(NFL_TIMEZONE)
LEGS_FILE = bt.BT_DIR / "legs.parquet"
PAIRS_FILE = bt.BT_DIR / "pairs.parquet"
REPORT_FILE = bt.BT_DIR / "report.json"
SEED = 11
PAIR_POOL = 24                       # patas por partido para medir correlación real (SGP)
PRIOR_W = {"team": 0.30, "player": 0.50}
EV_GRID = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
CAP_GRID = [10, 15, 20, 25, 35, 50]
LIFT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
SGP_HAIRCUT = 0.15                   # recorte típico de la casa en combinadas del mismo partido


# =========================================================================== #
# run(): selecciones punto-en-el-tiempo + resultado real
# =========================================================================== #
def _slot_kickoff(slot_id: str) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.strptime(slot_id, "%Y%m%d_%H%M").replace(tzinfo=ET)).tz_convert("UTC")


def _game_index(sched: pd.DataFrame) -> dict:
    s = sched.dropna(subset=["home_score", "away_score"])
    return {(r.home_team, r.away_team, str(r.gameday)[:10]): r for r in s.itertuples(index=False)}


def _match_game(row, idx: dict):
    home, away = TEAM_ABBR.get(row.home_team), TEAM_ABBR.get(row.away_team)
    start = pd.Timestamp(row.commence_time).tz_convert(ET)
    for d in (0, -1, 1):
        g = idx.get((home, away, (start + pd.Timedelta(days=d)).date().isoformat()))
        if g is not None:
            return g
    return None


def _grade(rows: pd.DataFrame, weekly: pd.DataFrame) -> pd.Series:
    """won / lost / void por fila (jugador: stat real de nflverse; equipo: marcador)."""
    w = weekly.set_index(["season", "week", "name_key"]).sort_index()
    out = []
    for r in rows.itertuples(index=False):
        if r.kind == "team":
            hs, as_ = r.home_score, r.away_score
            if r.market == "totals":
                diff = (hs + as_ - r.line) * (1 if r.side == "over" else -1)
            else:
                margin = hs - as_ if r.side == r.home else as_ - hs
                diff = margin + (r.line if r.market == "spreads" and not pd.isna(r.line) else 0.0)
        else:
            key = (r.season, r.week, normalize_name(r.player))
            if key not in w.index or r.stat not in w.columns:
                out.append("void")
                continue
            value = float(pd.to_numeric(w.loc[key, r.stat], errors="coerce").fillna(0).sum()) \
                if isinstance(w.loc[key], pd.DataFrame) else float(w.loc[key, r.stat] or 0)
            diff = (value - r.line) * (1 if r.side == "over" else -1)
        out.append("won" if diff > 0 else "lost" if diff < 0 else "void")
    return pd.Series(out, index=rows.index)


def _leg_row(leg: bb.Leg, g, event_id: str) -> dict:
    mkt = leg.market_prob
    if mkt is None or (isinstance(mkt, float) and math.isnan(mkt)):
        mkt = pe.no_vig_prob(leg.decimal_odds)
    return {"event_id": event_id, "game_id": g.game_id, "season": int(g.season), "week": int(g.week),
            "home": g.home_team, "away": g.away_team, "home_score": float(g.home_score),
            "away_score": float(g.away_score), "kind": leg.kind, "market": leg.market,
            "family": leg.calib_family, "side": leg.side, "line": leg.line, "label": leg.label,
            "player": leg.player, "stat": leg.stat, "team": leg.team, "odds": leg.decimal_odds,
            "mc": leg.model_prob, "mkt": mkt, "n_games": len(leg.history) if leg.kind == "player" else None,
            "trend": leg.trend_hit_rate, "realism_ok": True, "main": bb.is_main_market(leg.market)
            if leg.kind == "player" else True}


def build_game_legs(event_id: str, g, game_rows: pd.DataFrame, prop_rows: pd.DataFrame,
                    weekly: pd.DataFrame, sched: pd.DataFrame) -> list[bb.Leg]:
    """Selecciones de un partido con datos anteriores a él (igual que la app en vivo)."""
    wk_pit = weekly[(weekly["season"] < g.season) | ((weekly["season"] == g.season) & (weekly["week"] < g.week))]
    sc_pit = sched[pd.to_datetime(sched["gameday"]) < pd.Timestamp(str(g.gameday)[:10])]
    legs: list[bb.Leg] = []
    if not game_rows.empty:
        board = tracking._consensus_game_odds(game_rows)
        if not board.empty:
            legs += bb.build_team_legs(board, sc_pit, seed=SEED)
    if not prop_rows.empty:
        legs += bb.build_player_legs(prop_rows, wk_pit, seed=SEED, main_only=True)
    return legs, wk_pit


def run() -> str:
    pe.set_calibration({})                           # el registro guarda prob. crudas (MC y mercado)
    games, props = bt.load_raw()
    seasons = [bt.SEASON - 1, bt.SEASON]
    weekly, sched = load_weekly_stats(seasons), load_schedules(seasons)
    idx = _game_index(sched)

    # Cada evento con su propio snapshot (1 h antes de SU jornada)
    def own_slot(df):
        if df.empty:
            return df
        kick = df["slot"].map(_slot_kickoff)
        start = pd.to_datetime(df["commence_time"], utc=True)
        return df[(start - kick).abs() <= pd.Timedelta(bt.SLOT_GAP)]
    games, props = own_slot(games), own_slot(props)

    events = pd.concat([games, props])[["event_id", "commence_time", "home_team", "away_team"]] \
        .drop_duplicates("event_id")
    rows, pairs = [], []
    for n, ev in enumerate(events.itertuples(index=False)):
        g = _match_game(ev, idx)
        if g is None:
            continue
        legs, wk_pit = build_game_legs(ev.event_id, g, games[games["event_id"] == ev.event_id],
                                       props[props["event_id"] == ev.event_id], weekly, sched)
        base = len(rows)
        rows += [_leg_row(leg, g, ev.event_id) for leg in legs]

        # Correlación real entre patas del mismo partido (para calibrar el lift de SGP)
        cand = [i for i, leg in enumerate(legs) if leg.kind == "team" or
                (bb.is_main_market(leg.market) and 1.40 <= leg.decimal_odds <= 3.0)]
        rng = np.random.default_rng(SEED + n)
        if len(cand) > PAIR_POOL:
            cand = sorted(rng.choice(cand, PAIR_POOL, replace=False).tolist())
        if len(cand) >= 2:
            pool = [legs[i] for i in cand]
            sim = bb.JointSimulator(pool, wk_pit, seed=SEED + n)
            for a in range(len(pool)):
                for b in range(a + 1, len(pool)):
                    if pool[a].unique_key == pool[b].unique_key:
                        continue
                    pairs.append({"i": base + cand[a], "j": base + cand[b],
                                  "mc_i": sim.marginals[a], "mc_j": sim.marginals[b],
                                  "mc_joint": sim.joint((a, b))})
        if n % 10 == 0:
            log.info("run: %s/%s partidos · %s selecciones", n + 1, len(events), len(rows))

    df = pd.DataFrame(rows)
    df["result"] = _grade(df, weekly)
    bt.BT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LEGS_FILE, index=False)
    pd.DataFrame(pairs).to_parquet(PAIRS_FILE, index=False)
    return f"{len(df)} selecciones ({(df['result'] != 'void').sum()} calificadas) · {len(pairs)} parejas SGP"


# =========================================================================== #
# fit(): calibración, validación fuera de muestra y umbrales
# =========================================================================== #
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por línea en el lado 'Over/Sí' (o local en ML/spread), sin anuladas."""
    d = df[df["result"].isin(["won", "lost"])].copy()
    d["y"] = (d["result"] == "won").astype(float)
    flip = d["side"] == "under"
    d.loc[flip, ["mc", "mkt"]] = 1 - d.loc[flip, ["mc", "mkt"]].to_numpy()
    d.loc[flip, "y"] = 1 - d.loc[flip, "y"]
    team_side = d["kind"].eq("team") & d["market"].isin(["h2h", "spreads"])
    d = d[~team_side | (d["side"] == d["home"])]           # ML/spread: lado local
    key = ["event_id", "market", "player", "line"]
    return d.drop_duplicates(key).reset_index(drop=True)


def _features(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frac = np.where(d["n_games"].isna(), 1.0, np.clip(d["n_games"].fillna(10) / pe.REALISM_GAMES, 0, 1))
    return _logit(d["mkt"]), _logit(d["mc"]), frac


def _predict(params, lk, lm, frac):
    a, b, c = params
    z = (a + b * (1 - frac)) * lk + b * frac * lm + c
    return 1 / (1 + np.exp(-z))


VARIANTS = ("completo", "sesgo", "mercado")


def fit_family(d: pd.DataFrame, prior_w: float, symmetric: bool, variant: str = "completo") -> dict:
    """Logística con prior hacia la mezcla fija (evita sobreajuste con pocos datos).

    variant: "completo" = mercado + Monte Carlo + sesgo · "sesgo" = mercado + sesgo
    (el Monte Carlo no aporta) · "mercado" = probabilidad sin vig tal cual.
    """
    if variant == "mercado":
        return {"a": 1.0, "b": 0.0, "c": 0.0, "n": int(len(d)), "model_weight": 0.0, "variant": variant}
    lk, lm, frac = _features(d)
    y = d["y"].to_numpy(float)
    use_b = variant == "completo"
    prior = np.array([1 - prior_w, prior_w, 0.0]) if use_b else np.array([1.0, 0.0, 0.0])
    lam = 25.0                                            # fuerza del prior (≈ 25 observaciones)

    def unpack(x):
        return (x[0], x[1] if use_b else 0.0, 0.0 if symmetric else x[2])

    def loss(x):
        params = unpack(x)
        p = np.clip(_predict(params, lk, lm, frac), 1e-6, 1 - 1e-6)
        ll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        return ll + lam * np.sum((np.asarray(params) - prior) ** 2)

    res = minimize(loss, prior.copy(), method="L-BFGS-B",
                   bounds=[(0.0, 2.0), (0.0, 1.5), (-1.0, 1.0)])
    a, b, c = unpack(res.x)
    return {"a": round(float(a), 4), "b": round(float(max(b, 0.0)), 4),
            "c": round(float(c), 4), "n": int(len(d)),
            "model_weight": round(float(b / (a + b)) if a + b > 0 else 0.0, 3), "variant": variant}


def _family_kind(fam: str) -> str:
    return "team" if fam in pe.TEAM_FAMILIES else "player"


def fit_all(c: pd.DataFrame, variants: dict | None = None) -> dict:
    fams = {}
    for fam, d in c.groupby("family"):
        if len(d) < 40:
            continue
        fams[fam] = fit_family(d, PRIOR_W[_family_kind(fam)], symmetric=fam in ("h2h", "spreads"),
                               variant=(variants or {}).get(fam, "completo"))
    return fams


def lowo_predictions(c: pd.DataFrame, variants: dict | None = None) -> pd.Series:
    """Predicción fuera de muestra: cada semana con una calibración que no la vio."""
    out = pd.Series(np.nan, index=c.index)
    for week in sorted(c["week"].unique()):
        train, test = c[c["week"] != week], c[c["week"] == week]
        fams = fit_all(train, variants)
        for fam, d in test.groupby("family"):
            cal = fams.get(fam)
            if cal is None:
                continue
            lk, lm, frac = _features(d)
            out.loc[d.index] = _predict((cal["a"], cal["b"], cal["c"]), lk, lm, frac)
    return out


def select_variants(c: pd.DataFrame) -> tuple[dict, list]:
    """Por familia, la variante con menor log-loss FUERA de muestra."""
    preds = {v: lowo_predictions(c, {f: v for f in c["family"].unique()}) for v in VARIANTS}
    chosen, rows = {}, []
    for fam, d in c.groupby("family"):
        y = d["y"].to_numpy(float)
        ll = {}
        for v in VARIANTS:
            p = preds[v].loc[d.index]
            if p.notna().all():
                ll[v] = _scores(y, p.to_numpy(float))["log_loss"]
        if not ll:
            continue
        best = min(ll, key=ll.get)
        chosen[fam] = best
        rows.append({"familia": fam, "n": int(len(d)), **{f"ll_{k}": v for k, v in ll.items()}, "elegida": best})
    return chosen, rows


def _fixed_blend(d: pd.DataFrame) -> np.ndarray:
    pe.set_calibration({})
    out = []
    for r in d.itertuples(index=False):
        w = PRIOR_W[_family_kind(r.family)]
        if r.n_games is not None and not pd.isna(r.n_games):
            w = pe.model_weight(int(r.n_games), w)
        out.append(pe.blend_prob(r.mc, r.mkt, w))
    return np.array(out)


def _scores(y, p) -> dict:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {"log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
            "brier": round(float(np.mean((p - y) ** 2)), 4)}


def calibration_table(y, p, bins=(0, .2, .3, .4, .5, .6, .7, .8, 1.0)) -> list[dict]:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.cut(df["p"], bins, include_lowest=True)
    t = df.groupby("bin", observed=True).agg(n=("y", "size"), pred=("p", "mean"), real=("y", "mean"))
    return [{"rango": str(i), "n": int(r.n), "esperado_pct": round(r.pred * 100, 1),
             "real_pct": round(r.real * 100, 1)} for i, r in t.iterrows()]


def bets_from(c: pd.DataFrame, p_over: pd.Series, odds_lookup: pd.DataFrame) -> pd.DataFrame:
    """Por línea, la apuesta del lado con mayor EV usando la prob. ajustada fuera de muestra."""
    rows = []
    o = odds_lookup.set_index(["event_id", "market", "player", "line", "side"])["odds"]
    o = o[~o.index.duplicated()]
    for i, r in c.iterrows():
        p = p_over.loc[i]
        if pd.isna(p):
            continue
        sides = ([(r.side, p, r.y)] if r.kind == "team" and r.market in ("h2h", "spreads") else
                 [("over", p, r.y), ("under", 1 - p, 1 - r.y)])
        if r.kind == "team" and r.market in ("h2h", "spreads"):
            other = r.away if r.side == r.home else r.home
            sides.append((other, 1 - p, 1 - r.y))
        best = None
        for side, ps, ys in sides:
            odds = o.get((r.event_id, r.market, r.player if r.kind == "player" else None, r.line, side))
            if odds is None and r.kind == "team":
                odds = o.get((r.event_id, r.market, None,
                              -r.line if r.market == "spreads" and side != r.side else r.line, side))
            if odds is None or pd.isna(odds):
                continue
            ev = (ps * odds - 1) * 100
            if best is None or ev > best["ev"]:
                best = {"family": r.family, "week": r.week, "side": side, "odds": float(odds), "p": ps,
                        "ev": ev, "won": ys, "profit": (odds - 1) if ys == 1 else -1.0}
        if best:
            rows.append(best)
    return pd.DataFrame(rows)


def _roi(b: pd.DataFrame) -> dict:
    n = len(b)
    return {"n": int(n), "acierto_pct": round(float(b["won"].mean() * 100), 1) if n else None,
            "prob_media_pct": round(float(b["p"].mean() * 100), 1) if n else None,
            "unidades": round(float(b["profit"].sum()), 2) if n else 0.0,
            "roi_pct": round(float(b["profit"].mean() * 100), 1) if n else None}


def choose_thresholds(bets: pd.DataFrame) -> tuple[float, float, list, list]:
    """Umbral de EV: el de mejor ROI fuera de muestra con >= 100 apuestas (mín. 2%).
    Tope: el EV a partir del cual las apuestas dejan de ganar dinero."""
    grid = [{"umbral_ev": t, **_roi(bets[bets["ev"] >= t])} for t in EV_GRID]
    # Robusto: el umbral más bajo (>= 2%) que gana fuera de muestra con >= 300 apuestas.
    # No se persigue el pico de ROI: con pocas apuestas es ruido.
    ok = [g for g in grid if g["umbral_ev"] >= 2 and g["n"] >= 300 and (g["roi_pct"] or 0) > 0]
    min_edge = float(ok[0]["umbral_ev"]) if ok else 5.0
    above = bets[bets["ev"] >= min_edge]
    caps = []
    for cap in [c for c in CAP_GRID if c > min_edge]:
        out = above[above["ev"] > cap]
        caps.append({"tope_ev": cap, **_roi(above[above["ev"] <= cap]),
                     "fuera_roi_pct": _roi(out)["roi_pct"], "fuera_n": int(len(out))})
    cap = 25.0
    for row in caps:                                  # primer tope tras el cual lo de arriba pierde
        if row["fuera_n"] >= 20 and (row["fuera_roi_pct"] or 0) < 0:
            cap = float(row["tope_ev"])
            break
    return min_edge, max(cap, min_edge + 5), grid, caps


def validate_families(bets: pd.DataFrame, min_edge: float, cap: float, weeks: list[int]) -> dict:
    """Familias con ROI > 0 en la 1ª mitad de semanas, probadas en la 2ª mitad (estimación honesta)."""
    sel = bets[(bets["ev"] >= min_edge) & (bets["ev"] <= cap)]
    half = weeks[len(weeks) // 2]
    first, second = sel[sel["week"] < half], sel[sel["week"] >= half]
    picked = [f for f, g in first.groupby("family") if len(g) >= 15 and g["profit"].mean() >= 0.01]
    test = second[second["family"].isin(picked)]
    final = [f for f, g in sel.groupby("family") if len(g) >= 50 and g["profit"].mean() >= 0.01
             and f in picked]
    return {"seleccion_semanas": f"< {half}", "prueba_semanas": f">= {half}",
            "elegidas_1a_mitad": picked, "prueba_2a_mitad": _roi(test),
            "prueba_2a_mitad_todas": _roi(second), "validadas_final": final}


def fit_lift(pairs: pd.DataFrame, legs: pd.DataFrame, p_adj: pd.Series) -> tuple[float, list]:
    """Peso del lift de correlación que mejor predice que AMBAS patas ganen."""
    res = legs["result"]
    ok = pairs[pairs["i"].map(res).isin(["won", "lost"]) & pairs["j"].map(res).isin(["won", "lost"])].copy()
    ok = ok[ok["i"].map(p_adj).notna() & ok["j"].map(p_adj).notna()]
    if ok.empty:
        return bb.LIFT_WEIGHT, []
    y = ((ok["i"].map(res) == "won") & (ok["j"].map(res) == "won")).astype(float).to_numpy()
    pi, pj = ok["i"].map(p_adj).to_numpy(float), ok["j"].map(p_adj).to_numpy(float)
    lift = (ok["mc_joint"] / (ok["mc_i"] * ok["mc_j"]).clip(lower=1e-6)).to_numpy(float)
    rows = []
    for w in LIFT_GRID:
        p = np.clip(pi * pj * lift ** w, 1e-6, np.minimum(pi, pj))
        rows.append({"lift_weight": w, **_scores(y, p), "esperado_pct": round(float(p.mean() * 100), 2),
                     "real_pct": round(float(y.mean() * 100), 2)})
    best = min(rows, key=lambda r: r["log_loss"])
    return float(best["lift_weight"]), rows


def adjusted_side_prob(legs: pd.DataFrame, c: pd.DataFrame, p_over: pd.Series) -> pd.Series:
    """Prob. ajustada (fuera de muestra) de cada fila de `legs`, en su propio lado."""
    key = ["event_id", "market", "player", "line"]
    m = c.assign(p_over=p_over)[key + ["p_over", "side", "home", "kind"]]
    j = legs.reset_index().merge(m, on=key, how="left", suffixes=("", "_c")).set_index("index")
    p = j["p_over"]
    flip = ((j["kind"] == "player") | (j["market"] == "totals")) & (j["side"] == "under")
    team_flip = j["market"].isin(["h2h", "spreads"]) & (j["side"] != j["side_c"])
    p = p.where(~(flip | team_flip), 1 - p)
    return p.reindex(legs.index)


# --------------------------------------------------------------------------- #
# 🎯 Alta probabilidad (regla fija de bet_builder.HIGH_PROB_DEFAULTS)
# --------------------------------------------------------------------------- #
SAFE_TOP_N = 4              # combinadas seguras por tamaño que muestra la app (bb.build_safe_parlays)
BOOTSTRAP_N = 4000


def safe_parlay_rows(sel: pd.DataFrame, size: int, rule: dict) -> pd.DataFrame:
    """Exactamente lo que la app habría recomendado cada semana: mismo pool (las `pool`
    patas más probables, una por jugador+estadística), misma regla de selección
    (bb.select_safe_combos) y patas sin repetir entre combinadas."""
    rows = []
    for wk, g in sel.groupby("week"):
        key = g["player"].fillna(g["event_id"].astype(str) + "|" + g["market"].astype(str)) + "|" + g["family"].astype(str)
        g = g.assign(_k=key).sort_values("p", ascending=False).drop_duplicates("_k").head(rule["pool"])
        combos = bb.select_safe_combos(g["p"].to_numpy(), g["odds"].to_numpy(), g["event_id"].tolist(),
                                       size, SAFE_TOP_N)
        for c in combos:
            x = g.iloc[list(c)]
            won = bool(x["won"].all())
            odds = float(x["odds"].prod())
            rows.append({"week": int(wk), "p": float(x["p"].prod()), "odds": odds, "won": float(won),
                         "profit": odds - 1 if won else -1.0, "labels": " + ".join(x["label"].astype(str))})
    return pd.DataFrame(rows, columns=["week", "p", "odds", "won", "profit", "labels"])


def _week_bootstrap(rows: pd.DataFrame, n: int = BOOTSTRAP_N, seed: int = 0) -> tuple[float, float] | None:
    """Intervalo 90% del ROI remuestreando SEMANAS completas (las patas de una semana no son independientes
    del clima de esa semana, así que se remuestrea la semana entera)."""
    if rows.empty or rows["week"].nunique() < 3:
        return None
    rng = np.random.default_rng(seed)
    by_week = [(g["profit"].sum(), len(g)) for _, g in rows.groupby("week")]
    profit = np.array([b[0] for b in by_week]); count = np.array([b[1] for b in by_week])
    idx = rng.integers(0, len(by_week), size=(n, len(by_week)))
    roi = profit[idx].sum(axis=1) / count[idx].sum(axis=1)
    lo, hi = np.percentile(roi, [5, 95])
    return round(float(lo * 100), 1), round(float(hi * 100), 1)


def _parlay_stats(rows: pd.DataFrame, size: int, single_roi: float) -> dict:
    n = len(rows)
    ci = _week_bootstrap(rows)
    return {"n": int(n), "semanas": int(rows["week"].nunique()) if n else 0,
            "ganadas": int(rows["won"].sum()) if n else 0,
            "prob_media_pct": round(float(rows["p"].mean() * 100), 1) if n else None,
            "acierto_pct": round(float(rows["won"].mean() * 100), 1) if n else None,
            "cuota_media": round(float(rows["odds"].mean()), 2) if n else None,
            "roi_pct": round(float(rows["profit"].mean() * 100), 1) if n else None,
            "roi_ic90": list(ci) if ci else None,
            # Partidos distintos => EV de la combinada = producto de (1 + EV) de las sencillas
            "roi_esperado_por_sencillas_pct": round(((1 + single_roi) ** size - 1) * 100, 1),
            "detalle": rows.to_dict("records")}


def high_prob_backtest(legs: pd.DataFrame, p_side: pd.Series) -> dict:
    """Sencillas y combinadas de alta probabilidad con prob. calibrada fuera de muestra.
    La regla no se optimiza aquí (se fijó antes de mirar los datos): solo se mide."""
    import itertools
    rule = bb.HIGH_PROB_DEFAULTS
    d = legs[legs["result"].isin(["won", "lost"]) & legs["odds"].notna() & legs["main"]].copy()
    d["p"] = p_side.reindex(d.index)
    d = d.dropna(subset=["p"])
    d["won"] = (d["result"] == "won").astype(float)
    d["ev"] = (d["p"] * d["odds"] - 1) * 100
    d["profit"] = np.where(d["won"] == 1, d["odds"] - 1, -1.0)
    sel = d[(d["p"] >= rule["min_prob"]) & (d["ev"] >= rule["min_ev"]) & (d["ev"] <= pe.MAX_CREDIBLE_EV)
            & (d["odds"] >= rule["min_odds"])]
    sel = sel.sort_values("p", ascending=False).drop_duplicates(["event_id", "market", "player", "line"])
    weeks = sorted(int(w) for w in d["week"].unique())
    half = weeks[len(weeks) // 2] if weeks else 0
    first, second = _roi(sel[sel["week"] < half]), _roi(sel[sel["week"] >= half])
    resumen = _roi(sel)
    proven = bool(resumen["n"] >= 100 and (resumen["roi_pct"] or 0) > 0
                  and (first["roi_pct"] or 0) > 0 and (second["roi_pct"] or 0) > 0)
    single_roi = (resumen["roi_pct"] or 0) / 100
    combos = []
    for size in range(2, rule["max_legs"] + 1):
        rows = safe_parlay_rows(sel, size, rule)
        combos.append({"patas": size, **_parlay_stats(rows, size, single_roi)})
    return {"regla": {**rule, "max_credible_ev": pe.MAX_CREDIBLE_EV}, "resumen": resumen,
            "primera_mitad": {"semanas": f"< {half}", **first}, "segunda_mitad": {"semanas": f">= {half}", **second},
            "comprobada": proven,
            "por_familia": [{"familia": f, **_roi(g)} for f, g in sel.groupby("family")],
            "por_semana": [{"semana": int(w), **_roi(g)} for w, g in sel.groupby("week")],
            "sencillas_ic90": list(_week_bootstrap(sel.assign(profit=sel["profit"])) or []) or None,
            "combinadas": combos,
            "nota_combinadas": ("Mismas combinadas que habría mostrado la app cada semana (patas sin repetir). "
                                "IC90 = rango probable del ROI remuestreando semanas. El ROI esperado por "
                                "sencillas es la estimación más estable (partidos distintos = independientes).")}


# --------------------------------------------------------------------------- #
# Combinadas: backtest semana a semana con calibración fuera de muestra
# --------------------------------------------------------------------------- #
def parlay_backtest(cal_by_week: dict[int, dict], variants: dict[str, dict]) -> dict:
    games, props = bt.load_raw()
    seasons = [bt.SEASON - 1, bt.SEASON]
    weekly, sched = load_weekly_stats(seasons), load_schedules(seasons)
    idx = _game_index(sched)
    kick = lambda df: df[(pd.to_datetime(df["commence_time"], utc=True) - df["slot"].map(_slot_kickoff))
                         .abs() <= pd.Timedelta(bt.SLOT_GAP)] if not df.empty else df
    games, props = kick(games), kick(props)
    events = pd.concat([games, props])[["event_id", "commence_time", "home_team", "away_team"]] \
        .drop_duplicates("event_id")
    by_week: dict[int, list] = {}
    for ev in events.itertuples(index=False):
        g = _match_game(ev, idx)
        if g is not None:
            by_week.setdefault(int(g.week), []).append((ev, g))

    results = {name: [] for name in variants}
    for week, evs in sorted(by_week.items()):
        cal = cal_by_week.get(week)
        if cal is None:
            continue
        legs, grade, wk_all = [], {}, None
        for ev, g in evs:                                 # simulación una sola vez por semana
            lg, wk_pit = build_game_legs(ev.event_id, g, games[games["event_id"] == ev.event_id],
                                         props[props["event_id"] == ev.event_id], weekly, sched)
            wk_all = wk_pit
            if lg:
                df = pd.DataFrame([_leg_row(x, g, ev.event_id) for x in lg])
                for x, r in zip(lg, _grade(df, weekly)):
                    grade[x.label] = r
                legs += lg
        if len(legs) < 2:
            continue
        for name, extra in variants.items():
            pe.set_calibration({**cal, "parlay": {**cal.get("parlay", {}), **extra}})
            for x in legs:
                x.__dict__.pop("prob", None)              # recalcular con esta calibración
            for mode in ("mixed", "players", "teams"):
                tiers = bb.build_parlays(legs, weekly=wk_all, top_n=3, seed=SEED, mode=mode)
                for tier, items in tiers.items():
                    for p in items:
                        rs = [grade.get(lb, "void") for lb in p["labels"]]
                        odds_legs = [leg["decimal_odds"] for leg in p["legs"]]
                        if "lost" in rs:
                            profit = -1.0
                        else:
                            won = [o for o, r in zip(odds_legs, rs) if r == "won"]
                            profit = float(np.prod(won)) - 1 if won else 0.0
                        hair = (1 - SGP_HAIRCUT) if p["sgp"] else 1.0
                        results[name].append({"week": week, "mode": mode, "tier": tier, "n_legs": p["n_legs"],
                                              "odds": p["decimal_odds"], "p": p["joint_prob_pct"] / 100,
                                              "ev": p["ev_pct"], "sgp": p["sgp"],
                                              "won": float(profit > 0), "profit": profit,
                                              "profit_haircut": (profit + 1) * hair - 1 if profit > 0 else profit})
        log.info("combinadas semana %s listas", week)
    pe.set_calibration({})
    return {name: pd.DataFrame(rows) for name, rows in results.items()}


def _parlay_summary(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    out = []
    for (tier), g in [*df.groupby("tier"), ("TOTAL", df)]:
        out.append({"nivel": tier, "n": int(len(g)), "acierto_pct": round(float(g["won"].mean() * 100), 1),
                    "prob_media_pct": round(float(g["p"].mean() * 100), 1),
                    "ev_medio_pct": round(float(g["ev"].mean()), 1),
                    "unidades": round(float(g["profit"].sum()), 2),
                    "roi_pct": round(float(g["profit"].mean() * 100), 1),
                    "roi_recorte_sgp_pct": round(float(g["profit_haircut"].mean() * 100), 1)})
    return out


def fit(parlays: bool = True) -> dict:
    legs = pd.read_parquet(LEGS_FILE)
    pairs = pd.read_parquet(PAIRS_FILE) if PAIRS_FILE.exists() else pd.DataFrame()
    c = canonical(legs)
    y = c["y"].to_numpy(float)

    # --- 1) Calibración y comparación fuera de muestra ----------------------
    variants, variant_rows = select_variants(c)
    p_oos = lowo_predictions(c, variants)
    have = p_oos.notna()
    p_fixed = pd.Series(_fixed_blend(c), index=c.index)
    comp, by_family = {}, []
    for name, p in (("mercado", c["mkt"]), ("monte_carlo", c["mc"]), ("mezcla_fija", p_fixed),
                    ("calibrada", p_oos)):
        comp[name] = _scores(y[have], p[have].to_numpy(float))
    for fam, d in c[have].groupby("family"):
        yy = d["y"].to_numpy(float)
        by_family.append({"familia": fam, "n": int(len(d)),
                          "mercado": _scores(yy, d["mkt"])["log_loss"],
                          "monte_carlo": _scores(yy, d["mc"])["log_loss"],
                          "mezcla_fija": _scores(yy, p_fixed[d.index])["log_loss"],
                          "calibrada": _scores(yy, p_oos[d.index])["log_loss"],
                          "real_pct": round(float(yy.mean() * 100), 1),
                          "mc_pct": round(float(d["mc"].mean() * 100), 1),
                          "mercado_pct": round(float(d["mkt"].mean() * 100), 1)})

    # --- 2) Apuestas sencillas fuera de muestra ------------------------------
    graded = legs[legs["result"].isin(["won", "lost"])]
    bets = bets_from(c[have], p_oos[have], graded)
    min_edge, cap, ev_grid, cap_grid = choose_thresholds(bets)
    weeks_all = sorted(int(w) for w in c["week"].unique())
    validation = validate_families(bets, min_edge, cap, weeks_all)
    validated = validation["validadas_final"]
    chosen = bets[(bets["ev"] >= min_edge) & (bets["ev"] <= cap)]
    fam_rows = [{"familia": fam, **_roi(g), "validada": fam in validated} for fam, g in chosen.groupby("family")]
    disabled = sorted(set(c["family"].unique()) - set(validated))
    odds_rows = []
    for lo, hi in ((1.0, 1.7), (1.7, 2.0), (2.0, 2.5), (2.5, 3.5), (3.5, 100)):
        odds_rows.append({"cuotas": f"{lo:.2f}–{hi:.2f}" if hi < 100 else f"> {lo:.2f}",
                          **_roi(chosen[(chosen["odds"] >= lo) & (chosen["odds"] < hi)])})
    side_rows = [{"lado": s, **_roi(g)} for s, g in chosen.assign(
        lado=np.where(chosen["side"].isin(["over", "under"]), chosen["side"], "equipo")).groupby("lado")]
    weekly_rows = [{"semana": int(w), **_roi(g)} for w, g in chosen.groupby("week")]

    # --- 3) Correlación de combinadas (lift) ---------------------------------
    p_side = adjusted_side_prob(legs, c, p_oos)
    lift_w, lift_rows = fit_lift(pairs, legs, p_side) if not pairs.empty else (bb.LIFT_WEIGHT, [])

    # --- 4) Calibración final con todos los datos ----------------------------
    families = fit_all(c, variants)
    calibration = {
        "version": 1,
        "fitted_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "backtest con cuotas históricas de The-Odds-API (1 h antes de cada jornada)",
        "weeks": sorted(int(w) for w in c["week"].unique()),
        "n_lines": int(len(c)),
        "families": families,
        "min_edge_pct": min_edge,
        "max_credible_ev": cap,
        "disabled_families": disabled,
        "validated_families": validated,
        "parlay": {"lift_weight": lift_w},
    }
    high_prob = high_prob_backtest(legs, p_side)
    calibration["high_prob"] = {**bb.HIGH_PROB_DEFAULTS, "comprobada": high_prob["comprobada"],
                                "backtest": {"resumen": high_prob["resumen"],
                                             "primera_mitad": high_prob["primera_mitad"],
                                             "segunda_mitad": high_prob["segunda_mitad"],
                                             "por_familia": high_prob["por_familia"],
                                             "sencillas_ic90": high_prob["sencillas_ic90"],
                                             "combinadas": [{k: v for k, v in c.items() if k != "detalle"}
                                                            for c in high_prob["combinadas"]]}}

    # --- 5) Combinadas con calibración fuera de muestra ----------------------
    parlay_report, parlay_variants = {}, {}
    if parlays:
        cal_by_week = {}
        for week in sorted(c["week"].unique()):
            cal_by_week[int(week)] = {**calibration, "families": fit_all(c[c["week"] != week], variants)}
        pvariants = {"todas_familias": {"validated_only": False},
                     "todas_patas_ev_pos": {"validated_only": False, "min_leg_ev": 0.0},
                     "solo_validadas": {"validated_only": True, "min_leg_ev": min_edge}}
        res = parlay_backtest(cal_by_week, pvariants)
        for name, df in res.items():
            parlay_variants[name] = _parlay_summary(df)
        # Variante con mejor ROI con recorte SGP (lo realista), exigiendo muestra mínima
        scored = {n: float(df["profit_haircut"].mean()) if len(df) >= 20 else -1e9 for n, df in res.items()}
        best = max(scored, key=scored.get)
        calibration["parlay"].update(pvariants[best])
        best_df = res[best]
        tiers_off = []
        if not best_df.empty:
            for tier, g in best_df.groupby("tier"):
                if len(g) >= 20 and g["profit_haircut"].mean() < -0.10:
                    tiers_off.append(tier)
        calibration["parlay"]["disabled_tiers"] = tiers_off
        parlay_report = {"variante_elegida": best, "variantes": parlay_variants,
                         "por_modo": [{"modo": m, **_roi(g.assign(p=g["p"]))} for m, g in best_df.groupby("mode")]
                         if not best_df.empty else []}

    report = {
        "generated_at": calibration["fitted_at"],
        "weeks": calibration["weeks"],
        "n_lineas": int(len(c)), "n_oos": int(have.sum()),
        "comparacion": comp, "por_familia": by_family, "variantes": variant_rows,
        "validacion": validation,
        "calibracion_mercado": calibration_table(y[have], c["mkt"][have]),
        "calibracion_mc": calibration_table(y[have], c["mc"][have]),
        "calibracion_ajustada": calibration_table(y[have], p_oos[have]),
        "sencillas": {"umbral_elegido": min_edge, "tope_elegido": cap, "grid": ev_grid, "topes": cap_grid,
                      "resultado": _roi(chosen), "por_familia": fam_rows, "por_cuota": odds_rows,
                      "por_lado": side_rows, "por_semana": weekly_rows,
                      "sin_modelo_todas": _roi(bets.assign(won=bets["won"]))},
        "lift": lift_rows,
        "combinadas": parlay_report,
        "alta_probabilidad": high_prob,
        "calibracion": calibration,
    }
    pe.CALIBRATION_FILE.write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pe.load_calibration()
    return {"comparacion": comp, "sencillas": report["sencillas"]["resultado"], "umbral": min_edge,
            "tope": cap, "validadas": validated, "validacion": validation, "lift_weight": lift_w,
            "variantes": variants,
            "combinadas": parlay_report.get("variantes", {}).get(parlay_report.get("variante_elegida"), []),
            "familias": {f: v["model_weight"] for f, v in families.items()}}
