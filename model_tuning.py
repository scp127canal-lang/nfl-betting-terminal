"""Entrenamiento del Monte Carlo de jugadores SIN overfitting.

Separación estricta en el tiempo (cada bloque se usa para UNA sola cosa):
  1. ENTRENAMIENTO  temporada 2025 (todas las semanas) contra el resultado real de
                    cada jugador. Métrica: CRPS, que puntúa la distribución completa
                    (no hace falta la cuota, así hay miles de partidos por mercado).
  2. VALIDACIÓN     2026 semanas 3–8 con las líneas reales de las casas: log-loss de
                    P(Over línea) del modelo nuevo vs el actual.
  3. PRUEBA FINAL   2026 semanas 9–11, intocable hasta el final. La calibración se
                    ajusta en validación y se mide aquí una sola vez. Se adopta el
                    modelo nuevo solo si gana en esta prueba.

Parámetros por mercado (pocos y con rango acotado):
  window    partidos de historial (6–17)
  decay     peso de cada partido vs el siguiente más reciente (0.70–1.00)
  disp      multiplicador del ancho de la distribución (1.0–2.0): corrige sobreconfianza
  opp_beta  ajuste por defensa rival: media × factor_rival^beta (0 = sin ajuste)

    python model_tuning.py           # entrena, valida, prueba y escribe model_params.json si gana
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
import prediction_engine as pe  # noqa: E402
from data_pipeline import load_weekly_stats, normalize_name  # noqa: E402

ROOT = Path(__file__).parent
PARAMS_FILE = ROOT / "model_params.json"
TUNING_REPORT = ROOT / "backtest" / "tuning_report.json"
import backtest as _bt  # noqa: E402

TEST_SEASON = _bt.SEASON
TRAIN_SEASON = TEST_SEASON - 1
HOLDOUT_N = 3                       # últimas 3 semanas con líneas reales = prueba final intocable
VALID_WEEKS: list[int] = []         # se fijan al leer backtest/legs.parquet
HOLDOUT_WEEKS: list[int] = []


def split_weeks(legs: pd.DataFrame) -> None:
    weeks = sorted(int(w) for w in legs.loc[legs["kind"] == "player", "week"].unique())
    VALID_WEEKS[:], HOLDOUT_WEEKS[:] = weeks[:-HOLDOUT_N], weeks[-HOLDOUT_N:]
N_TUNE = 1500                       # simulaciones por predicción al entrenar (mismas semillas en todas)
OPP_WINDOW, OPP_SHRINK = 8, 4.0     # defensa rival: últimos 8 partidos, encogidos hacia la media de liga

FAMILIES = {                          # stat -> (tipo, columna de elegibilidad, mínimo de su media previa)
    "receiving_yards": ("yards", "receiving_yards", 15.0),
    "rushing_yards": ("yards", "rushing_yards", 15.0),
    "passing_yards": ("yards", "attempts", 15.0),
    "receptions": ("count", "receptions", 1.5),
    "total_tds": ("count", "touches", 4.0),
}
GRID = {
    "window": [6, 8, 10, 13, 17],
    "decay": [0.70, 0.80, 0.85, 0.90, 0.95, 1.00],
    "disp": [1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0],
    "opp_beta": [0.0, 0.25, 0.5, 0.75, 1.0],
}

log = logging.getLogger("model_tuning")


@dataclass(frozen=True)
class SimParams:
    window: int = 10
    decay: float = pe.RECENCY_DECAY
    disp: float = 1.0
    opp_beta: float = 0.0

    @staticmethod
    def current() -> "SimParams":
        return SimParams()


# --------------------------------------------------------------------------- #
# Datos punto-en-el-tiempo
# --------------------------------------------------------------------------- #
def load_weekly() -> pd.DataFrame:
    w = load_weekly_stats([TRAIN_SEASON - 1, TRAIN_SEASON, TEST_SEASON])
    w = w[w["season_type"] == "REG"].copy()
    w["touches"] = w[["carries", "receptions"]].fillna(0).sum(axis=1)
    w["order"] = w["season"] * 100 + w["week"]
    return w.sort_values(["player_id", "order"]).reset_index(drop=True)


def opponent_factors(w: pd.DataFrame, stat: str) -> pd.Series:
    """Factor de la defensa rival para cada fila (stat permitido a esa posición / media de liga),
    calculado SOLO con partidos anteriores y encogido hacia 1."""
    allowed = (w.groupby(["opponent_team", "position_group", "order"])[stat].sum()
               .rename("allowed").reset_index().sort_values("order"))
    g = allowed.groupby(["opponent_team", "position_group"])["allowed"]
    allowed["def_mean"] = g.transform(lambda s: s.shift(1).rolling(OPP_WINDOW, min_periods=1).mean())
    allowed["def_n"] = g.transform(lambda s: s.shift(1).rolling(OPP_WINDOW, min_periods=1).count())
    league = (allowed.groupby(["position_group", "order"])["allowed"].mean().rename("lg").reset_index()
              .sort_values("order"))
    league["league"] = league.groupby("position_group")["lg"].transform(
        lambda s: s.shift(1).expanding().mean())
    allowed = allowed.merge(league[["position_group", "order", "league"]], on=["position_group", "order"])
    n = allowed["def_n"].fillna(0)
    shrunk = (n * allowed["def_mean"].fillna(allowed["league"]) + OPP_SHRINK * allowed["league"]) / (n + OPP_SHRINK)
    allowed["factor"] = (shrunk / allowed["league"]).clip(0.6, 1.6).fillna(1.0)
    m = w[["opponent_team", "position_group", "order"]].merge(
        allowed[["opponent_team", "position_group", "order", "factor"]], how="left",
        on=["opponent_team", "position_group", "order"])
    return pd.Series(m["factor"].fillna(1.0).to_numpy(), index=w.index)


def build_rows(w: pd.DataFrame, stat: str, season: int, weeks=None) -> pd.DataFrame:
    """Una fila por jugador-partido objetivo con su historial previo (hasta 17 partidos)."""
    kind, elig_col, elig_min = FAMILIES[stat]
    opp = opponent_factors(w, stat)
    rows = []
    for pid, g in w.groupby("player_id", sort=False):
        vals = pd.to_numeric(g[stat], errors="coerce").fillna(0).to_numpy(float)
        elig = pd.to_numeric(g[elig_col], errors="coerce").fillna(0).to_numpy(float)
        seasons, wks, idx = g["season"].to_numpy(), g["week"].to_numpy(), g.index.to_numpy()
        for i in range(len(g)):
            if seasons[i] != season or (weeks is not None and wks[i] not in weeks) or i < 3:
                continue
            hist = vals[max(0, i - 17):i]
            if elig[max(0, i - 5):i].mean() < elig_min:
                continue                               # jugador sin volumen: no tendría prop
            rows.append({"player_id": pid, "name_key": g["name_key"].iat[i], "season": int(seasons[i]),
                         "week": int(wks[i]), "y": vals[i], "hist": hist, "opp": float(opp.loc[idx[i]])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Simulación parametrizada (misma lógica que prediction_engine, con parámetros)
# --------------------------------------------------------------------------- #
def simulate(hist: np.ndarray, stat: str, p: SimParams, opp: float = 1.0, n: int = N_TUNE,
             seed: int | None = None) -> np.ndarray:
    return pe.simulate_stat_params(hist, stat, window=p.window, decay=p.decay, disp=p.disp,
                                   mean_mult=opp ** p.opp_beta if p.opp_beta else 1.0, n=n, seed=seed)


def crps(samples: np.ndarray, y: float) -> float:
    """CRPS por muestras: E|X − y| − ½ E|X − X'| (menor = mejor)."""
    x = np.sort(samples)
    n = len(x)
    term1 = np.mean(np.abs(x - y))
    term2 = np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * n)
    return float(term1 - term2)


def score(rows: pd.DataFrame, stat: str, p: SimParams) -> float:
    total = 0.0
    for i, r in enumerate(rows.itertuples(index=False)):
        total += crps(simulate(r.hist, stat, p, r.opp, seed=i), r.y)
    return total / max(len(rows), 1)


def tune_family(rows: pd.DataFrame, stat: str, passes: int = 2) -> tuple[SimParams, list]:
    """Búsqueda coordenada (un parámetro a la vez) sobre la rejilla acotada."""
    best = SimParams.current()
    best_score = score(rows, stat, best)
    trace = [{"params": asdict(best), "crps": round(best_score, 4), "paso": "actual"}]
    for _ in range(passes):
        improved = False
        for name, values in GRID.items():
            for v in values:
                cand = SimParams(**{**asdict(best), name: v})
                if cand == best:
                    continue
                s = score(rows, stat, cand)
                if s < best_score - 1e-4:
                    best, best_score, improved = cand, s, True
                    trace.append({"params": asdict(best), "crps": round(s, 4), "paso": name})
        if not improved:
            break
    return best, trace


# --------------------------------------------------------------------------- #
# Validación y prueba final con líneas reales (backtest/legs.parquet)
# --------------------------------------------------------------------------- #
def line_probs(legs: pd.DataFrame, w: pd.DataFrame, params: dict[str, SimParams]) -> pd.Series:
    """P(Over línea) del Monte Carlo con `params`, recalculada punto-en-el-tiempo por selección."""
    hist_by_player = {k: g for k, g in w.groupby("name_key")}
    opp_cache = {stat: opponent_factors(w, stat) for stat in params}
    out = pd.Series(np.nan, index=legs.index)
    for i, r in legs.iterrows():
        g = hist_by_player.get(normalize_name(r["player"]))
        p = params.get(r["family"])
        if g is None or p is None:
            continue
        order = r["season"] * 100 + r["week"]
        prior = g[g["order"] < order]
        if len(prior) == 0:
            continue
        cur = g[g["order"] == order]
        opp = float(opp_cache[r["family"]].loc[cur.index[0]]) if len(cur) else 1.0
        hist = pd.to_numeric(prior[r["family"]], errors="coerce").fillna(0).to_numpy(float)[-17:]
        sims = simulate(hist, r["family"], p, opp, n=4000, seed=int(i))
        out.loc[i] = float((sims > r["line"]).mean())
    return out


def evaluate_lines(legs: pd.DataFrame, w: pd.DataFrame, params_new: dict[str, SimParams]) -> dict:
    import backtest_eval as be
    c = be.canonical(legs[legs["kind"] == "player"])
    c = c[c["family"].isin(params_new)].copy()
    c["mc_old"] = c["mc"]
    c["mc_new"] = line_probs(c, w, params_new)
    c = c.dropna(subset=["mc_new"])
    out = {"por_familia": [], "total": {}}

    def ll(y, p):
        return be._scores(y, p)["log_loss"]

    # Calibración ajustada SOLO en validación, medida en la prueba final
    valid, hold = c[c["week"].isin(VALID_WEEKS)], c[c["week"].isin(HOLDOUT_WEEKS)]
    res = {}
    for tag, col in (("actual", "mc_old"), ("nuevo", "mc_new")):
        v, h = valid.assign(mc=valid[col]), hold.assign(mc=hold[col])
        fams = be.fit_all(v)
        ph = pd.Series(np.nan, index=h.index)
        for fam, d in h.groupby("family"):
            if fam in fams:
                lk, lm, frac = be._features(d)
                cal = fams[fam]
                ph.loc[d.index] = be._predict((cal["a"], cal["b"], cal["c"]), lk, lm, frac)
        res[tag] = {"mc_valid": ll(valid["y"].to_numpy(float), valid[col].to_numpy(float)),
                    "mc_prueba": ll(hold["y"].to_numpy(float), hold[col].to_numpy(float)),
                    "calibrada_prueba": ll(hold["y"].to_numpy(float), ph.fillna(hold["mkt"]).to_numpy(float)),
                    "peso_modelo": {f: v["model_weight"] for f, v in fams.items()}}
    res["mercado_prueba"] = ll(hold["y"].to_numpy(float), hold["mkt"].to_numpy(float))
    for fam, d in c.groupby("family"):
        hv = d[d["week"].isin(HOLDOUT_WEEKS)]
        out["por_familia"].append({
            "familia": fam, "n_valid": int(d["week"].isin(VALID_WEEKS).sum()), "n_prueba": int(len(hv)),
            "mc_actual_prueba": ll(hv["y"].to_numpy(float), hv["mc_old"].to_numpy(float)) if len(hv) else None,
            "mc_nuevo_prueba": ll(hv["y"].to_numpy(float), hv["mc_new"].to_numpy(float)) if len(hv) else None,
            "mercado_prueba": ll(hv["y"].to_numpy(float), hv["mkt"].to_numpy(float)) if len(hv) else None})
    out["total"] = res
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("streamlit", "urllib3", "data_pipeline"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    w = load_weekly()
    tuned, report = {}, {"entrenamiento": {}}
    for stat in FAMILIES:
        rows = build_rows(w, stat, TRAIN_SEASON)
        best, trace = tune_family(rows, stat)
        tuned[stat] = best
        report["entrenamiento"][stat] = {"n": int(len(rows)), "crps_actual": trace[0]["crps"],
                                         "crps_nuevo": trace[-1]["crps"], "params": asdict(best),
                                         "pasos": trace}
        log.info("%s: %s filas · CRPS %.3f -> %.3f · %s", stat, len(rows), trace[0]["crps"],
                 trace[-1]["crps"], asdict(best))

    legs = pd.read_parquet(ROOT / "backtest" / "legs.parquet")
    split_weeks(legs)
    ev = evaluate_lines(legs, w, tuned)
    report["validacion_y_prueba"] = ev
    t = ev["total"]
    adopt = t["nuevo"]["calibrada_prueba"] <= t["actual"]["calibrada_prueba"]
    report["adoptado"] = bool(adopt)
    report["regla"] = ("Se adopta el modelo nuevo solo si la probabilidad calibrada (ajustada en semanas "
                       "3–8) predice mejor las semanas 9–11, que no se usaron para nada más.")
    TUNING_REPORT.parent.mkdir(parents=True, exist_ok=True)
    TUNING_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if adopt:
        PARAMS_FILE.write_text(json.dumps({
            "fitted_on": f"temporada {TRAIN_SEASON} (CRPS) · validado {TEST_SEASON} semanas "
                         f"{min(VALID_WEEKS)}–{max(HOLDOUT_WEEKS)}",
            "families": {s: asdict(p) for s, p in tuned.items()}}, indent=2), encoding="utf-8")
    print(json.dumps({"adoptado": adopt, "prueba_final": t,
                      "params": {s: asdict(p) for s, p in tuned.items()}}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
