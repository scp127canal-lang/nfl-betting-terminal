"""Motor de predicción: Monte Carlo (10,000 iteraciones), share de WR1 y Realism Guard.

Todas las cuotas son decimales puras (float). Las probabilidades se devuelven en %.
"""

from __future__ import annotations

from typing import Iterable

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import TEAM_ABBR, get_player_game_log, team_td_log
from odds_pipeline import MARKET_STAT_MAP

N_SIMULATIONS = 10_000
REALISM_TOLERANCE = 0.15          # la línea no puede superar el máximo reciente en >15%
REALISM_GAMES = 10
WR1_SHARE_RANGE = (0.28, 0.35)    # target share / air yards share de un WR1 titular
RECENCY_DECAY = 0.85              # peso de cada partido respecto al siguiente más reciente
MIN_CV = 0.25                     # dispersión mínima para no sobreconfiar con pocos datos
COUNT_STATS = {"receptions", "passing_tds", "passing_interceptions", "attempts", "completions",
               "carries", "total_tds"}
TD_STATS = {"passing_tds", "total_tds", "passing_interceptions"}
TD_MIN_RATE = 0.05                # λ mínimo para TDs/INT (nunca probabilidad exacta 0)
LEAGUE_TEAM_TDS = 2.4             # TDs ofensivos medios por equipo y partido (referencia)
EVENT_LINE_MAX = 0.5              # "Over 0.5" = evento (anota / lanza INT): sin Realism Guard


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _extract_values(player_stats, stat_col: str | None, n: int) -> np.ndarray:
    """Acepta DataFrame (game log), Series, lista o array. Devuelve los últimos `n`
    valores en orden cronológico."""
    if isinstance(player_stats, pd.DataFrame):
        if stat_col is None:
            raise ValueError("stat_col es obligatorio cuando player_stats es un DataFrame")
        df = player_stats
        if {"season", "week"} <= set(df.columns):
            df = df.sort_values(["season", "week"])
        values = df[stat_col]
    else:
        values = pd.Series(list(player_stats) if player_stats is not None else [], dtype=float)
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return values[-n:]


def _weighted_moments(values: np.ndarray) -> tuple[float, float]:
    """Media y varianza con más peso en los partidos recientes."""
    w = RECENCY_DECAY ** np.arange(len(values))[::-1]
    mean = float(np.average(values, weights=w))
    var = float(np.average((values - mean) ** 2, weights=w))
    return mean, max(var, (MIN_CV * mean) ** 2)


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #
def simulate_yards(game_values: Iterable[float], n: int = N_SIMULATIONS,
                   seed: int | None = None) -> np.ndarray:
    """Yardas simuladas con una Gamma ajustada por momentos (no negativa y con
    cola derecha, como las yardas reales)."""
    values = _extract_values(game_values, None, REALISM_GAMES)
    if len(values) == 0:
        raise ValueError("Sin historial para simular")
    rng = _rng(seed)
    # Cero-inflación: una Gamma nunca da 0, pero un TE sin recepciones o un QB que
    # no corre sí. Se modela P(0) con la frecuencia (ponderada) de juegos <= 0.
    zero = values <= 0
    if zero.all():
        return np.zeros(n)
    w = RECENCY_DECAY ** np.arange(len(values))[::-1]
    p_zero = float(w[zero].sum() / w.sum())
    mean, var = _weighted_moments(values[~zero])
    sims = rng.gamma(shape=mean ** 2 / var, scale=var / mean, size=n)
    if p_zero > 0:
        sims[rng.random(n) < p_zero] = 0.0
    return sims


def simulate_count(game_values: Iterable[float], n: int = N_SIMULATIONS,
                   seed: int | None = None, min_rate: float = 0.0) -> np.ndarray:
    """Conteos discretos (TDs, INT, recepciones, intentos, acarreos).

    Binomial negativa si hay sobredispersión (var > media), Poisson si no.
    `min_rate` evita P = 0 exacto en TDs de jugadores sin anotaciones en L10.
    Ej.: P(2+ TD de pase) = media(sims >= 2) = media(sims > 1.5).
    """
    values = _extract_values(game_values, None, REALISM_GAMES)
    if len(values) == 0:
        raise ValueError("Sin historial para simular")
    mean, var = _weighted_moments(values)
    rng = _rng(seed)
    mean = max(mean, min_rate)
    if mean <= 0:
        return np.zeros(n)
    if var > mean:
        r = mean ** 2 / (var - mean)
        return rng.negative_binomial(r, r / (r + mean), size=n).astype(float)
    return rng.poisson(mean, size=n).astype(float)


simulate_receptions = simulate_count  # compatibilidad


def simulate_stat(game_values, stat_col: str, n: int = N_SIMULATIONS,
                  seed: int | None = None) -> np.ndarray:
    """Despacha al modelo adecuado: Gamma cero-inflada (yardas) o Poisson/NB (conteos)."""
    if stat_col == "first_td":
        raise ValueError("Primer TD necesita el contexto del partido: usa simulate_first_td")
    if stat_col in COUNT_STATS:
        return simulate_count(game_values, n=n, seed=seed,
                              min_rate=TD_MIN_RATE if stat_col in TD_STATS else 0.0)
    return simulate_yards(game_values, n=n, seed=seed)


def simulate_first_td(player_td_values: Iterable[float], team_td_values: Iterable[float],
                      opp_td_values: Iterable[float], n: int = N_SIMULATIONS,
                      seed: int | None = None) -> np.ndarray:
    """Primer anotador de TD del partido (1 = el jugador anota el primer TD).

    TDs del partido N ~ Poisson(λ_equipo + λ_rival). Si N >= 1, el primer TD es
    del jugador con probabilidad λ_jugador / λ_partido (cada TD se reparte en
    proporción a la tasa de anotación de cada jugador).
    """
    rng = _rng(seed)
    p_vals = _extract_values(player_td_values, None, REALISM_GAMES)
    lam_p = max(_weighted_moments(p_vals)[0] if len(p_vals) else 0.0, TD_MIN_RATE)
    team = _extract_values(team_td_values, None, REALISM_GAMES)
    opp = _extract_values(opp_td_values, None, REALISM_GAMES)
    lam_game = (team.mean() if len(team) else LEAGUE_TEAM_TDS) + (opp.mean() if len(opp) else LEAGUE_TEAM_TDS)
    lam_game = max(lam_game, lam_p + 0.5)
    tds = rng.poisson(lam_game, size=n)
    first = (tds >= 1) & (rng.random(n) < lam_p / lam_game)
    return first.astype(float)


def simulate_game_points(home_pf: Iterable[float], home_pa: Iterable[float],
                         away_pf: Iterable[float], away_pa: Iterable[float],
                         home_field: float = 1.5, n: int = N_SIMULATIONS,
                         seed: int | None = None) -> dict:
    """Puntos de un partido. Media de cada equipo = promedio entre lo que anota
    y lo que permite el rival; ventaja de local sumada al local."""
    rng = _rng(seed)

    def side(pf, opp_pa, bonus):
        pf, opp_pa = np.asarray(list(pf), float), np.asarray(list(opp_pa), float)
        mean = (pf.mean() + opp_pa.mean()) / 2 + bonus
        sd = max(np.concatenate([pf, opp_pa]).std(ddof=1), 7.0)
        return np.clip(np.round(rng.normal(mean, sd, n)), 0, None)

    home = side(home_pf, away_pa, home_field / 2)
    away = side(away_pf, home_pa, -home_field / 2)
    total, margin = home + away, home - away
    return {
        "home": home, "away": away, "total": total, "margin": margin,
        "home_win_pct": round(float((margin > 0).mean() + 0.5 * (margin == 0).mean()) * 100, 2),
        "proj_total": round(float(total.mean()), 1),
        "proj_margin": round(float(margin.mean()), 1),
    }


# --------------------------------------------------------------------------- #
# WR1: Target share / Air yards share
# --------------------------------------------------------------------------- #
def clamp_wr1_share(share: float, is_starter: bool = True) -> float:
    """Para un WR1 titular, el share se acota al rango realista 28%-35%."""
    if share is None or np.isnan(share):
        share = WR1_SHARE_RANGE[0] if is_starter else 0.0
    return float(np.clip(share, *WR1_SHARE_RANGE)) if is_starter else float(share)


def simulate_wr1_volume(team_pass_attempts: Iterable[float], target_share: float,
                        yards_per_target: float, team_air_yards: Iterable[float] | None = None,
                        air_yards_share: float | None = None, is_starter: bool = True,
                        ypt_sd: float = 9.5, n: int = N_SIMULATIONS,
                        seed: int | None = None) -> dict:
    """Yardas de recepción por volumen: intentos del equipo -> targets
    (Binomial con target share acotado) -> yardas (suma Gamma por target)."""
    rng = _rng(seed)
    t_share = clamp_wr1_share(target_share, is_starter)

    att = np.asarray(list(team_pass_attempts), float)
    attempts = np.clip(np.round(rng.normal(att.mean(), max(att.std(ddof=1), 4.0), n)), 15, None)
    targets = rng.binomial(attempts.astype(int), t_share)

    k = yards_per_target ** 2 / ypt_sd ** 2       # Gamma por target
    theta = ypt_sd ** 2 / yards_per_target
    yards = np.where(targets > 0, rng.gamma(np.maximum(targets * k, 1e-9), theta), 0.0)

    out = {"target_share": t_share, "targets": targets, "receiving_yards": yards,
           "proj_targets": round(float(targets.mean()), 1),
           "proj_receiving_yards": round(float(yards.mean()), 1)}

    if team_air_yards is not None and air_yards_share is not None:
        a_share = clamp_wr1_share(air_yards_share, is_starter)
        air = np.asarray(list(team_air_yards), float)
        team_air = np.clip(rng.normal(air.mean(), max(air.std(ddof=1), 25.0), n), 0, None)
        out.update({"air_yards_share": a_share, "air_yards": team_air * a_share,
                    "proj_air_yards": round(float(team_air.mean() * a_share), 1)})
    return out


# --------------------------------------------------------------------------- #
# Realism Guard
# --------------------------------------------------------------------------- #
def validate_prop_realism(player_stats, prop_line: float, stat_col: str | None = None,
                          n_games: int = REALISM_GAMES,
                          tolerance: float = REALISM_TOLERANCE) -> dict:
    """Invalida líneas que superan en más de `tolerance` (15%) el máximo real
    del jugador en sus últimos `n_games` partidos.

    Ej.: si el máximo de Chase Brown en 10 juegos es 80 yardas, el tope es 92;
    una línea de 100+ se rechaza.
    """
    values = _extract_values(player_stats, stat_col, n_games)
    if len(values) == 0:
        return {"is_valid": False, "prop_line": prop_line, "max_last_n": None,
                "threshold": None, "games": 0, "reason": "Sin historial reciente del jugador"}

    max_val = float(values.max())
    threshold = round(max_val * (1 + tolerance), 2)
    if prop_line <= EVENT_LINE_MAX:
        # "Anota un TD", "lanza una INT", "1+ recepción": es un evento, no una línea inflable
        return {"is_valid": True, "prop_line": prop_line, "max_last_n": max_val,
                "threshold": max(threshold, EVENT_LINE_MAX), "games": int(len(values)),
                "reason": "OK (línea de evento 0.5)"}
    is_valid = bool(prop_line <= threshold)
    reason = ("OK" if is_valid else
              f"Línea {prop_line} supera en más de {tolerance:.0%} el máximo "
              f"de {max_val:g} en los últimos {len(values)} juegos (tope {threshold:g})")
    return {"is_valid": is_valid, "prop_line": prop_line, "max_last_n": max_val,
            "threshold": threshold, "games": int(len(values)), "reason": reason}


# --------------------------------------------------------------------------- #
# Probabilidades y EV
# --------------------------------------------------------------------------- #
def implied_probability(decimal_odds: float) -> float:
    """Probabilidad implícita en % (1 / 1.57 = 63.69)."""
    if decimal_odds is None or decimal_odds <= 1:
        raise ValueError(f"Cuota decimal inválida: {decimal_odds}")
    return round(100 / decimal_odds, 2)


def probability_from_sims(sims: np.ndarray, line: float, side: str = "over") -> float:
    side = side.lower()
    if side not in {"over", "under"}:
        raise ValueError("side debe ser 'over' o 'under'")
    hit = sims > line if side == "over" else sims < line
    return float(hit.mean())


def mode_window(mean: float) -> float:
    """Ancho del rango de densidad para stats continuas (yardas): ~10 yds en volumen alto."""
    return 20.0 if mean >= 150 else 10.0 if mean >= 60 else 5.0 if mean >= 15 else 2.0


def most_probable_line(sims: np.ndarray, stat_col: str | None = None,
                       window: float | None = None) -> dict:
    """Línea Más Probable: moda de la distribución Monte Carlo (10k sims).

    - Conteos (TDs, recepciones, INT, intentos...): valor entero más repetido y
      su frecuencia exacta.
    - Yardas (continuas): ventana de ancho fijo con mayor densidad de
      simulaciones (rango de mayor densidad); la moda es la mediana dentro de ese
      rango y la concentración es el % de sims que caen en él.
    """
    s = np.asarray(sims, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        return {"mode_val": None, "concentration_pct": 0.0, "range_low": None, "range_high": None,
                "window": None, "discrete": True}
    discrete = stat_col in COUNT_STATS or bool(np.all(np.mod(s, 1) == 0))
    if discrete:
        vals = np.clip(np.round(s), 0, None).astype(int)
        counts = np.bincount(vals)
        mode = int(counts.argmax())
        return {"mode_val": float(mode), "concentration_pct": round(counts[mode] / n * 100, 1),
                "range_low": float(mode), "range_high": float(mode), "window": 1.0, "discrete": True}
    w = window or mode_window(float(s.mean()))
    srt = np.sort(s)
    ends = np.searchsorted(srt, srt + w, side="right")
    counts = ends - np.arange(n)
    i = int(counts.argmax())
    inside = srt[i:ends[i]]
    return {"mode_val": round(float(np.median(inside)), 1),
            "concentration_pct": round(counts[i] / n * 100, 1),
            "range_low": round(float(srt[i]), 1), "range_high": round(float(srt[i] + w), 1),
            "window": w, "discrete": False}


def expected_value_pct(real_prob: float, decimal_odds: float) -> float:
    """% +EV = (p_real x cuota - 1) x 100, con p_real en 0-1."""
    return round((real_prob * decimal_odds - 1) * 100, 2)


# --------------------------------------------------------------------------- #
# Probabilidad AJUSTADA: Monte Carlo mezclado con el mercado (sin vig)
#
# El Monte Carlo solo ve 10 partidos del jugador; la línea de la casa incorpora
# lesiones, rival, clima y el dinero de apostadores profesionales. Mezclar ambas
# frena las "ventajas" falsas (líneas extremas, poco historial) y deja +EV solo
# donde el modelo discrepa del mercado de forma clara.
# --------------------------------------------------------------------------- #
MODEL_WEIGHT = 0.50          # props de jugador con 10 partidos de historial
MODEL_WEIGHT_TEAM = 0.30     # moneyline / spread / total: mercado muy eficiente
MODEL_WEIGHT_FIRST_TD = 0.30  # Primer TD: modelo más simple
ONE_SIDED_VIG = 0.07         # margen estimado cuando la casa solo ofrece un lado (alternas, Sí/No)
MIN_EDGE_PCT = 3.0           # +EV solo si EV ajustado >= 3% (colchón ante error del modelo)
MAX_CREDIBLE_EV = 25.0       # EV mayor = "demasiado bueno": el mercado sabe algo (lesión, rol) -> revisar
DISAGREE_SCALE = 1.0         # cuanto más discrepa el modelo del mercado (log-odds), menos peso tiene
PROB_FLOOR, PROB_CEIL = 0.01, 0.97
_DEFAULTS = {"MIN_EDGE_PCT": MIN_EDGE_PCT, "MAX_CREDIBLE_EV": MAX_CREDIBLE_EV}


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    return float(np.log(p / (1 - p)))


def no_vig_prob(decimal_odds: float | None, other_odds: float | None = None) -> float | None:
    """Probabilidad del mercado sin margen de la casa.

    Con los dos lados (Over/Under, local/visita): p = (1/a) / (1/a + 1/b).
    Con un solo lado: p = (1/a) / (1 + margen estimado).
    """
    if decimal_odds is None or pd.isna(decimal_odds) or decimal_odds <= 1:
        return None
    a = 1 / float(decimal_odds)
    if other_odds is not None and not pd.isna(other_odds) and other_odds > 1:
        return a / (a + 1 / float(other_odds))
    return a / (1 + ONE_SIDED_VIG)


def model_weight(n_games: int | None = None, base: float = MODEL_WEIGHT) -> float:
    """Peso del modelo: completo con 10 partidos, proporcionalmente menor con menos."""
    if n_games is None:
        return base
    return base * min(1.0, max(n_games, 0) / REALISM_GAMES)


def effective_weight(model_prob: float, market_prob: float | None, weight: float = MODEL_WEIGHT) -> float:
    """Peso real del modelo tras penalizar la discrepancia con el mercado."""
    if market_prob is None:
        return 1.0
    delta = _logit(model_prob) - _logit(market_prob)
    return weight / (1 + (delta / DISAGREE_SCALE) ** 2)


def blend_prob(model_prob: float, market_prob: float | None, weight: float = MODEL_WEIGHT) -> float:
    """p_ajustada = w' · p_modelo + (1 − w') · p_mercado (acotada a [1%, 97%]).

    w' = w / (1 + (Δlogit / DISAGREE_SCALE)²): si el modelo y el mercado discrepan
    mucho (ej. 40% vs 5%), casi siempre es información que el modelo no tiene, así
    que el peso del modelo se reduce fuertemente.
    """
    if market_prob is None:
        return float(np.clip(model_prob, PROB_FLOOR, PROB_CEIL))
    w = effective_weight(model_prob, market_prob, weight)
    p = w * model_prob + (1 - w) * market_prob
    return float(np.clip(p, PROB_FLOOR, PROB_CEIL))


# --------------------------------------------------------------------------- #
# Calibración con datos reales (backtest con cuotas históricas)
#
# `model_calibration.json` (lo escribe `python backtest.py fit`) trae, por
# familia de mercado, una regresión logística ajustada con resultados reales:
#     logit(p) = a · logit(p_mercado) + b · logit(p_MonteCarlo) + c
# b/(a+b) es el peso real del modelo; c corrige su sesgo (ej. sobrestima Overs).
# Se ajusta sobre el lado "Over"/"Sí" y el Under es el complemento, así ambos
# lados siempre suman 100%. Sin calibración para una familia se usa la mezcla fija.
# --------------------------------------------------------------------------- #
CALIBRATION_FILE = Path(__file__).parent / "model_calibration.json"
CALIBRATION: dict = {}
TEAM_FAMILIES = ("h2h", "spreads", "totals")
_TUNABLE = {"min_edge_pct": "MIN_EDGE_PCT", "max_credible_ev": "MAX_CREDIBLE_EV"}


def set_calibration(cal: dict | None) -> dict:
    """Activa una calibración (o ninguna) y aplica sus umbrales globales."""
    global CALIBRATION
    CALIBRATION = dict(cal or {})
    g = globals()
    for key, name in _TUNABLE.items():
        g[name] = float(CALIBRATION.get(key, _DEFAULTS[name]))
    return CALIBRATION


def load_calibration(path: Path | None = None) -> dict:
    try:
        cal = json.loads(Path(path or CALIBRATION_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cal = {}
    return set_calibration(cal)


def family_of(kind: str, market: str | None, stat: str | None) -> str | None:
    """Familia de calibración: columna de stat para jugadores (ej. receiving_yards), mercado para equipos."""
    return market if kind == "team" else stat


def adjusted_prob(model_prob: float, market_prob: float | None, family: str | None = None,
                  side: str = "over", n_games: int | None = None,
                  weight: float | None = None) -> tuple[float, float]:
    """Probabilidad AJUSTADA y peso efectivo del modelo.

    Con calibración de la familia: regresión logística ajustada con resultados
    reales. El peso del Monte Carlo se reduce si el jugador tiene < 10 partidos.
    Sin calibración: mezcla lineal fija con penalización por discrepancia.
    """
    cal = (CALIBRATION.get("families") or {}).get(family) if family else None
    if cal and market_prob is not None:
        flip = str(side).lower() == "under"
        m, k = (1 - model_prob, 1 - market_prob) if flip else (model_prob, market_prob)
        frac = 1.0 if n_games is None else min(1.0, max(n_games, 0) / REALISM_GAMES)
        b = float(cal["b"]) * frac
        a = float(cal["a"]) + float(cal["b"]) * (1 - frac)
        z = a * _logit(k) + b * _logit(m) + float(cal.get("c", 0.0))
        p = 1 / (1 + np.exp(-z))
        p = 1 - p if flip else p
        w = b / (a + b) if a + b > 0 else 0.0
        return float(np.clip(p, PROB_FLOOR, PROB_CEIL)), float(np.clip(w, 0, 1))
    w = MODEL_WEIGHT if weight is None else weight
    return blend_prob(model_prob, market_prob, w), effective_weight(model_prob, market_prob, w)


def family_validated(family: str | None) -> bool:
    """¿El backtest comprobó ventaja real en esta familia? Sin calibración cargada: sí (modo antiguo)."""
    validated = CALIBRATION.get("validated_families")
    if validated is None:
        return True
    return bool(family) and family in validated


def ev_status(ev_pct: float | None, family: str | None = None) -> str | None:
    """'plus' (+EV comprobado) · 'unvalidated' (EV alto en mercado sin ventaja comprobada) ·
    'suspicious' (EV irreal) · 'neg'."""
    if ev_pct is None or pd.isna(ev_pct):
        return None
    if ev_pct > MAX_CREDIBLE_EV:
        return "suspicious"
    if ev_pct >= MIN_EDGE_PCT:
        return "plus" if family_validated(family) else "unvalidated"
    return "neg"


def is_plus_ev(ev_pct: float | None, family: str | None = None) -> bool:
    """+EV real: EV entre el umbral y el tope creíble, en una familia validada por el backtest."""
    return ev_status(ev_pct, family) == "plus"


def prop_edge(model_prob: float, decimal_odds: float, market_prob: float | None = None,
              weight: float = MODEL_WEIGHT, family: str | None = None, side: str = "over",
              n_games: int | None = None) -> dict:
    """Ventaja de una selección con probabilidad AJUSTADA (modelo + mercado).

    Ej.: cuota 1.85 (sin vig ≈ 52%), Monte Carlo 60%, peso 0.5 -> ajustada 56%:
    EV% = 0.56 * 1.85 - 1 = +3.6%  ·  Fair Odds = 1 / 0.56 = 1.79
    `model_prob` y `market_prob` en 0-1; si no se da `market_prob` se estima de la cuota.
    """
    implied = implied_probability(decimal_odds)
    mkt = market_prob if market_prob is not None else no_vig_prob(decimal_odds)
    p, w = adjusted_prob(model_prob, mkt, family, side, n_games, weight)
    ev = expected_value_pct(p, decimal_odds)
    return {
        "decimal_odds": round(float(decimal_odds), 2),
        "implied_prob_pct": implied,
        "mc_prob_pct": round(model_prob * 100, 2),            # Monte Carlo puro
        "market_prob_pct": round(mkt * 100, 2) if mkt is not None else None,
        "model_prob_pct": round(p * 100, 2),                  # probabilidad ajustada (la que cuenta)
        "edge_pct": round(p * 100 - implied, 2),
        "ev_pct": ev,
        "fair_odds": round(1 / p, 2) if p > 0 else None,
        "is_plus_ev": is_plus_ev(ev, family),
        "status": ev_status(ev, family),
        "suspicious": bool(ev > MAX_CREDIBLE_EV),
        "weight": round(w, 2),
        "calibrated": bool(family and family in (CALIBRATION.get("families") or {})),
    }


def evaluate_prop_lines(sims: np.ndarray, lines: pd.DataFrame,
                        player_stats=None, stat_col: str | None = None,
                        weight: float | None = None) -> pd.DataFrame:
    """Evalúa TODAS las líneas (oficiales y alternas) de un prop con una sola simulación.

    `lines`: columnas `line`, `over`, `under` (cuotas decimales; NaN si la casa no la ofrece).
    Añade por lado: *_prob_pct (AJUSTADA modelo+mercado), *_mc_pct (Monte Carlo puro),
    *_ev_pct, *_fair, *_plus_ev, y `realism_ok`.
    Las líneas que no pasan el Realism Guard no reciben EV.
    """
    out = lines.copy()
    for col in ("over", "under"):
        if col not in out:
            out[col] = np.nan
    n_hist = len(_extract_values(player_stats, stat_col, REALISM_GAMES)) if player_stats is not None else None
    if weight is None:
        weight = model_weight(n_hist)
    rows = []
    for r in out.itertuples(index=False):
        guard_ok = (validate_prop_realism(player_stats, r.line, stat_col)["is_valid"]
                    if player_stats is not None else True)
        row = {"realism_ok": guard_ok}
        for side, other in (("over", "under"), ("under", "over")):
            mc = probability_from_sims(sims, r.line, side)
            odds, other_odds = getattr(r, side), getattr(r, other)
            mkt = no_vig_prob(odds, other_odds)
            p, _ = adjusted_prob(mc, mkt, stat_col, side, n_hist, weight)
            row[f"{side}_mc_pct"] = round(mc * 100, 2)
            row[f"{side}_prob_pct"] = round(p * 100, 2)
            row[f"{side}_fair"] = round(1 / p, 2) if p > 0 else None
            if guard_ok and odds is not None and not pd.isna(odds) and odds > 1:
                edge = prop_edge(mc, float(odds), mkt, weight, stat_col, side, n_hist)
                row[f"{side}_ev_pct"], row[f"{side}_plus_ev"] = edge["ev_pct"], edge["is_plus_ev"]
                row[f"{side}_status"] = edge["status"]
            else:
                row[f"{side}_ev_pct"], row[f"{side}_plus_ev"] = None, False
                row[f"{side}_status"] = None
        rows.append(row)
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def evaluate_manual_prop(line: float, decimal_odds: float, player_stats=None,
                         stat_col: str | None = None, side: str = "over",
                         sims: np.ndarray | None = None, n: int = N_SIMULATIONS,
                         seed: int | None = None, market_prob: float | None = None,
                         weight: float | None = None, family: str | None = None) -> dict:
    """Consulta manual: línea + cuota decimal pura -> prob. implícita, prob. Monte
    Carlo, prob. AJUSTADA (modelo + mercado sin vig) y % EV sobre la ajustada.
    Pasa siempre por el Realism Guard cuando hay historial; una línea rechazada
    no recibe probabilidad ni EV.

    `market_prob`: prob. sin vig del mercado para este lado (0-1). Si no se da,
    se estima de la propia cuota con un margen típico.
    """
    implied = implied_probability(decimal_odds)
    result = {"line": line, "decimal_odds": decimal_odds, "side": side,
              "implied_prob_pct": implied, "real_prob_pct": None, "mc_prob_pct": None,
              "market_prob_pct": None, "ev_pct": None,
              "fair_odds": None, "projection_mean": None, "projection_median": None,
              "realism": None, "verdict": None}

    if player_stats is not None:
        guard = validate_prop_realism(player_stats, line, stat_col)
        result["realism"] = guard
        if not guard["is_valid"]:
            result["verdict"] = "RECHAZADA"
            return result
        if sims is None:
            sims = simulate_stat(_extract_values(player_stats, stat_col, REALISM_GAMES),
                                 stat_col or "", n=n, seed=seed)

    if sims is None:
        raise ValueError("Se necesita player_stats o sims para estimar la probabilidad real")

    n_hist = (len(_extract_values(player_stats, stat_col, REALISM_GAMES))
              if player_stats is not None and stat_col is not None else None)
    if weight is None:
        weight = model_weight(n_hist)
    p = probability_from_sims(sims, line, side)
    edge = prop_edge(p, decimal_odds, market_prob, weight, family or stat_col, side, n_hist)
    ev = edge["ev_pct"]
    result.update({
        "real_prob_pct": edge["model_prob_pct"],     # ajustada
        "mc_prob_pct": edge["mc_prob_pct"],
        "market_prob_pct": edge["market_prob_pct"],
        "model_weight": edge["weight"],
        "ev_pct": ev,
        "edge_pct": edge["edge_pct"],
        "is_plus_ev": edge["is_plus_ev"],
        "fair_odds": edge["fair_odds"],
        "projection_mean": round(float(np.mean(sims)), 1),
        "projection_median": round(float(np.median(sims)), 1),
        "calibrated": edge["calibrated"],
        "status": edge["status"],
        "verdict": {"plus": "+EV", "suspicious": "⚠️ Revisar (EV irreal)",
                    "unvalidated": "EV sin ventaja comprobada"}.get(edge["status"],
                                                                      "EV marginal" if ev > 0 else "-EV"),
    })
    return result


def first_td_sims(weekly: pd.DataFrame, log: pd.DataFrame, home: str | None, away: str | None,
                  n: int = N_SIMULATIONS, seed: int | None = None) -> np.ndarray:
    """Simulación de Primer TD con las tasas reales de TD del equipo del jugador y del rival."""
    team = log["recent_team"].iloc[-1] if "recent_team" in log and not log.empty else None
    opp = away if team == home else home
    team_tds = team_td_log(weekly, team) if team else []
    opp_tds = team_td_log(weekly, opp) if opp else []
    return simulate_first_td(log["total_tds"], team_tds, opp_tds, n=n, seed=seed)


def market_probs(df: pd.DataFrame, keys: list[str], outcome_col: str = "outcome",
                 odds_col: str = "decimal_odds") -> pd.Series:
    """Probabilidad sin vig del mercado para cada fila.

    Por grupo (`keys`, ej. jugador + mercado + línea) toma la cuota mediana entre
    casas de cada outcome y quita el margen con los dos lados; si el grupo solo
    tiene un lado (líneas alternas, Anytime TD) estima el margen.
    """
    if df.empty:
        return pd.Series(dtype=float)
    med = df.groupby(keys + [outcome_col], dropna=False)[odds_col].median().rename("_med").reset_index()
    inv = (1 / med["_med"]).where(med["_med"] > 1)
    med["_inv"] = inv
    grp = med.groupby(keys, dropna=False)
    med["_sum"] = grp["_inv"].transform("sum")
    med["_n"] = grp["_inv"].transform("count")
    med["_p"] = np.where(med["_n"] >= 2, med["_inv"] / med["_sum"], med["_inv"] / (1 + ONE_SIDED_VIG))
    merged = df[keys + [outcome_col]].reset_index().merge(
        med[keys + [outcome_col, "_p"]], on=keys + [outcome_col], how="left").set_index("index")
    return merged["_p"].reindex(df.index)


def evaluate_props_board(props: pd.DataFrame, weekly: pd.DataFrame,
                         n: int = N_SIMULATIONS, seed: int | None = None) -> pd.DataFrame:
    """Evalúa todo el tablero de props de `odds_pipeline` contra el historial de
    `data_pipeline`. Simula una vez por jugador y mercado y reutiliza las
    simulaciones para todas sus líneas alternativas. El EV usa la probabilidad
    AJUSTADA (Monte Carlo + mercado sin vig)."""
    board = props[props["market"].isin(MARKET_STAT_MAP) & props["player"].notna()
                  & props["outcome"].isin(["Over", "Under"])].copy()
    if board.empty:
        return board
    board["market_prob"] = market_probs(board, ["player", "market", "line"])

    results = []
    for (player, market), rows in board.groupby(["player", "market"]):
        stat = MARKET_STAT_MAP[market]
        hist_stat = "total_tds" if stat == "first_td" else stat
        log = get_player_game_log(weekly, player, REALISM_GAMES)
        values = (_extract_values(log, hist_stat, REALISM_GAMES)
                  if not log.empty and hist_stat in log else np.array([]))
        sims = None
        if len(values):
            if stat == "first_td":
                home = TEAM_ABBR.get(rows["home_team"].iloc[0]) if "home_team" in rows else None
                away = TEAM_ABBR.get(rows["away_team"].iloc[0]) if "away_team" in rows else None
                sims = first_td_sims(weekly, log, home, away, n=n, seed=seed)
            else:
                sims = simulate_stat(values, stat, n=n, seed=seed)
        weight = model_weight(len(values), MODEL_WEIGHT_FIRST_TD if stat == "first_td" else MODEL_WEIGHT)

        for idx, row in rows.iterrows():
            if pd.isna(row["decimal_odds"]) or pd.isna(row["line"]):
                continue
            if sims is None:
                res = {"verdict": "SIN DATOS", "implied_prob_pct": implied_probability(row["decimal_odds"])}
            else:
                mkt = row["market_prob"] if not pd.isna(row["market_prob"]) else None
                res = evaluate_manual_prop(row["line"], row["decimal_odds"], values, hist_stat,
                                           side=row["outcome"].lower(), sims=sims, market_prob=mkt,
                                           weight=weight, family=stat)
                res["realism_reason"] = res.pop("realism")["reason"]
            res["_idx"] = idx
            results.append(res)

    keep = ["_idx", "implied_prob_pct", "real_prob_pct", "mc_prob_pct", "ev_pct", "fair_odds",
            "is_plus_ev", "projection_mean", "verdict", "realism_reason"]
    res_df = pd.DataFrame(results).reindex(columns=keep).set_index("_idx")
    return board.join(res_df).sort_values("ev_pct", ascending=False, na_position="last")


load_calibration()
