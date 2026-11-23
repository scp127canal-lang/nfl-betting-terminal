"""Generador de combinadas +EV agrupadas por nivel de riesgo.

Flujo:
    legs = build_player_legs(props, weekly) + build_team_legs(game_odds, schedules)
    tiers = build_parlays(legs, weekly=weekly)          # {"BAJO": [...], "MODERADO": [...], "ALTO": [...]}
    custom = evaluate_parlay([leg_a, leg_b, leg_c])      # combinada manual de 2+ patas

Probabilidad conjunta: cópula gaussiana. Cada pata conserva su distribución
Monte Carlo marginal (prediction_engine) y una matriz de correlaciones
(heurística NFL + historial real cuando existe) acopla las patas.

Cuotas siempre en decimal puro (float). Probabilidades en %.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
from scipy.stats import norm

from data_pipeline import TEAM_ABBR, get_player_game_log, normalize_name, team_points_log  # noqa: F401
from odds_pipeline import MARKET_STAT_MAP, is_main_market, main_lines
from prediction_engine import (
    N_SIMULATIONS, REALISM_GAMES, expected_value_pct, implied_probability,
    simulate_game_points, simulate_stat, validate_prop_realism,
)

# --------------------------------------------------------------------------- #
# Niveles de riesgo
# --------------------------------------------------------------------------- #
RISK_TIERS = {
    "BAJO": {"emoji": "🟢", "badge": "🛡️ RIESGO BAJO",
             "rule": "Probabilidad > 50% | Cuota 1.80 a 3.20"},
    "MODERADO": {"emoji": "🟡", "badge": "⚖️ RIESGO MODERADO",
                 "rule": "Probabilidad 30%-50% | Cuota 3.20 a 7.50"},
    "ALTO": {"emoji": "🔴", "badge": "🚀 ALTO RIESGO",
             "rule": "Probabilidad < 30% | Cuota > 7.50"},
}

# Coherencia de Alto Riesgo: la cuota alta sale de ACUMULAR patas sólidas,
# nunca de una línea individual inflada.
HIGH_RISK_MIN_LEGS = 3
HIGH_RISK_MIN_LEG_PROB = 0.40      # cada pata debe ser realista por sí sola
HIGH_RISK_MAX_LEG_ODDS = 3.00      # sin "lotería" en una sola selección
TREND_MIN_HIT_RATE = 0.50          # la pata se cumplió en >=50% de los últimos 10 juegos
POOL_MIN_LEG_PROB = 0.35

KELLY_FRACTION = 0.25
MAX_STAKE_PCT = 0.05               # tope de seguridad: 5% del bankroll por combinada

STAT_FAMILY = {
    "passing_yards": "pass", "passing_tds": "pass", "attempts": "pass", "completions": "pass",
    "passing_interceptions": "int",
    "rushing_yards": "rush", "carries": "rush", "rush_rec_yds": "rush",
    "receiving_yards": "rec", "receptions": "rec",
    "total_tds": "td",
}
STAT_LABEL = {
    "passing_yards": "yds pase", "passing_tds": "TD pase", "attempts": "intentos pase",
    "completions": "pases completos", "passing_interceptions": "INT lanzadas",
    "rushing_yards": "yds carrera", "carries": "acarreos", "rush_rec_yds": "yds carrera+rec",
    "receiving_yards": "yds recepción", "receptions": "recepciones", "total_tds": "TDs",
}
YES_NO_LABEL = {"player_anytime_td": "Anytime TD", "player_1st_td": "Primer TD"}

# --------------------------------------------------------------------------- #
# Correlaciones (sobre la variable subyacente, no sobre el acierto)
# --------------------------------------------------------------------------- #
_SAME_TEAM = {("pass", "rec"): 0.45, ("rec", "rec"): -0.10, ("pass", "rush"): -0.15,
              ("rec", "rush"): -0.05, ("rush", "rush"): -0.25, ("pass", "pass"): 0.0,
              ("pass", "td"): 0.30, ("rec", "td"): 0.10, ("rush", "td"): 0.10, ("td", "td"): 0.0,
              ("int", "pass"): 0.10, ("int", "rec"): 0.0}
_OPP_TEAM = {("pass", "rec"): 0.15, ("rec", "rec"): 0.10, ("pass", "pass"): 0.20,
             ("pass", "rush"): 0.0, ("rec", "rush"): 0.0, ("rush", "rush"): -0.05,
             ("td", "td"): 0.10, ("pass", "td"): 0.10}
# Misma persona, distinta estadística (yds + TDs, yds + recepciones...)
_SAME_PLAYER = {("rec", "rec"): 0.75, ("rec", "rush"): 0.10, ("pass", "rush"): 0.0,
                ("pass", "rec"): 0.0, ("pass", "pass"): 0.60, ("rush", "rush"): 0.70,
                ("rec", "td"): 0.40, ("rush", "td"): 0.40, ("pass", "td"): 0.0,
                ("int", "pass"): 0.20}
_WITH_TOTAL = {"pass": 0.35, "rec": 0.30, "rush": 0.15, "td": 0.30, "int": 0.0}
_WITH_OWN_SIDE = {"rush": 0.30, "pass": -0.05, "rec": -0.05, "td": 0.20, "int": -0.25}
EMPIRICAL_MIN_GAMES = 6
EMPIRICAL_SHRINK = 10              # partidos "virtuales" de la heurística al mezclar


def _pair(table: dict, f1: str, f2: str) -> float:
    return table.get(tuple(sorted((f1, f2))), 0.0)


# --------------------------------------------------------------------------- #
# Pata
# --------------------------------------------------------------------------- #
@dataclass(eq=False)
class Leg:
    label: str
    kind: str                      # "player" | "team"
    market: str                    # mercado de The-Odds-API
    side: str                      # "over" | "under" | abreviatura del equipo
    line: float | None
    decimal_odds: float
    sims: np.ndarray = field(repr=False)
    event_id: str
    player: str | None = None
    team: str | None = None        # equipo del jugador / equipo elegido
    stat: str | None = None
    history: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    realism: dict | None = field(default=None, repr=False)

    def hits(self, values: np.ndarray) -> np.ndarray:
        if self.market == "h2h":
            return values > 0
        if self.market == "spreads":
            return values + self.line > 0
        return values > self.line if self.side == "over" else values < self.line

    @cached_property
    def prob(self) -> float:
        return float(self.hits(self.sims).mean())

    @property
    def ev_pct(self) -> float:
        return expected_value_pct(self.prob, self.decimal_odds)

    @property
    def implied_prob_pct(self) -> float:
        return implied_probability(self.decimal_odds)

    @property
    def trend_hit_rate(self) -> float:
        return float(self.hits(self.history).mean()) if len(self.history) else float("nan")

    @property
    def trend_supported(self) -> bool:
        return bool(self.trend_hit_rate >= TREND_MIN_HIT_RATE)

    @property
    def family(self) -> str | None:
        return STAT_FAMILY.get(self.stat) if self.stat else None

    @property
    def unique_key(self) -> tuple:
        """Dos patas con la misma clave no pueden ir en la misma combinada
        (misma estadística del mismo jugador, o dos lados del mismo partido)."""
        if self.kind == "player":
            return ("player", normalize_name(self.player), self.stat)
        return ("total" if self.market == "totals" else "side", self.event_id)

    def to_dict(self) -> dict:
        return {"label": self.label, "market": self.market, "line": self.line,
                "decimal_odds": self.decimal_odds, "prob_pct": round(self.prob * 100, 2),
                "ev_pct": self.ev_pct,
                "trend_hit_rate_pct": None if math.isnan(self.trend_hit_rate)
                else round(self.trend_hit_rate * 100, 1)}


# --------------------------------------------------------------------------- #
# Construcción de patas
# --------------------------------------------------------------------------- #
def build_player_legs(props: pd.DataFrame, weekly: pd.DataFrame,
                      n: int = N_SIMULATIONS, seed: int | None = None,
                      main_only: bool = False) -> list[Leg]:
    """Patas de props de jugador. Toda línea que no pasa el Realism Guard
    se descarta aquí y nunca llega a una combinada.

    `main_only=True` (combinadas sugeridas): solo la línea principal/fija de cada
    mercado con cuota de consenso; ninguna escalera alterna.
    """
    if main_only:
        props = main_lines(props)
    rows = props[props["market"].isin(MARKET_STAT_MAP) & props["player"].notna()
                 & props["outcome"].isin(["Over", "Under"])
                 & props["line"].notna() & props["decimal_odds"].notna()]
    cache: dict[tuple, tuple] = {}
    legs = []
    for row in rows.itertuples(index=False):
        stat = MARKET_STAT_MAP[row.market]
        if stat not in STAT_FAMILY:      # Primer TD: depende de todo el partido, no se combina
            continue
        key = (normalize_name(row.player), stat)
        if key not in cache:
            log = get_player_game_log(weekly, row.player, REALISM_GAMES)
            values = (pd.to_numeric(log[stat], errors="coerce").dropna().to_numpy(float)
                      if not log.empty else np.array([]))
            team = log["recent_team"].iloc[-1] if "recent_team" in log and not log.empty else None
            sims = simulate_stat(values, stat, n=n, seed=seed) if len(values) else None
            cache[key] = (values, team, sims)
        values, team, sims = cache[key]
        if sims is None:
            continue
        guard = validate_prop_realism(values, row.line)
        if not guard["is_valid"]:
            continue
        side = row.outcome.lower()
        if row.market in YES_NO_LABEL:
            label = f"{row.player} {YES_NO_LABEL[row.market]} ({'Sí' if side == 'over' else 'No'})"
        else:
            label = f"{row.player} {row.outcome} {row.line:g} {STAT_LABEL[stat]}"
        legs.append(Leg(
            label=label,
            kind="player", market=row.market, side=side, line=float(row.line),
            decimal_odds=float(row.decimal_odds), sims=sims, event_id=row.event_id,
            player=row.player, team=team, stat=stat, history=values, realism=guard))
    return legs


def build_team_legs(game_odds: pd.DataFrame, schedules: pd.DataFrame,
                    n: int = N_SIMULATIONS, seed: int | None = None,
                    min_games: int = 3) -> list[Leg]:
    """Patas de moneyline, spread y total a partir de la simulación de puntos."""
    legs = []
    rows = game_odds[game_odds["market"].isin(["h2h", "spreads", "totals"])
                     & game_odds["decimal_odds"].notna()]
    for i, (event_id, g) in enumerate(rows.groupby("event_id")):
        home_name, away_name = g["home_team"].iloc[0], g["away_team"].iloc[0]
        home, away = TEAM_ABBR.get(home_name), TEAM_ABBR.get(away_name)
        if not home or not away:
            continue
        hl, al = team_points_log(schedules, home), team_points_log(schedules, away)
        if len(hl) < min_games or len(al) < min_games:
            continue
        sim = simulate_game_points(hl["points_for"], hl["points_against"],
                                   al["points_for"], al["points_against"], n=n,
                                   seed=None if seed is None else seed + i)
        margin_hist = {home: (hl["points_for"] - hl["points_against"]).to_numpy(float),
                       away: (al["points_for"] - al["points_against"]).to_numpy(float)}
        total_hist = np.concatenate([(hl["points_for"] + hl["points_against"]).to_numpy(float),
                                     (al["points_for"] + al["points_against"]).to_numpy(float)])

        for row in g.itertuples(index=False):
            if row.market == "totals":
                if pd.isna(row.line):
                    continue
                legs.append(Leg(
                    label=f"Total {away}@{home} {row.outcome} {row.line:g}", kind="team",
                    market="totals", side=row.outcome.lower(), line=float(row.line),
                    decimal_odds=float(row.decimal_odds), sims=sim["total"],
                    event_id=event_id, history=total_hist))
                continue
            team = TEAM_ABBR.get(row.outcome)
            if team not in (home, away):
                continue
            margin = sim["margin"] if team == home else -sim["margin"]
            if row.market == "spreads":
                if pd.isna(row.line):
                    continue
                label, line = f"{team} {row.line:+g}", float(row.line)
            else:
                label, line = f"{team} ML", None
            legs.append(Leg(
                label=label, kind="team", market=row.market, side=team, line=line,
                decimal_odds=float(row.decimal_odds), sims=margin, event_id=event_id,
                team=team, history=margin_hist[team]))
    return legs


# --------------------------------------------------------------------------- #
# Correlación y simulación conjunta
# --------------------------------------------------------------------------- #
def _empirical_corr(weekly: pd.DataFrame, a: Leg, b: Leg) -> tuple[float, int]:
    """Correlación histórica entre dos jugadores del mismo equipo en la misma semana."""
    cols = ["season", "week", "recent_team"]
    if weekly is None or weekly.empty or not set(cols) <= set(weekly.columns):
        return 0.0, 0
    wa = weekly.loc[weekly["name_key"] == normalize_name(a.player), cols + [a.stat]]
    wb = weekly.loc[weekly["name_key"] == normalize_name(b.player), cols + [b.stat]]
    m = wa.merge(wb, on=cols, suffixes=("_a", "_b")).tail(2 * REALISM_GAMES)
    if len(m) < EMPIRICAL_MIN_GAMES:
        return 0.0, len(m)
    xa = m.iloc[:, 3].astype(float)
    xb = m.iloc[:, 4].astype(float)
    if xa.std() == 0 or xb.std() == 0:
        return 0.0, 0
    return float(np.corrcoef(xa, xb)[0, 1]), len(m)


def pair_correlation(a: Leg, b: Leg, weekly: pd.DataFrame | None = None) -> float:
    if a.event_id != b.event_id:
        return 0.0
    if a.kind == "team" and b.kind == "team":
        if "totals" in (a.market, b.market):
            return 0.0
        return 0.90 if a.team == b.team else -0.90
    if a.kind == "team":
        a, b = b, a                                   # a = jugador
    if b.kind == "team":
        if b.market == "totals":
            return _WITH_TOTAL.get(a.family, 0.0)
        if a.team is None:
            return 0.0
        c = _WITH_OWN_SIDE.get(a.family, 0.0)
        return c if a.team == b.team else -c

    if normalize_name(a.player) == normalize_name(b.player):
        return _pair(_SAME_PLAYER, a.family, b.family)
    if a.team is None or b.team is None:
        return 0.0
    if a.team != b.team:
        return _pair(_OPP_TEAM, a.family, b.family)
    prior = _pair(_SAME_TEAM, a.family, b.family)
    emp, games = _empirical_corr(weekly, a, b) if weekly is not None else (0.0, 0)
    if games >= EMPIRICAL_MIN_GAMES:
        prior = (games * emp + EMPIRICAL_SHRINK * prior) / (games + EMPIRICAL_SHRINK)
    return float(np.clip(prior, -0.95, 0.95))


def correlation_matrix(legs: list[Leg], weekly: pd.DataFrame | None = None) -> np.ndarray:
    k = len(legs)
    c = np.eye(k)
    for i, j in itertools.combinations(range(k), 2):
        c[i, j] = c[j, i] = pair_correlation(legs[i], legs[j], weekly)
    return _nearest_psd(c)


def _nearest_psd(c: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    vals, vecs = np.linalg.eigh(c)
    if vals.min() >= eps:
        return c
    c = vecs @ np.diag(np.clip(vals, eps, None)) @ vecs.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


class JointSimulator:
    """Una sola simulación conjunta de todas las patas (cópula gaussiana).
    La probabilidad de cualquier combinada = media del AND de sus aciertos."""

    def __init__(self, legs: list[Leg], weekly: pd.DataFrame | None = None,
                 n: int = N_SIMULATIONS, seed: int | None = None):
        self.legs = legs
        self.corr = correlation_matrix(legs, weekly)
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, len(legs))) @ np.linalg.cholesky(self.corr).T
        u = norm.cdf(z)
        self.hits = np.empty((n, len(legs)), dtype=bool)
        for i, leg in enumerate(legs):
            ordered = np.sort(leg.sims)
            idx = np.minimum((u[:, i] * len(ordered)).astype(int), len(ordered) - 1)
            self.hits[:, i] = leg.hits(ordered[idx])
        self.marginals = self.hits.mean(axis=0)

    def joint(self, idxs) -> float:
        return float(self.hits[:, list(idxs)].all(axis=1).mean())


# --------------------------------------------------------------------------- #
# Clasificación, coherencia y stake
# --------------------------------------------------------------------------- #
def classify_risk(joint_probability: float, decimal_odds: float) -> str | None:
    """Clasificación estricta: deben cumplirse probabilidad Y cuota."""
    p = joint_probability
    if p > 0.50 and 1.80 <= decimal_odds <= 3.20:
        return "BAJO"
    if 0.30 <= p <= 0.50 and 3.20 < decimal_odds <= 7.50:
        return "MODERADO"
    if p < 0.30 and decimal_odds > 7.50:
        return "ALTO"
    return None


def check_high_risk_coherence(legs: list[Leg]) -> tuple[bool, list[str]]:
    """Alto riesgo = 3+ patas realistas y respaldadas por tendencia."""
    issues = []
    if len(legs) < HIGH_RISK_MIN_LEGS:
        issues.append(f"Necesita {HIGH_RISK_MIN_LEGS}+ selecciones (tiene {len(legs)})")
    for leg in legs:
        if leg.kind == "player" and not (leg.realism or {}).get("is_valid", False):
            issues.append(f"{leg.label}: línea fuera de rango histórico")
        if leg.prob < HIGH_RISK_MIN_LEG_PROB:
            issues.append(f"{leg.label}: prob. individual {leg.prob:.0%} < {HIGH_RISK_MIN_LEG_PROB:.0%}")
        if leg.decimal_odds > HIGH_RISK_MAX_LEG_ODDS:
            issues.append(f"{leg.label}: cuota {leg.decimal_odds} > {HIGH_RISK_MAX_LEG_ODDS} (línea inflada)")
        if not leg.trend_supported:
            issues.append(f"{leg.label}: sin respaldo de tendencia")
    return not issues, issues


def calculate_stake_suggestion(ev_percentage: float, joint_probability: float,
                               bankroll: float = 1000.0, kelly_fraction: float = KELLY_FRACTION,
                               max_stake_pct: float = MAX_STAKE_PCT) -> dict:
    """Kelly fraccional (25%).

    ev_percentage: % EV (ej. 12.5). joint_probability: 0-1 o en % (ej. 0.42 o 42).
    Cuota implícita d = (1 + EV) / p  ->  Kelly completo f* = EV / (d - 1).
    Se aplica 25% de f* con tope de `max_stake_pct` del bankroll.
    """
    p = joint_probability / 100 if joint_probability > 1 else joint_probability
    ev = ev_percentage / 100
    zero = {"stake": 0.0, "stake_per_1000": 0.0, "kelly_full_pct": 0.0,
            "kelly_applied_pct": 0.0, "decimal_odds": None}
    if not 0 < p < 1 or ev <= 0:
        return zero
    odds = (1 + ev) / p
    b = odds - 1
    if b <= 0:
        return zero
    full = ev / b
    applied = min(full * kelly_fraction, max_stake_pct)
    return {"stake": round(bankroll * applied, 2),
            "stake_per_1000": round(1000 * applied, 2),
            "kelly_full_pct": round(full * 100, 2),
            "kelly_applied_pct": round(applied * 100, 2),
            "decimal_odds": round(odds, 2)}


# --------------------------------------------------------------------------- #
# Same Game Parlay (SGP) y cuota ajustada por la casa
# --------------------------------------------------------------------------- #
SGP_ALERT = "🔥 SGP Correlacionado (Mismo Partido)"
SGP_NOTE = ("⚠️ En combinadas del mismo partido, Draftea suele ajustar la cuota multiplicada "
            "a un valor menor (ej. 2.45 ➔ 2.09).")


def detect_sgp(legs: list[Leg]) -> dict:
    """Detecta si 2+ selecciones pertenecen al mismo partido.

    Devuelve {"is_sgp": bool, "games": {event_id: [labels]}, "n_games": int}.
    """
    groups: dict[str, list[str]] = {}
    for leg in legs:
        groups.setdefault(leg.event_id, []).append(leg.label)
    sgp = {event: labels for event, labels in groups.items() if len(labels) >= 2}
    return {"is_sgp": bool(sgp), "games": sgp, "n_games": len(groups)}


def theoretical_odds(legs: list[Leg]) -> float:
    """Cuota multiplicada teórica (producto de cuotas decimales), ej. 1.58 * 1.55 = 2.45."""
    return round(float(np.prod([leg.decimal_odds for leg in legs])), 2)


def reprice_parlay(joint_probability: float, adjusted_odds: float, stake: float = 100.0,
                   bankroll: float = 1000.0, theoretical: float | None = None) -> dict:
    """Recalcula EV, payout y stake con la cuota REAL que paga la casa.

    La probabilidad conjunta Monte Carlo (ya correlacionada) no cambia: solo cambia
    lo que se cobra. Ej.: p = 52%, cuota ajustada 2.09, stake $100 ->
    cobro $209.00, neto +$109.00, EV = 0.52 * 2.09 - 1 = +8.68%.
    `joint_probability` en 0-1 o en %.
    """
    if adjusted_odds is None or adjusted_odds <= 1:
        raise ValueError(f"Cuota ajustada inválida: {adjusted_odds}")
    p = joint_probability / 100 if joint_probability > 1 else joint_probability
    ev = expected_value_pct(p, adjusted_odds)
    payout = stake * adjusted_odds
    out = {
        "adjusted_odds": round(float(adjusted_odds), 2),
        "theoretical_odds": theoretical,
        "house_haircut_pct": (round((1 - adjusted_odds / theoretical) * 100, 2)
                              if theoretical and theoretical > 1 else None),
        "joint_prob_pct": round(p * 100, 2),
        "implied_prob_pct": implied_probability(adjusted_odds),
        "ev_pct": ev,
        "is_plus_ev": ev > 0,
        "stake": round(stake, 2),
        "payout": round(payout, 2),
        "net_profit": round(payout - stake, 2),
        "ev_cash": round(stake * ev / 100, 2),
        "tier": classify_risk(p, adjusted_odds),
        "kelly": calculate_stake_suggestion(ev, p, bankroll),
    }
    # Odds mínimas para que la combinada siga siendo +EV
    out["break_even_odds"] = round(1 / p, 2) if p > 0 else None
    return out


# --------------------------------------------------------------------------- #
# Combinadas
# --------------------------------------------------------------------------- #
def _compatible(legs: list[Leg]) -> bool:
    keys = [leg.unique_key for leg in legs]
    return len(keys) == len(set(keys))


def _summarize(legs: list[Leg], joint_p: float, indep_p: float, bankroll: float) -> dict:
    odds = round(float(np.prod([leg.decimal_odds for leg in legs])), 2)
    ev = expected_value_pct(joint_p, odds)
    tier = classify_risk(joint_p, odds)
    coherent, issues = check_high_risk_coherence(legs) if tier == "ALTO" else (True, [])
    if not coherent:
        tier = None
    info = RISK_TIERS.get(tier, {})
    sgp = detect_sgp(legs)
    return {
        "sgp": sgp["is_sgp"],
        "sgp_games": sgp["games"],
        "n_legs": len(legs),
        "n_player_legs": sum(leg.kind == "player" for leg in legs),
        "n_team_legs": sum(leg.kind == "team" for leg in legs),
        "legs": [leg.to_dict() for leg in legs],
        "labels": [leg.label for leg in legs],
        "decimal_odds": odds,
        "joint_prob_pct": round(joint_p * 100, 2),
        "independent_prob_pct": round(indep_p * 100, 2),
        "correlation_lift": round(joint_p / indep_p, 3) if indep_p > 0 else None,
        "implied_prob_pct": implied_probability(odds),
        "ev_pct": ev,
        "tier": tier,
        "emoji": info.get("emoji"),
        "badge": info.get("badge"),
        "coherence_issues": issues,
        "stake": calculate_stake_suggestion(ev, joint_p, bankroll),
    }


def evaluate_parlay(legs: list[Leg], weekly: pd.DataFrame | None = None,
                    bankroll: float = 1000.0, n: int = N_SIMULATIONS,
                    seed: int | None = None) -> dict:
    """Evalúa una combinada armada por el usuario (2 o más patas)."""
    if len(legs) < 2:
        raise ValueError("Una combinada necesita al menos 2 patas")
    if not _compatible(legs):
        raise ValueError("Patas incompatibles: misma estadística/jugador o dos lados del mismo partido")
    sim = JointSimulator(legs, weekly, n, seed)
    idx = range(len(legs))
    return _summarize(legs, sim.joint(idx), float(np.prod(sim.marginals)), bankroll)


PARLAY_MODES = ("mixed", "players", "teams")
PLAYER_POOL_MIN_PROB = 0.40        # props de jugador: líneas realistas de la escalera
PLAYER_POOL_MAX_ODDS = 3.00        # nada de líneas alternas "lotería"
PLAYER_POOL_MIN_ODDS = 1.40        # ni líneas triviales (Over 0.5, Over 19.5 @ 1.06)


def select_pool(legs: list[Leg], mode: str = "mixed", pool_size: int = 30) -> list[Leg]:
    """Pool de candidatas por tipo de parlay.

    Props de jugador: SOLO líneas principales/fijas (nunca `_alternate`). De cada
    jugador + estadística se queda el lado (Over/Under) de mayor EV entre los
    realistas (prob >= 40%, cuota 1.40-3.00).
    En modo mixto se reservan ~2/3 del pool para jugadores.
    """
    if mode not in PARLAY_MODES:
        raise ValueError(f"mode debe ser uno de {PARLAY_MODES}")
    best: dict[tuple, Leg] = {}
    for leg in legs:
        if (leg.kind != "player" or not is_main_market(leg.market) or leg.prob < PLAYER_POOL_MIN_PROB
                or not PLAYER_POOL_MIN_ODDS <= leg.decimal_odds <= PLAYER_POOL_MAX_ODDS):
            continue
        if leg.unique_key not in best or leg.ev_pct > best[leg.unique_key].ev_pct:
            best[leg.unique_key] = leg
    players = sorted(best.values(), key=lambda l: l.ev_pct, reverse=True)
    teams = sorted((l for l in legs if l.kind == "team" and l.prob >= POOL_MIN_LEG_PROB),
                   key=lambda l: l.ev_pct, reverse=True)
    if mode == "players":
        return players[:pool_size]
    if mode == "teams":
        return teams[:pool_size]
    n_players = min(len(players), max(pool_size * 2 // 3, pool_size - len(teams)))
    return players[:n_players] + teams[:pool_size - n_players]


def build_parlays(legs: list[Leg], weekly: pd.DataFrame | None = None,
                  min_legs: int = 2, max_legs: int = 6, pool_size: int = 30,
                  beam_width: int = 300, min_ev: float = 0.0, top_n: int = 5,
                  bankroll: float = 1000.0, n: int = N_SIMULATIONS,
                  seed: int | None = None, mode: str = "mixed") -> dict[str, list[dict]]:
    """Genera combinadas +EV de 2 a `max_legs` patas, agrupadas por riesgo.

    mode: "mixed" (al menos 1 jugador y 1 equipo), "players" (solo props de
    jugador) o "teams" (solo spread / moneyline / total).

    Búsqueda en haz: todas las parejas del pool se evalúan; cada tamaño
    siguiente extiende las mejores `beam_width` combinadas (por EV y por
    probabilidad, para no sesgar hacia un solo nivel).
    """
    pool = select_pool(legs, mode, pool_size)
    out = {tier: [] for tier in RISK_TIERS}
    if len(pool) < min_legs:
        return out

    sim = JointSimulator(pool, weekly, n, seed)
    hits, marg, k = sim.hits, sim.marginals, len(pool)
    odds = np.array([leg.decimal_odds for leg in pool])
    keys = [leg.unique_key for leg in pool]
    is_player = np.array([leg.kind == "player" for leg in pool])

    frontier = {(i,): hits[:, i] for i in range(k)}
    seen, found = set(), []
    for size in range(2, max_legs + 1):
        candidates = {}
        for combo, vec in frontier.items():
            used = {keys[i] for i in combo}
            for j in range(k):
                if j in combo or keys[j] in used:
                    continue
                new = tuple(sorted(combo + (j,)))
                if new not in candidates and new not in seen:
                    candidates[new] = vec & hits[:, j]
        if not candidates:
            break
        seen.update(candidates)

        scored = []
        for combo, vec in candidates.items():
            p = float(vec.mean())
            combo_odds = float(np.prod(odds[list(combo)]))
            ev = p * combo_odds - 1
            scored.append((combo, p, ev))
            n_pl = int(is_player[list(combo)].sum())
            mixed_ok = mode != "mixed" or 0 < n_pl < len(combo)
            if (size >= min_legs and mixed_ok and ev * 100 > min_ev
                    and classify_risk(p, round(combo_odds, 2))):
                found.append((combo, p))

        half = max(beam_width // 2, 1)
        keep = {c for c, _, _ in sorted(scored, key=lambda s: s[2], reverse=True)[:half]}
        keep |= {c for c, _, _ in sorted(scored, key=lambda s: s[1], reverse=True)[:half]}
        frontier = {c: candidates[c] for c in keep}

    for combo, p in found:
        chosen = [pool[i] for i in combo]
        summary = _summarize(chosen, p, float(np.prod(marg[list(combo)])), bankroll)
        if summary["tier"] and summary["ev_pct"] > min_ev:
            out[summary["tier"]].append(summary)

    for tier in out:
        out[tier] = _diverse_top(sorted(out[tier], key=lambda s: s["ev_pct"], reverse=True), top_n)
    return out


def _diverse_top(parlays: list[dict], top_n: int, max_shared: float = 0.6) -> list[dict]:
    """Evita devolver 5 variantes casi idénticas de la misma combinada."""
    chosen = []
    for p in parlays:
        labels = set(p["labels"])
        if all(len(labels & set(c["labels"])) / len(labels) <= max_shared for c in chosen):
            chosen.append(p)
        if len(chosen) == top_n:
            break
    return chosen


def parlays_to_frame(tiers: dict[str, list[dict]]) -> pd.DataFrame:
    """Vista tabular (una fila por combinada) para Streamlit."""
    rows = [{"badge": p["badge"], "n_legs": p["n_legs"], "selecciones": " + ".join(p["labels"]),
             "decimal_odds": p["decimal_odds"], "joint_prob_pct": p["joint_prob_pct"],
             "ev_pct": p["ev_pct"], "stake_per_1000": p["stake"]["stake_per_1000"]}
            for tier in tiers.values() for p in tier]
    return pd.DataFrame(rows)
