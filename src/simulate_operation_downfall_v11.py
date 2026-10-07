import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

WEEKS = 416
NODES = 48
KYUSHU = np.arange(24)
HONSHU = np.arange(24, 48)
OUTCOMES = [
    "coordinated_surrender",
    "fragmented_surrender",
    "conditional_settlement",
    "allied_operational_pause",
    "humanitarian_collapse",
    "war_continues_416_weeks",
]


@dataclass(frozen=True)
class Structure:
    name: str
    network_weight: float
    queue_weight: float
    stochasticity: float
    isolation_weight: float
    smooth_weight: float
    game_weight: float = 0.0


STRUCTURES = [
    Structure("network_campaign", 1.0, 0.75, 1.0, 1.0, 0.20),
    Structure("aggregate_lanchester", 0.0, 0.35, 0.85, 0.35, 0.55),
    Structure("discrete_event_logistics", 0.55, 1.25, 1.10, 0.75, 0.25),
    Structure("system_dynamics", 0.45, 0.55, 0.45, 0.60, 1.0),
    Structure("differential_game", 0.75, 0.80, 0.75, 0.85, 0.45, 1.0),
]

SCENARIOS = {
    "baseline_refusal_at_xday": dict(no_surrender_until=4, total_war=0.75, blockade=1.0, nuclear=0.18),
    "central_refusal_26_weeks": dict(no_surrender_until=26, total_war=0.90, blockade=1.05, nuclear=0.24),
    "central_refusal_52_weeks": dict(no_surrender_until=52, total_war=1.08, blockade=1.12, nuclear=0.32),
    "hundred_million_total_war": dict(no_surrender_until=40, total_war=1.35, blockade=1.08, nuclear=0.28),
    "emperor_threshold_surrender": dict(no_surrender_until=8, total_war=1.35, blockade=1.12, nuclear=0.30),
    "blockade_delay_18_weeks": dict(no_surrender_until=18, total_war=0.88, blockade=1.30, nuclear=0.24),
    "central_refusal_260_weeks": dict(no_surrender_until=260, total_war=1.30, blockade=1.15, nuclear=0.30),
    "extermination_refusal_416_weeks": dict(no_surrender_until=416, total_war=1.55, blockade=1.22, nuclear=0.36),
}

CASES = {
    "iwo_jima": dict(days=36, japanese=21.0, japanese_dead=20.0, captured=0.216,
                       allied_kia=6.871, fort=0.96, supply=0.58, disease=0.10,
                       isolation=0.88, resolve=0.98, contact_density=35.0,
                       withdrawal_space=0.02, capture_opportunity=0.04, confidence=1.0),
    "okinawa": dict(days=91, japanese=117.4, japanese_dead=110.0, captured=7.4,
                     allied_kia=12.52, fort=0.86, supply=0.46, disease=0.22,
                     isolation=0.72, resolve=0.92, contact_density=10.0,
                     withdrawal_space=0.10, capture_opportunity=0.32, confidence=1.0),
    "luzon": dict(days=180, japanese=230.0, japanese_dead=192.0, captured=9.7,
                   allied_kia=7.933, fort=0.68, supply=0.24, disease=0.56,
                   isolation=0.78, resolve=0.88, contact_density=3.5,
                   withdrawal_space=0.48, capture_opportunity=0.24, confidence=0.85),
    "imphal": dict(days=120, japanese=155.0, japanese_dead=65.0, captured=1.5,
                    allied_kia=4.0, fort=0.42, supply=0.12, disease=0.88,
                    isolation=0.54, resolve=0.76, contact_density=1.2,
                    withdrawal_space=0.62, capture_opportunity=0.08, confidence=0.80),
    "burma_1944_45": dict(days=450, japanese=180.0, japanese_dead=97.0, captured=6.0,
                           allied_kia=14.0, fort=0.38, supply=0.20, disease=0.82,
                           isolation=0.48, resolve=0.78, contact_density=0.35,
                           withdrawal_space=0.78, capture_opportunity=0.14, confidence=0.65),
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def nuclear_stock_target(week):
    # DOE early-stockpile anchors, with a broad continued-war extrapolation.
    anchor_week = np.array([0, 52, 104, 156, 260, 416], dtype=float)
    anchor_stock = np.array([2, 9, 13, 50, 400, 800], dtype=float)
    return float(np.interp(week, anchor_week, anchor_stock))


def q(x):
    if len(x) == 0:
        return {k: None for k in ("p01", "p10", "median", "p90", "p99", "mean")}
    return {
        "p01": float(np.quantile(x, 0.01)),
        "p10": float(np.quantile(x, 0.10)),
        "median": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p99": float(np.quantile(x, 0.99)),
        "mean": float(np.mean(x)),
    }


def build_network():
    terrain = np.array(
        ["coastal"] * 6 + ["urban"] * 5 + ["mountain"] * 9 + ["logistics"] * 4
        + ["coastal"] * 5 + ["urban"] * 7 + ["mountain"] * 8 + ["logistics"] * 4)
    adj = np.zeros((NODES, NODES), dtype=float)
    for offset in (0, 24):
        for i in range(24):
            a, b = offset + i, offset + ((i + 1) % 24)
            adj[a, b] = adj[b, a] = 1.0
        for i in range(0, 24, 3):
            a, b = offset + i, offset + ((i + 7) % 24)
            adj[a, b] = adj[b, a] = 1.0
    for i in range(4):
        adj[20 + i, 44 + i] = adj[44 + i, 20 + i] = 0.35
    degree = np.maximum(adj.sum(axis=1), 1.0)
    return terrain, adj, degree


TERRAIN, ADJ, DEGREE = build_network()
TERRAIN_FORT = np.array([
    0.92 if x == "coastal" else 1.12 if x == "urban" else 1.22 if x == "mountain" else 0.82
    for x in TERRAIN
])
TERRAIN_ACCESS = np.array([
    1.15 if x == "coastal" else 0.84 if x == "urban" else 0.63 if x == "mountain" else 1.06
    for x in TERRAIN
])
CAPITAL_SYSTEM = np.array([
    i for i in HONSHU if TERRAIN[i] in ("urban", "logistics")], dtype=int)


def build_transport_cost():
    dist = np.full((NODES, NODES), 99.0)
    np.fill_diagonal(dist, 0.0)
    edge_i, edge_j = np.where(ADJ > 0)
    dist[edge_i, edge_j] = 1.0 / ADJ[edge_i, edge_j]
    for k in range(NODES):
        dist = np.minimum(dist, dist[:, k, None] + dist[None, k, :])
    sources = np.array([3, 18, 29, 43])
    cost = dist[sources]
    return cost / np.maximum(cost.max(), 1.0)


TRANSPORT_COST = build_transport_cost()
TRANSPORT_KERNEL = np.exp(-TRANSPORT_COST / 0.22)


def mean_field_distribution(previous, need, risk, congestion_capacity, transport_penalty):
    """Discrete logit MFG fixed point for decentralized auxiliary deployment."""
    mass = np.maximum(previous, 1e-9)
    mass /= mass.sum(axis=1, keepdims=True)
    residual = np.zeros(len(mass))
    for _ in range(5):
        congestion = mass / np.maximum(congestion_capacity, 1e-6)
        utility = 2.0 * need - 1.25 * risk - 0.80 * congestion - 0.75 * transport_penalty
        utility -= utility.max(axis=1, keepdims=True)
        response = np.exp(utility / 0.42)
        response /= np.maximum(response.sum(axis=1, keepdims=True), 1e-12)
        updated = 0.55 * mass + 0.45 * response
        residual = np.abs(updated - mass).sum(axis=1)
        mass = updated
    return mass, residual


def entropic_transport_share(source_mass, target_mass):
    """Capacity-free Sinkhorn core; capacity is applied to the resulting node flow."""
    a = source_mass / np.maximum(source_mass.sum(axis=1, keepdims=True), 1e-12)
    b = target_mass / np.maximum(target_mass.sum(axis=1, keepdims=True), 1e-12)
    u = np.ones_like(a)
    v = np.ones_like(b)
    kernel = TRANSPORT_KERNEL[None, :, :]
    for _ in range(6):
        u = a / np.maximum(np.einsum("ij,nj->ni", TRANSPORT_KERNEL, v), 1e-12)
        v = b / np.maximum(np.einsum("ni,ij->nj", u, TRANSPORT_KERNEL), 1e-12)
    plan = u[:, :, None] * kernel * v[:, None, :]
    node_share = plan.sum(axis=1)
    node_share /= np.maximum(node_share.sum(axis=1, keepdims=True), 1e-12)
    mean_cost = (plan * TRANSPORT_COST[None, :, :]).sum(axis=(1, 2))
    return node_share, mean_cost


def case_predict(theta, case):
    direct, isolation, disease, capture, fort_protection = theta.T
    duration = case["days"] / 30.0
    protected_exposure = (1.15 - fort_protection * case["fort"])
    contact_exposure = case["contact_density"] * duration ** 0.75
    attrition_exposure = np.sqrt(duration) * (1.0 - 0.70 * case["withdrawal_space"])
    combat_cumulative = (direct * protected_exposure
                         * (0.55 + 0.45 * case["resolve"]) * contact_exposure)
    isolation_cumulative = (isolation * case["isolation"] * (1.0 - case["supply"])
                            * attrition_exposure)
    disease_cumulative = (disease * case["disease"] * (1.0 - 0.55 * case["supply"])
                          * attrition_exposure)
    death_fraction = 1.0 - np.exp(-(combat_cumulative + isolation_cumulative
                                    + disease_cumulative))
    capture_fraction = (1.0 - death_fraction) * sigmoid(
        -5.0 + 3.0 * capture + 2.1 * (1.0 - case["resolve"])
        + 1.2 * case["isolation"] + 3.0 * case["capture_opportunity"])
    return case["japanese"] * death_fraction, case["japanese"] * capture_fraction


def historical_prior(seed=1945, samples=60000):
    rng = np.random.default_rng(seed)
    prior = np.column_stack([
        rng.uniform(0.055, 0.22, samples),
        rng.uniform(0.025, 0.18, samples),
        rng.uniform(0.015, 0.16, samples),
        rng.uniform(0.05, 0.95, samples),
        rng.uniform(0.25, 0.82, samples),
    ])
    errors = np.zeros(samples)
    for case in CASES.values():
        pred_dead, pred_captured = case_predict(prior, case)
        dead_error = np.log1p(pred_dead) - np.log1p(case["japanese_dead"])
        cap_error = np.log1p(pred_captured) - np.log1p(case["captured"])
        errors += case["confidence"] * (dead_error ** 2 + 0.30 * cap_error ** 2)
    keep = np.argsort(errors)[:4000]
    posterior = prior[keep]

    loo = []
    names = list(CASES)
    for held_name in names:
        loo_error = np.zeros(samples)
        for name, case in CASES.items():
            if name == held_name:
                continue
            pred_dead, pred_captured = case_predict(prior, case)
            loo_error += case["confidence"] * (
                (np.log1p(pred_dead) - np.log1p(case["japanese_dead"])) ** 2
                + 0.30 * (np.log1p(pred_captured) - np.log1p(case["captured"])) ** 2)
        loo_post = prior[np.argsort(loo_error)[:4000]]
        pred_dead, pred_cap = case_predict(loo_post, CASES[held_name])
        # Posterior-predictive intervals include between-campaign structural
        # discrepancy; parameter uncertainty alone is falsely precise.
        pred_dead *= rng.lognormal(-0.5 * 0.30 ** 2, 0.30, len(pred_dead))
        pred_cap *= rng.lognormal(-0.5 * 0.80 ** 2, 0.80, len(pred_cap))
        loo.append({
            "held_out": held_name,
            "observed_dead_thousands": CASES[held_name]["japanese_dead"],
            "predicted_dead_thousands": q(pred_dead),
            "observed_captured_thousands": CASES[held_name]["captured"],
            "predicted_captured_thousands": q(pred_cap),
            "dead_covered_by_p01_p99": bool(
                np.quantile(pred_dead, 0.01) <= CASES[held_name]["japanese_dead"]
                <= np.quantile(pred_dead, 0.99)),
            "captured_covered_by_p01_p99": bool(
                np.quantile(pred_cap, 0.01) <= CASES[held_name]["captured"]
                <= np.quantile(pred_cap, 0.99)),
        })
    return posterior, loo


def regime_shares(control, fort, supply, command, week, coronet):
    eligible = np.ones_like(control, dtype=bool)
    eligible[:, HONSHU] = coronet[:, None]
    landing = eligible & (control < 0.08) & (week < 10)
    beachhead = eligible & (control >= 0.08) & (control < 0.24)
    mobile = eligible & (control >= 0.24) & (control < 0.58)
    siege = eligible & (control >= 0.10) & (control < 0.88) & (
        ((TERRAIN == "urban") | (TERRAIN == "mountain"))[None, :]) & (fort > 0.35)
    collapse = eligible & ((supply < 0.14) | (command < 0.20)) & (control > 0.15)
    pursuit = eligible & (control >= 0.58) & (~siege)
    return landing, beachhead, mobile, siege, pursuit, collapse


def conditional_summary(mask, arrays):
    count = int(mask.sum())
    return {
        "paths": count,
        "share": float(mask.mean()),
        **{name: q(values[mask]) for name, values in arrays.items()},
    }


def simulate(structure, scenario_name, posterior, n=500, seed=19451101):
    cfg = SCENARIOS[scenario_name]
    rng = np.random.default_rng(seed)
    theta = posterior[rng.integers(0, len(posterior), n)]
    direct_theta, isolation_theta, disease_theta, capture_theta, fort_protection = theta.T

    active = np.ones(n, dtype=bool)
    outcome = np.full(n, -1, dtype=np.int8)
    terminal_week = np.full(n, WEEKS, dtype=np.int16)
    central_order_week = np.full(n, -1, dtype=np.int16)

    initial_kyushu = 905.0 * rng.lognormal(0.0, 0.09, n)
    initial_honshu = 1180.0 * rng.lognormal(0.0, 0.11, n)
    initial_north = 310.0 * rng.lognormal(0.0, 0.14, n)
    node_weight = rng.dirichlet(np.where(TERRAIN == "mountain", 2.8, 2.0), n)
    defenders = node_weight * (initial_kyushu + initial_honshu)[:, None]
    defenders[:, KYUSHU] *= (initial_kyushu / np.maximum(defenders[:, KYUSHU].sum(axis=1), 1.0))[:, None]
    defenders[:, HONSHU] *= (initial_honshu / np.maximum(defenders[:, HONSHU].sum(axis=1), 1.0))[:, None]
    wounded = np.zeros((n, NODES))
    isolated = np.zeros((n, NODES))
    node_dead = np.zeros((n, NODES))
    node_captured = np.zeros((n, NODES))
    node_surrendered = np.zeros((n, NODES))
    control = np.zeros((n, NODES))
    fort = np.clip(TERRAIN_FORT[None, :] * rng.normal(0.80, 0.09, (n, NODES)), 0.35, 1.0)
    supply = np.clip(rng.normal(0.48, 0.09, (n, NODES)), 0.20, 0.72)
    command = np.clip(rng.normal(0.82, 0.07, (n, NODES)), 0.45, 0.96)
    resolve = np.clip(rng.beta(5.1, 2.0, (n, NODES)) * cfg["total_war"], 0.30, 0.995)
    order_received = np.zeros((n, NODES), dtype=bool)

    allied = 582.5 * rng.lognormal(0.0, 0.055, n)
    allied_reserve = 184.2 * rng.lognormal(0.0, 0.07, n)
    allied_strategic_pool = 6200.0 * rng.lognormal(0.0, 0.10, n)
    allied_cumulative_reinforcement = np.zeros(n)
    allied_kia = np.zeros(n)
    allied_wia = np.zeros(n)
    allied_nonbattle = np.zeros(n)
    allied_medical_queue = np.zeros(n)
    sealift_queue = np.zeros(n)
    shore_queue = np.zeros(n)
    inland_queue = np.zeros(n)
    allied_supply = np.clip(rng.normal(2.7, 0.32, n), 1.5, 3.8)
    shipping_damage = np.zeros(n)
    special_stock = np.clip(rng.normal(1.0, 0.16, n), 0.55, 1.45)

    air_stock = 8200.0 * rng.lognormal(0.0, 0.10, n)
    naval_stock = 145.0 * rng.lognormal(0.0, 0.14, n)
    initial_air_stock = air_stock.copy()
    initial_naval_stock = naval_stock.copy()
    north_stock = initial_north.copy()
    north_dead = np.zeros(n)
    north_disabled = np.zeros(n)
    civilian_population = 69900.0 * rng.lognormal(0.0, 0.025, n)
    initial_civilian_population = civilian_population.copy()
    nominal_pool_scale = 28000.0 if scenario_name in (
        "hundred_million_total_war", "emperor_threshold_surrender", "central_refusal_260_weeks",
        "extermination_refusal_416_weeks") else 9000.0
    nominal_total = nominal_pool_scale * rng.lognormal(0.0, 0.06, n)
    extreme_mobilization = scenario_name in (
        "hundred_million_total_war", "emperor_threshold_surrender", "central_refusal_260_weeks",
        "extermination_refusal_416_weeks")
    stage_mean = (np.array([0.60, 0.25, 0.08, 0.04, 0.03]) if extreme_mobilization
                  else np.array([0.805, 0.15, 0.03, 0.01, 0.005]))
    stage_share = rng.dirichlet(stage_mean * 80.0, n)
    mobilizable_pool = nominal_total * stage_share[:, 0]
    mobilized_registered = nominal_total * stage_share[:, 1]
    mobilized_trained = nominal_total * stage_share[:, 2]
    mobilized_equipped = nominal_total * stage_share[:, 3]
    initial_aux_deployed = nominal_total * stage_share[:, 4]
    initial_mobilizable_pool = nominal_total.copy()
    cumulative_aux_deployed = initial_aux_deployed.copy()
    defenders += initial_aux_deployed[:, None] * node_weight
    civilian_population -= initial_aux_deployed
    mfg_share = node_weight.copy()
    deployment_share = node_weight.copy()
    mfg_residual = np.zeros(n)
    mfg_residual_sum = np.zeros(n)
    mfg_updates = np.zeros(n)
    distribution_l1_sum = np.zeros(n)
    transport_cost_observed = np.zeros(n)
    transport_updates = np.zeros(n)
    overseas = np.column_stack([
        2000.0 * rng.lognormal(0.0, 0.08, n),
        320.0 * rng.lognormal(0.0, 0.10, n),
        600.0 * rng.lognormal(0.0, 0.10, n),
        850.0 * rng.lognormal(0.0, 0.10, n),
        400.0 * rng.lognormal(0.0, 0.12, n),
    ])
    initial_overseas = overseas.copy()
    overseas_dead = np.zeros_like(overseas)
    overseas_disabled = np.zeros_like(overseas)
    overseas_surrendered = np.zeros_like(overseas)
    overseas_order_received = np.zeros_like(overseas, dtype=bool)

    food = np.clip(rng.normal(0.52, 0.055, n), 0.34, 0.68)
    industry = np.clip(rng.normal(0.34, 0.05, n), 0.18, 0.50)
    governance = np.clip(rng.normal(0.76, 0.05, n), 0.58, 0.88)
    hardline = np.clip(rng.beta(5.2, 2.4, n) * cfg["total_war"], 0.25, 0.995)
    emperor = np.clip(rng.beta(2.0, 3.4, n), 0.02, 0.92)
    allied_tolerance = np.clip(rng.beta(4.8, 2.4, n), 0.25, 0.97)
    civilian_dead = np.zeros(n)
    bombing_dead = np.zeros(n)
    atomic_dead = np.zeros(n)
    atomic_strikes = np.zeros(n)
    atomic_inventory = rng.poisson(2.0, n).astype(float)
    nuclear_commitment = np.clip(rng.normal(-0.8, 0.25, n), -2.0, 1.0)
    nuclear_excitation = np.zeros(n)
    nuclear_stage = np.zeros(n, dtype=np.int8)
    nuclear_intercepted = np.zeros(n)
    nuclear_production_multiplier = rng.lognormal(0.0, 0.25, n)

    surrender_pressure = np.zeros(n)
    emperor_threshold = np.clip(
        rng.normal(3.7, 0.55, n) + 1.2 * hardline - 1.1 * emperor, 1.8, 6.5)
    surrender_trigger = np.zeros(n, dtype=np.int8)
    surrender_blocked_until = np.zeros(n, dtype=np.int16)
    palace_coup_attempts = np.zeros(n)
    palace_coup_successes = np.zeros(n)
    first_decision_cause = np.zeros(n, dtype=np.int8)
    first_decision_week = np.full(n, -1, dtype=np.int16)
    decision_risk_entries = np.zeros(n)

    coronet = np.zeros(n, dtype=bool)
    soviet_hokkaido = np.zeros(n, dtype=bool)
    paused = np.zeros(n, dtype=bool)
    pause_until = np.zeros(n, dtype=np.int16)
    ever_paused = np.zeros(n, dtype=bool)
    governance_collapsed = np.zeros(n, dtype=bool)
    central_refused_26 = np.ones(n, dtype=bool)
    central_refused_52 = np.ones(n, dtype=bool)
    central_refused_260 = np.ones(n, dtype=bool)
    central_refused_416 = np.ones(n, dtype=bool)
    total_war_path = np.zeros(n, dtype=bool)
    alive_at_26 = np.zeros(n, dtype=bool)
    alive_at_52 = np.zeros(n, dtype=bool)
    alive_at_104 = np.zeros(n, dtype=bool)
    alive_at_156 = np.zeros(n, dtype=bool)
    alive_at_260 = np.zeros(n, dtype=bool)
    regime_history = []
    death_direct = np.zeros(n)
    death_wounds = np.zeros(n)
    death_siege = np.zeros(n)
    death_disease = np.zeros(n)
    death_suicide = np.zeros(n)
    allied_effort_sum = np.zeros(n)
    japanese_effort_sum = np.zeros(n)
    effort_weeks = np.zeros(n)

    initial_accounted = (
        defenders.sum(axis=1) + initial_north + air_stock / 1000.0
        + naval_stock + overseas.sum(axis=1))

    for week in range(WEEKS):
        alive = active.copy()
        if not alive.any():
            break
        civilian_dead_start = civilian_dead.copy()
        if week == 25:
            alive_at_26 = active.copy()
        if week == 51:
            alive_at_52 = active.copy()
        if week == 103:
            alive_at_104 = active.copy()
        if week == 155:
            alive_at_156 = active.copy()
        if week == 259:
            alive_at_260 = active.copy()

        can_resume = paused & (week >= pause_until) & active
        resume_score = sigmoid(
            -0.6 + 1.3 * allied_tolerance + 1.0 * allied_supply
            - 1.1 * shipping_damage)
        resumed = can_resume & (rng.random(n) < 0.18 * resume_score)
        paused[resumed] = False

        weather = np.clip(rng.normal(0.80 + 0.12 * np.sin((week - 7) * np.pi / 26),
                                     0.10 * structure.stochasticity, n), 0.38, 1.02)
        blockade = cfg["blockade"] * (0.64 + 0.36 * min(week / 20.0, 1.0))
        food[alive] = np.clip(food[alive] - 0.0058 * blockade
                              + rng.normal(0, 0.0025, alive.sum()), 0.0, 0.72)
        industry[alive] = np.clip(industry[alive] - 0.0044 * blockade
                                  + rng.normal(0, 0.0022, alive.sum()), 0.0, 0.50)
        governance[alive] = np.clip(governance[alive] - 0.0025 * (1.0 - food[alive]), 0.0, 0.90)
        bombing = 0.35 * blockade * rng.lognormal(0.0, 0.18, n)
        bombing_dead[alive] += bombing[alive]
        civilian_dead[alive] += bombing[alive]

        production_increment = max(
            0.0, nuclear_stock_target(week + 1) - nuclear_stock_target(week))
        atomic_inventory += rng.poisson(
            production_increment * nuclear_production_multiplier) * active

        nuclear_excitation *= 0.90
        strategic_pressure = (
            0.55 * week / WEEKS + 0.65 * (allied_kia + allied_wia) / 500.0
            + 0.50 * control.mean(axis=1) + 0.35 * paused
            - 0.30 * allied_tolerance)
        heavy_jump = rng.random(n) < (
            0.004 + 0.012 * sigmoid(strategic_pressure + nuclear_excitation))
        jump_size = np.zeros(n)
        jump_size[heavy_jump] = np.clip(
            np.abs(rng.standard_t(3.0, heavy_jump.sum())) * 0.45, 0.0, 2.2)
        nuclear_commitment = np.clip(
            nuclear_commitment
            + 0.055 * (strategic_pressure + 0.75 * nuclear_commitment
                       - nuclear_commitment ** 3)
            + jump_size + rng.normal(0.0, 0.035, n), -2.5, 3.0)

        advance_prob = cfg["nuclear"] * 0.42 * sigmoid(
            nuclear_commitment + nuclear_excitation - 0.28 * nuclear_stage)
        intercept_prob = 0.10 * sigmoid(
            0.8 - nuclear_commitment + 0.35 * nuclear_stage
            + 0.45 * allied_tolerance)
        advance = active & (nuclear_stage < 3) & (rng.random(n) < advance_prob)
        intercepted = active & (nuclear_stage > 0) & (~advance) & (
            rng.random(n) < intercept_prob)
        nuclear_stage[advance] += 1
        nuclear_stage[intercepted] -= 1
        nuclear_intercepted[intercepted] += 1

        strike = (active & (nuclear_stage >= 3) & (atomic_inventory >= 1.0)
                  & (weather > 0.55) & (rng.random(n) < 0.48))
        if strike.any():
            diminishing_target = np.exp(-0.035 * atomic_strikes[strike])
            d = (rng.lognormal(np.log(72.0), 0.34, strike.sum())
                 * np.clip(diminishing_target, 0.22, 1.0))
            atomic_inventory[strike] -= 1.0
            atomic_strikes[strike] += 1
            atomic_dead[strike] += d
            civilian_dead[strike] += d
            industry[strike] *= 0.88
            governance[strike] *= 0.92
            nuclear_excitation[strike] += 1.20
            nuclear_commitment[strike] = np.clip(
                nuclear_commitment[strike] + 0.35, -2.5, 3.0)
            nuclear_stage[strike] = 1

        # Overseas theaters remain separate stocks. Pressure combines Soviet,
        # Chinese and Commonwealth campaigns without permitting free return.
        theater_pressure = np.array([0.0075, 0.0045, 0.0055, 0.0040, 0.0035])
        overseas_loss = overseas * theater_pressure[None, :] * (
            0.70 + 0.30 * min(week / 26.0, 1.0)) * active[:, None]
        overseas -= overseas_loss
        fatal_share = np.array([0.55, 0.45, 0.50, 0.58, 0.62])[None, :]
        surrender_share_theater = np.array([0.05, 0.08, 0.07, 0.04, 0.03])[None, :]
        overseas_dead += overseas_loss * fatal_share
        overseas_surrendered += overseas_loss * surrender_share_theater
        overseas_disabled += overseas_loss * (1.0 - fatal_share - surrender_share_theater)

        # Aircraft and residual naval units are capability stocks, while the
        # northern army is a personnel stock activated only if that front opens.
        air_stock *= 1.0 - active * (0.010 + 0.006 * blockade)
        naval_stock *= 1.0 - active * (0.006 + 0.004 * blockade)
        north_loss = np.minimum(
            north_stock, north_stock * soviet_hokkaido * (0.010 + 0.004 * blockade))
        north_stock -= north_loss
        north_dead += 0.52 * north_loss
        north_disabled += 0.48 * north_loss

        # Delay comparator withholds the amphibious phase while blockade runs.
        invaded = not (scenario_name == "blockade_delay_18_weeks" and week < 18)
        if invaded:
            combat_active = active & (~paused)
            attack_window = np.exp(-max(0, week - (18 if scenario_name == "blockade_delay_18_weeks" else 0)) / 8.0)
            special = 0.050 * special_stock * attack_window * (1.0 - 0.72) * rng.lognormal(0, 0.28, n)
            shipping_damage[combat_active] = np.clip(
                shipping_damage[combat_active] + special[combat_active] - 0.010, 0.0, 0.65)
            special_stock[combat_active] = np.maximum(
                0.0, special_stock[combat_active] - 0.065 * attack_window)
            allied_kia += 7.5 * special * combat_active
            allied_wia += 28.0 * special * combat_active

            arrivals = 18.0 * weather * (1.0 - shipping_damage)
            sealift_queue[combat_active] += arrivals[combat_active]
            unload = np.minimum(sealift_queue, 14.8 * weather * (1.0 - 0.45 * shipping_damage))
            sealift_queue -= unload
            shore_queue += unload
            inland = np.minimum(shore_queue, 12.8 * (0.92 - 0.18 * control[:, KYUSHU].mean(axis=1)))
            shore_queue -= inland
            inland_queue += inland
            delivered = np.minimum(inland_queue, 11.2)
            inland_queue -= delivered
            allied_supply += delivered / 42.0
            queue_index = (sealift_queue + shore_queue + inland_queue) / 100.0
            logistics = np.clip(0.30 + 0.52 * np.tanh(allied_supply / 2.1)
                                - structure.queue_weight * 0.15 * np.tanh(queue_index)
                                - 0.22 * shipping_damage, 0.08, 0.98)
            allied_supply = np.clip(allied_supply - (0.45 + 0.0006 * allied), 0.0, 5.0)
            if week < 14:
                reinf = np.minimum(allied_reserve, 10.5 * logistics) * combat_active
                allied += reinf
                allied_reserve -= reinf
            else:
                long_reinf = np.minimum(
                    allied_strategic_pool,
                    np.clip(3.0 + 7.0 * logistics * allied_tolerance
                            - 3.0 * shipping_damage, 0.0, 11.0) * combat_active)
                allied += long_reinf
                allied_strategic_pool -= long_reinf
                allied_cumulative_reinforcement += long_reinf
        else:
            combat_active = np.zeros(n, dtype=bool)
            logistics = np.zeros(n)
            queue_index = np.zeros(n)

        kyushu_control = control[:, KYUSHU].mean(axis=1)
        coronet_hazard = 0.045 * sigmoid(9.0 * (kyushu_control - 0.18)) * logistics
        coronet_start = (combat_active & (~coronet) & (week >= 17) & (kyushu_control > 0.12)
                         & (logistics > 0.30) & (rng.random(n) < coronet_hazard))
        coronet[coronet_start] = True

        if invaded:
            landing, beachhead, mobile, siege, pursuit, collapse = regime_shares(
                control, fort, supply, command, week, coronet)
            eligible = np.zeros((n, NODES), dtype=bool)
            eligible[:, KYUSHU] = combat_active[:, None]
            eligible[:, HONSHU] = (combat_active & coronet)[:, None]

            if structure.network_weight > 0:
                friendly_neighbor = ((1.0 - control) @ ADJ) / DEGREE[None, :]
                enemy_neighbor = (control @ ADJ) / DEGREE[None, :]
            else:
                friendly_neighbor = np.repeat((1.0 - control.mean(axis=1))[:, None], NODES, axis=1)
                enemy_neighbor = np.repeat(control.mean(axis=1)[:, None], NODES, axis=1)

            supply_diffusion = ((supply @ ADJ) / DEGREE[None, :] - supply)
            supply = np.clip(
                supply + structure.network_weight * 0.025 * supply_diffusion
                - 0.010 * blockade - 0.010 * enemy_neighbor
                + rng.normal(0, 0.003 * structure.stochasticity, (n, NODES)), 0.0, 0.78)
            command = np.clip(command - 0.006 * enemy_neighbor - 0.005 * (1.0 - supply), 0.0, 0.98)

            terrain_protection = np.clip(
                1.15 - fort_protection[:, None] * fort * TERRAIN_FORT[None, :], 0.18, 1.05)
            regime_exposure = (
                1.18 * landing + 1.02 * beachhead + 0.92 * mobile
                + 0.62 * siege + 1.08 * pursuit + 1.30 * collapse)
            attack_weight = eligible * TERRAIN_ACCESS[None, :] * (1.05 - 0.55 * control)
            attack_weight /= np.maximum(attack_weight.sum(axis=1, keepdims=True), 1e-9)
            allied_pressure = allied[:, None] * logistics[:, None] * attack_weight
            defender_effective = defenders * (0.24 + 0.76 * supply) * (0.35 + 0.65 * command)
            local_ratio = np.clip(allied_pressure / np.maximum(defender_effective, 0.5), 0.05, 8.0)

            allied_effort = np.ones(n)
            japanese_effort = np.ones(n)
            if structure.game_weight > 0:
                horizon = 4.0 if week < 104 else 13.0
                candidates = np.array([0.45, 0.65, 0.85, 1.05, 1.25])
                defense_pressure = np.sqrt(
                    np.maximum(defender_effective.sum(axis=1), 1.0)
                    / np.maximum(allied * logistics, 1.0))
                remaining_control = 1.0 - control.mean(axis=1)
                total_home_stock = np.maximum(
                    defenders.sum(axis=1) + wounded.sum(axis=1) + isolated.sum(axis=1), 1.0)
                # Receding-horizon game MPC: alternate bounded best responses
                # over a four-week forecast, then apply only the first control.
                for _ in range(3):
                    a = candidates[None, :]
                    j_fixed = japanese_effort[:, None]
                    predicted_allied_loss = (
                        horizon * 0.0038 * defense_pressure[:, None] * allied[:, None]
                        * j_fixed * (0.90 + 0.15 * a))
                    predicted_progress = (
                        horizon * 0.018 * logistics[:, None] * a
                        / (0.82 + 0.18 * j_fixed))
                    allied_cost = (
                        predicted_allied_loss / 18.0
                        + 6.50 * np.maximum(remaining_control[:, None] - predicted_progress, 0.0)
                        + (0.045 / np.maximum(allied_tolerance[:, None], 0.20)) * a ** 2
                        + 0.30 * queue_index[:, None] * a)
                    allied_effort = candidates[np.argmin(allied_cost, axis=1)]

                    j = candidates[None, :]
                    a_fixed = allied_effort[:, None]
                    predicted_japanese_loss = (
                        horizon * 0.008 * total_home_stock[:, None] * a_fixed
                        * (0.86 + 0.20 * j))
                    predicted_progress_j = (
                        horizon * 0.018 * logistics[:, None] * a_fixed
                        / (0.82 + 0.18 * j))
                    japanese_cost = (
                        predicted_japanese_loss / 65.0
                        + (10.0 + 80.0 * hardline[:, None] ** 2) * predicted_progress_j
                        + (0.15 / np.maximum(hardline[:, None], 0.20)) * j ** 2
                        + 0.30 * (1.0 - food[:, None]) * j)
                    japanese_effort = candidates[np.argmin(japanese_cost, axis=1)]
            allied_effort_sum += allied_effort * combat_active
            japanese_effort_sum += japanese_effort * combat_active
            effort_weeks += combat_active

            japanese_loss_rate = (
                direct_theta[:, None] / 5.0 * terrain_protection * regime_exposure
                * np.sqrt(local_ratio) * (0.72 + 0.42 * structure.smooth_weight))
            japanese_loss_rate *= rng.lognormal(
                0, 0.16 * structure.stochasticity, (n, NODES))
            japanese_loss_rate *= allied_effort[:, None] * (
                0.86 + 0.20 * japanese_effort[:, None])
            j_loss = np.minimum(defenders, defenders * japanese_loss_rate * eligible)
            defenders -= j_loss

            direct_death_share = np.clip(
                0.38 + 0.13 * regime_exposure + 0.08 * (1.0 - supply), 0.42, 0.72)
            capture_share = np.clip(
                0.005 + 0.045 * (1.0 - resolve) * enemy_neighbor
                * capture_theta[:, None], 0.003, 0.10)
            wound_share = np.maximum(0.0, 1.0 - direct_death_share - capture_share)
            d_direct = j_loss * direct_death_share
            c_direct = j_loss * capture_share
            w_new = j_loss * wound_share
            node_dead += d_direct
            node_captured += c_direct
            wounded += w_new
            death_direct += d_direct.sum(axis=1)

            isolation_rate = np.clip(
                isolation_theta[:, None] / 4.0 * enemy_neighbor
                * (1.0 - friendly_neighbor) * fort
                * (0.45 + 0.55 * structure.isolation_weight), 0.0, 0.06)
            new_isolated = np.minimum(defenders, defenders * isolation_rate * eligible)
            defenders -= new_isolated
            isolated += new_isolated

            siege_death_rate = np.clip(
                0.006 + isolation_theta[:, None] / 5.0 * (1.0 - supply)
                * (0.55 + 0.45 * resolve), 0.004, 0.10)
            siege_capture_rate = np.clip(
                0.002 + 0.025 * (1.0 - resolve) * enemy_neighbor
                * capture_theta[:, None], 0.002, 0.055)
            breakout_rate = np.clip(0.020 * supply * command * friendly_neighbor, 0.0, 0.02)
            d_siege = isolated * siege_death_rate * eligible
            c_siege = isolated * siege_capture_rate * eligible
            breakout = isolated * breakout_rate * eligible
            flow = d_siege + c_siege + breakout
            scale = np.minimum(1.0, isolated / np.maximum(flow, 1e-9))
            d_siege *= scale
            c_siege *= scale
            breakout *= scale
            isolated -= d_siege + c_siege + breakout
            defenders += breakout
            node_dead += d_siege
            node_captured += c_siege
            death_siege += d_siege.sum(axis=1)

            medical = np.clip(0.10 + 0.46 * supply + 0.20 * command + 0.16 * industry[:, None], 0.05, 0.82)
            wound_death_rate = np.clip(0.015 + 0.075 * (1.0 - medical), 0.015, 0.10)
            wound_return_rate = np.clip(0.085 * medical * food[:, None], 0.004, 0.06)
            wound_capture_rate = np.clip(0.003 + 0.018 * enemy_neighbor * (1.0 - command), 0.002, 0.025)
            d_wound = wounded * wound_death_rate * eligible
            c_wound = wounded * wound_capture_rate * eligible
            r_wound = wounded * wound_return_rate * eligible
            w_flow = d_wound + c_wound + r_wound
            w_scale = np.minimum(1.0, wounded / np.maximum(w_flow, 1e-9))
            d_wound *= w_scale
            c_wound *= w_scale
            r_wound *= w_scale
            wounded -= d_wound + c_wound + r_wound
            defenders += r_wound
            node_dead += d_wound
            node_captured += c_wound
            death_wounds += d_wound.sum(axis=1)

            disease_rate = np.clip(
                disease_theta[:, None] / 10.0 * (1.0 - supply) ** 2
                * (0.55 + 0.45 * (1.0 - food[:, None])), 0.0003, 0.025)
            disease_loss = np.minimum(defenders, defenders * disease_rate * eligible)
            defenders -= disease_loss
            d_disease = disease_loss * np.clip(0.44 + 0.28 * (1.0 - supply), 0.44, 0.72)
            wounded += disease_loss - d_disease
            node_dead += d_disease
            death_disease += d_disease.sum(axis=1)

            suicide_rate = np.clip(
                0.002 * collapse * resolve * (1.0 - command), 0.0, 0.002)
            d_suicide = np.minimum(defenders, defenders * suicide_rate)
            defenders -= d_suicide
            node_dead += d_suicide
            death_suicide += d_suicide.sum(axis=1)

            # Allied casualties depend on the active defensive system, not on a
            # prescribed exchange ratio.
            defender_fire = defender_effective * fort * regime_exposure
            allied_ground_loss = (
                0.0038 * np.sqrt(np.maximum(defender_fire.sum(axis=1), 1.0)
                                 / np.maximum(allied * logistics, 1.0))
                * allied * active * rng.lognormal(0, 0.16 * structure.stochasticity, n))
            allied_ground_loss *= japanese_effort * (0.90 + 0.15 * allied_effort)
            allied_ground_loss *= combat_active
            allied_ground_loss = np.minimum(allied, allied_ground_loss)
            allied -= allied_ground_loss
            allied_kia += 0.235 * allied_ground_loss
            allied_wia += 0.765 * allied_ground_loss
            allied_nonbattle += combat_active * (
                0.45 + 0.72 * (1.0 - logistics) + 0.30 * queue_index)
            allied_medical_queue += 0.765 * allied_ground_loss
            treated = np.minimum(allied_medical_queue, 15.5 * (0.72 + 0.28 * logistics))
            allied_medical_queue -= treated
            med_overflow = np.maximum(0.0, allied_medical_queue - 28.0)
            allied_kia += 0.045 * med_overflow

            progress_delta = np.clip(
                0.010 + 0.020 * np.tanh(np.log(np.maximum(local_ratio, 0.05)))
                + 0.009 * logistics[:, None] - 0.006 * fort, -0.001, 0.045)
            progress_delta *= allied_effort[:, None] / (
                0.82 + 0.18 * japanese_effort[:, None])
            control = np.clip(control + progress_delta * eligible, 0.0, 1.0)
            fort = np.clip(fort - 0.007 * eligible * (0.6 + 0.4 * control), 0.06, 1.0)
            civilian_dead += combat_active * (
                0.40 + 1.30 * np.mean(eligible * (control > 0.05) * (control < 0.95), axis=1)
                + 0.85 * (1.0 - food))

            if week in (0, 7, 16, 25, 38, 51, 77, 103, 155, 207, 259, 311, 415):
                regime_history.append({
                    "week": week + 1,
                    "landing": float(landing.mean()),
                    "beachhead": float(beachhead.mean()),
                    "mobile": float(mobile.mean()),
                    "siege": float(siege.mean()),
                    "pursuit": float(pursuit.mean()),
                    "collapse": float(collapse.mean()),
                })

        # Multi-year mobilization pipeline. The nominal pool is not combat
        # strength: registration, training, equipment and transport are
        # separate capacity-constrained stocks.
        register_rate = np.clip(
            (0.0012 + 0.0018 * cfg["total_war"]) * governance * (0.35 + 0.65 * food),
            0.0, 0.006)
        register_flow = np.minimum(mobilizable_pool, mobilizable_pool * register_rate * active)
        mobilizable_pool -= register_flow
        mobilized_registered += register_flow

        train_flow = np.minimum(
            mobilized_registered,
            mobilized_registered * 0.010 * food * governance * active)
        mobilized_registered -= train_flow
        mobilized_trained += train_flow

        equip_flow = np.minimum(
            mobilized_trained,
            mobilized_trained * 0.014 * industry * (0.45 + 0.55 * food) * active)
        mobilized_trained -= equip_flow
        mobilized_equipped += equip_flow

        update_interval = 4 if week < 104 else 13
        if week % update_interval == 0:
            need = np.clip(
                0.45 * (1.0 - control) + 0.35 * (1.0 - supply)
                + 0.20 * (defenders.mean(axis=1, keepdims=True)
                          / np.maximum(defenders, 1.0)), 0.0, 3.0)
            risk = np.clip(control + 0.35 * (1.0 - command), 0.0, 1.5)
            node_capacity = np.clip(
                TERRAIN_ACCESS[None, :] * (0.40 + 0.60 * supply), 0.10, 1.30)
            transport_penalty = np.min(TRANSPORT_COST, axis=0)[None, :]
            mfg_share, mfg_residual = mean_field_distribution(
                mfg_share, need, risk, node_capacity, transport_penalty)
            source_mass = mobilized_equipped[:, None] * np.array([0.31, 0.24, 0.25, 0.20])[None, :]
            deployment_share, ot_cost = entropic_transport_share(source_mass, mfg_share)
            mfg_residual_sum += mfg_residual * active
            mfg_updates += active
            distribution_l1_sum += np.abs(deployment_share - mfg_share).sum(axis=1) * active
            transport_cost_observed += ot_cost * active
            transport_updates += active

        deploy_rate = np.clip(
            0.018 * supply.mean(axis=1) * governance
            * np.exp(-0.85 * transport_cost_observed / np.maximum(transport_updates, 1.0)),
            0.0, 0.018)
        deploy_flow = np.minimum(
            mobilized_equipped, mobilized_equipped * deploy_rate * active)
        mobilized_equipped -= deploy_flow
        cumulative_aux_deployed += deploy_flow
        civilian_population = np.maximum(0.0, civilian_population - deploy_flow)
        defenders += deploy_flow[:, None] * deployment_share
        total_war_path |= cumulative_aux_deployed > 1000.0

        # Central political decision is separate from command propagation.
        total_home_dead = node_dead.sum(axis=1)
        home_initial = initial_kyushu + initial_honshu + cumulative_aux_deployed
        fighting_now = defenders.sum(axis=1) + wounded.sum(axis=1) + isolated.sum(axis=1)
        capital_control = control[:, CAPITAL_SYSTEM].mean(axis=1)
        military_depletion = 1.0 - fighting_now / np.maximum(home_initial, 1.0)
        civilian_loss_fraction = civilian_dead / np.maximum(initial_civilian_population, 1.0)
        capital_collapse = capital_control > 0.68
        national_ruin = (food < 0.08) & (industry < 0.06) & (governance < 0.24)
        population_destruction = civilian_loss_fraction > 0.32
        military_collapse = military_depletion > 0.68
        nuclear_cascade = (atomic_strikes >= 5) | (atomic_dead > 320.0)

        surrender_signal = (
            1.60 * capital_control + 1.10 * military_depletion
            + 1.45 * civilian_loss_fraction + 0.80 * (1.0 - food)
            + 0.65 * (1.0 - industry) + 0.45 * (1.0 - governance)
            + 0.24 * atomic_strikes + 0.55 * nuclear_excitation
            + 0.70 * emperor - 1.05 * hardline)
        surrender_pressure = np.clip(
            0.92 * surrender_pressure + 0.08 * surrender_signal
            + 0.18 * strike + rng.normal(0.0, 0.025, n), -1.0, 10.0)
        threshold_crossed = surrender_pressure >= emperor_threshold

        unset = surrender_trigger == 0
        surrender_trigger[unset & capital_collapse] = 1
        surrender_trigger[unset & (~capital_collapse) & national_ruin] = 2
        surrender_trigger[unset & (~capital_collapse) & (~national_ruin)
                          & population_destruction] = 3
        surrender_trigger[unset & (~capital_collapse) & (~national_ruin)
                          & (~population_destruction) & military_collapse] = 4
        surrender_trigger[unset & (~capital_collapse) & (~national_ruin)
                          & (~population_destruction) & (~military_collapse)
                          & nuclear_cascade] = 5
        surrender_trigger[unset & threshold_crossed] = 6

        can_order = (active & (week >= cfg["no_surrender_until"])
                     & (week >= surrender_blocked_until) & (central_order_week < 0))
        decision_ready = can_order & (threshold_crossed | capital_collapse | national_ruin)
        decision_risk_entries += decision_ready

        # Stratified semi-Markov competing risks. The structure-specific term changes
        # the baseline hazard; all remaining terms are time-varying covariates.
        stratum_order = 0.18 * structure.smooth_weight - 0.10 * structure.stochasticity
        stratum_block = 0.12 * structure.network_weight + 0.08 * structure.stochasticity
        pressure_gap = surrender_pressure - emperor_threshold
        order_log_hazard = (
            -3.80 + stratum_order + 1.20 * pressure_gap + 0.82 * capital_collapse
            + 0.72 * national_ruin + 0.52 * population_destruction
            + 0.48 * military_collapse + 0.58 * emperor - 0.48 * hardline
        )
        block_log_hazard = (
            -5.10 + stratum_block + 1.05 * pressure_gap + 1.15 * hardline
            + 0.55 * command.mean(axis=1) - 1.10 * emperor
            - 0.72 * capital_control - 0.62 * military_depletion
        )
        h_order = np.exp(np.clip(order_log_hazard, -12.0, -0.35)) * decision_ready
        h_block = np.exp(np.clip(block_log_hazard, -12.0, -0.35)) * decision_ready
        h_total = h_order + h_block
        decision_event = decision_ready & (rng.random(n) < (1.0 - np.exp(-h_total)))
        block_event = decision_event & (rng.random(n) < np.divide(
            h_block, h_total, out=np.zeros_like(h_total), where=h_total > 0))
        issue_order = decision_event & (~block_event)

        first = decision_event & (first_decision_cause == 0)
        first_decision_cause[first & issue_order] = 1
        first_decision_cause[first & block_event] = 2
        first_decision_week[first] = week + 1
        palace_coup_attempts[block_event] += 1
        palace_coup_successes[block_event] += 1
        coup_success = block_event
        coup_failed = np.zeros(n, dtype=bool)
        if coup_success.any():
            surrender_blocked_until[coup_success] = (
                week + rng.integers(3, 15, coup_success.sum()))
            emperor_threshold[coup_success] = np.minimum(
                7.0, emperor_threshold[coup_success] + 0.30)
            hardline[coup_success] = np.minimum(0.995, hardline[coup_success] + 0.08)
        if coup_failed.any():
            hardline[coup_failed] *= 0.88
            emperor[coup_failed] = np.minimum(1.0, emperor[coup_failed] + 0.10)

        central_order_week[issue_order] = week + 1
        if week < 26:
            central_refused_26 &= central_order_week < 0
        if week < 52:
            central_refused_52 &= central_order_week < 0
        if week < 260:
            central_refused_260 &= central_order_week < 0
        central_refused_416 &= central_order_week < 0

        has_order = central_order_week >= 0
        if has_order.any():
            if structure.network_weight > 0:
                order_neighbor = (order_received.astype(float) @ ADJ) / DEGREE[None, :]
            else:
                order_neighbor = np.repeat(order_received.mean(axis=1)[:, None], NODES, axis=1)
            receive_prob = np.clip(
                0.10 + 0.55 * command + 0.28 * order_neighbor * structure.network_weight,
                0.05, 0.92)
            receive = has_order[:, None] & (~order_received) & (rng.random((n, NODES)) < receive_prob)
            order_received |= receive
            comply_prob = sigmoid(
                0.4 + 2.1 * command - 2.2 * resolve + 0.8 * (1.0 - supply))
            comply = order_received & (rng.random((n, NODES)) < comply_prob)
            for stock in (defenders, wounded, isolated):
                moved = stock * comply * np.clip(0.62 + 0.30 * command, 0.0, 0.95)
                stock -= moved
                node_surrendered += moved
            order_received[comply] = False

            theater_receive_prob = np.clip(
                0.05 + 0.11 * np.arange(1, 6)[None, :] + 0.18 * governance[:, None],
                0.08, 0.78)
            theater_receive = (has_order[:, None] & (~overseas_order_received)
                               & (rng.random(overseas.shape) < theater_receive_prob))
            overseas_order_received |= theater_receive
            theater_comply = overseas_order_received & (
                rng.random(overseas.shape) < (0.18 + 0.52 * governance[:, None]))
            theater_moved = overseas * theater_comply * 0.72
            overseas -= theater_moved
            overseas_surrendered += theater_moved

        fighting_stock = defenders.sum(axis=1) + isolated.sum(axis=1)
        surrender_share = node_surrendered.sum(axis=1) / np.maximum(home_initial, 1.0)
        ordered_terminal = active & has_order & (
            (surrender_share > 0.62) | (fighting_stock < 0.18 * home_initial))
        coordinated = ordered_terminal & (surrender_share > 0.82)
        fragmented = ordered_terminal & (~coordinated)
        outcome[coordinated] = 0
        outcome[fragmented] = 1
        terminal_week[ordered_terminal] = week + 1
        active[ordered_terminal] = False

        total_allied = allied_kia + allied_wia + allied_nonbattle
        pause_hazard = 0.026 * sigmoid(
            -2.2 + total_allied / 145.0 + 1.25 * shipping_damage
            + allied_medical_queue / 35.0 - 1.0 * control.mean(axis=1) - 0.8 * allied_tolerance)
        pause = combat_active & (~paused) & (rng.random(n) < pause_hazard)
        if pause.any():
            paused[pause] = True
            ever_paused[pause] = True
            pause_until[pause] = week + rng.integers(8, 40, pause.sum())

        collapse_hazard = 0.024 * sigmoid(-4.0 + 4.0 * (0.18 - food) + 2.5 * (0.30 - governance))
        humanitarian = active & (~governance_collapsed) & (rng.random(n) < collapse_hazard)
        governance_collapsed[humanitarian] = True
        governance[humanitarian] *= 0.55

        famine_rate = np.clip(
            0.00005 + 0.0040 * np.maximum(0.30 - food, 0.0) ** 1.5
            + 0.0015 * (1.0 - governance) + 0.0008 * governance_collapsed,
            0.0, 0.005)
        civilian_dead += civilian_population * famine_rate * active
        weekly_civilian_loss = np.maximum(0.0, civilian_dead - civilian_dead_start)
        loss_scale = np.minimum(
            1.0, civilian_population / np.maximum(weekly_civilian_loss, 1e-12))
        civilian_dead = civilian_dead_start + weekly_civilian_loss * loss_scale
        civilian_population -= weekly_civilian_loss * loss_scale

        hokkaido = (active & (week >= 20) & (~soviet_hokkaido)
                    & (rng.random(n) < 0.0009 * (1.0 + week / WEEKS)))
        soviet_hokkaido[hokkaido] = True

    outcome[active & paused] = 3
    outcome[active & governance_collapsed & (~paused)] = 4
    outcome[active & (~governance_collapsed) & (~paused)] = 5
    total_allied = allied_kia + allied_wia + allied_nonbattle
    home_dead = node_dead.sum(axis=1)
    home_captured = node_captured.sum(axis=1)
    home_surrendered = node_surrendered.sum(axis=1)
    home_remaining = defenders.sum(axis=1) + wounded.sum(axis=1) + isolated.sum(axis=1)
    home_balance = home_dead + home_captured + home_surrendered + home_remaining - (
        initial_kyushu + initial_honshu + cumulative_aux_deployed)
    assert np.max(np.abs(home_balance)) < 1e-6
    mobilization_balance = (
        mobilizable_pool + mobilized_registered + mobilized_trained
        + mobilized_equipped + cumulative_aux_deployed - initial_mobilizable_pool)
    assert np.max(np.abs(mobilization_balance)) < 1e-6
    civilian_balance = (
        civilian_population + cumulative_aux_deployed + civilian_dead
        - initial_civilian_population)
    assert np.max(np.abs(civilian_balance)) < 1e-6
    overseas_balance = (overseas + overseas_dead + overseas_disabled
                        + overseas_surrendered - initial_overseas).sum(axis=1)
    assert np.max(np.abs(overseas_balance)) < 1e-6
    north_balance = north_stock + north_dead + north_disabled - initial_north
    assert np.max(np.abs(north_balance)) < 1e-6

    arrays = {
        "allied_kia_thousands": allied_kia,
        "allied_total_casualties_thousands": total_allied,
        "japanese_main_invasion_front_dead_thousands": home_dead,
        "japanese_home_dead_thousands": home_dead + north_dead,
        "japanese_overseas_dead_thousands": overseas_dead.sum(axis=1),
        "japanese_overseas_disabled_thousands": overseas_disabled.sum(axis=1),
        "japanese_overseas_surrendered_thousands": overseas_surrendered.sum(axis=1),
        "japanese_hokkaido_northern_defense_dead_thousands": north_dead,
        "japanese_military_total_dead_thousands": home_dead + overseas_dead.sum(axis=1) + north_dead,
        "japanese_total_dead_thousands": (
            home_dead + overseas_dead.sum(axis=1) + north_dead + civilian_dead),
        "japanese_home_captured_thousands": home_captured,
        "japanese_home_surrendered_thousands": home_surrendered,
        "japanese_home_remaining_thousands": home_remaining,
        "japanese_nominal_mobilization_pool_remaining_thousands": mobilizable_pool,
        "japanese_registered_not_trained_thousands": mobilized_registered,
        "japanese_trained_not_equipped_thousands": mobilized_trained,
        "japanese_equipped_not_deployed_thousands": mobilized_equipped,
        "japanese_cumulative_auxiliary_deployed_thousands": cumulative_aux_deployed,
        "japanese_to_allied_death_ratio": home_dead / np.maximum(allied_kia, 0.5),
        "japanese_civilian_dead_thousands": civilian_dead,
        "japanese_civilian_population_remaining_thousands": civilian_population,
        "allied_field_strength_remaining_thousands": allied,
        "allied_strategic_pool_remaining_thousands": allied_strategic_pool,
        "allied_cumulative_reinforcement_thousands": allied_cumulative_reinforcement,
        "terminal_week": terminal_week.astype(float),
        "aircraft_stock_remaining": air_stock,
        "naval_stock_remaining": naval_stock,
    }
    conditions = {
        "all_paths": np.ones(n, dtype=bool),
        "coronet_launched": coronet,
        "war_active_at_26_weeks": alive_at_26,
        "war_active_at_52_weeks": alive_at_52,
        "war_active_at_104_weeks": alive_at_104,
        "war_active_at_156_weeks": alive_at_156,
        "war_active_at_260_weeks": alive_at_260,
        "war_active_at_416_weeks": active,
        "central_refused_through_26_weeks": central_refused_26,
        "central_refused_through_52_weeks": central_refused_52,
        "central_refused_through_260_weeks": central_refused_260,
        "central_refused_through_416_weeks": central_refused_416,
        "total_war_mobilization_realized": total_war_path,
        "amphibious_operations_ever_paused": ever_paused,
    }
    conditional = {name: conditional_summary(mask, arrays) for name, mask in conditions.items()}
    return {
        "structure": structure.name,
        "scenario": scenario_name,
        "trajectories": n,
        "outcome_distribution": {OUTCOMES[i]: float(np.mean(outcome == i)) for i in range(6)},
        "unconditional": {name: q(values) for name, values in arrays.items()},
        "conditional": conditional,
        "death_decomposition": {
            "direct_combat": q(death_direct),
            "died_of_wounds": q(death_wounds),
            "siege_and_isolation": q(death_siege),
            "disease_and_starvation": q(death_disease),
            "suicide_and_final_resistance": q(death_suicide),
        },
        "atomic_strikes": q(atomic_strikes),
        "atomic_inventory_remaining": q(atomic_inventory),
        "nuclear_policy_interceptions": q(nuclear_intercepted),
        "atomic_civilian_dead_thousands": q(atomic_dead),
        "conventional_bombing_dead_thousands": q(bombing_dead),
        "food_index": q(food),
        "coronet_launch_share": float(coronet.mean()),
        "central_surrender_order_share": float(np.mean(central_order_week >= 0)),
        "central_surrender_order_week": q(
            central_order_week[central_order_week >= 0].astype(float)),
        "surrender_trigger_distribution": {
            "capital_system_collapse": float(np.mean(
                (central_order_week >= 0) & (surrender_trigger == 1))),
            "national_system_ruin": float(np.mean(
                (central_order_week >= 0) & (surrender_trigger == 2))),
            "population_destruction": float(np.mean(
                (central_order_week >= 0) & (surrender_trigger == 3))),
            "military_depletion": float(np.mean(
                (central_order_week >= 0) & (surrender_trigger == 4))),
            "nuclear_cascade": float(np.mean(
                (central_order_week >= 0) & (surrender_trigger == 5))),
            "cumulative_threshold": float(np.mean(
                (central_order_week >= 0) & (surrender_trigger == 6))),
        },
        "palace_coup_attempts": q(palace_coup_attempts),
        "palace_coup_successes": q(palace_coup_successes),
        "decision_competing_risks": {
            "model": "structure-stratified semi-Markov cause-specific Cox hazards",
            "first_successful_order": float(np.mean(first_decision_cause == 1)),
            "first_palace_block": float(np.mean(first_decision_cause == 2)),
            "no_decision_event": float(np.mean(first_decision_cause == 0)),
            "first_event_week": q(first_decision_week[first_decision_week >= 0].astype(float)),
            "mean_weeks_at_risk": float(np.mean(decision_risk_entries)),
            "recurrent_blocking": "A blocked path returns to the risk set after its blocking interval.",
        },
        "soviet_hokkaido_share": float(soviet_hokkaido.mean()),
        "mass_balance_error_thousands": q(home_balance),
        "overseas_mass_balance_error_thousands": q(overseas_balance),
        "northern_mass_balance_error_thousands": q(north_balance),
        "mobilization_mass_balance_error_thousands": q(mobilization_balance),
        "civilian_population_balance_error_thousands": q(civilian_balance),
        "aircraft_stock_loss_share": q(1.0 - air_stock / initial_air_stock),
        "naval_stock_loss_share": q(1.0 - naval_stock / initial_naval_stock),
        "mean_allied_feedback_effort": q(
            allied_effort_sum / np.maximum(effort_weeks, 1.0)),
        "mean_japanese_feedback_effort": q(
            japanese_effort_sum / np.maximum(effort_weeks, 1.0)),
        "mean_field_fixed_point_l1_residual": q(
            mfg_residual_sum / np.maximum(mfg_updates, 1.0)),
        "mfg_to_transport_distribution_l1": q(
            distribution_l1_sum / np.maximum(mfg_updates, 1.0)),
        "mean_optimal_transport_cost": q(
            transport_cost_observed / np.maximum(transport_updates, 1.0)),
        "regime_history": regime_history,
    }


def ensemble_summary(results):
    grouped = {}
    for scenario in SCENARIOS:
        rows = [r for r in results if r["scenario"] == scenario]
        metrics = {}
        for metric in (
            "allied_kia_thousands", "allied_total_casualties_thousands",
            "japanese_home_dead_thousands", "japanese_total_dead_thousands",
            "japanese_to_allied_death_ratio", "japanese_civilian_dead_thousands",
        ):
            medians = np.array([r["unconditional"][metric]["median"] for r in rows])
            metrics[metric] = {
                "model_min_median": float(medians.min()),
                "ensemble_median_of_medians": float(np.median(medians)),
                "model_max_median": float(medians.max()),
            }
        grouped[scenario] = metrics
    return grouped


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    posterior, loo = historical_prior()
    results = []
    for s_idx, scenario in enumerate(SCENARIOS):
        for m_idx, structure in enumerate(STRUCTURES):
            results.append(simulate(
                structure, scenario, posterior, n=500,
                seed=20261001 + 10007 * s_idx + 2081 * m_idx))

    for result in results:
        assert abs(sum(result["outcome_distribution"].values()) - 1.0) < 1e-10
        assert abs(result["mass_balance_error_thousands"]["median"]) < 1e-8
        assert abs(result["overseas_mass_balance_error_thousands"]["median"]) < 1e-8
        assert abs(result["northern_mass_balance_error_thousands"]["median"]) < 1e-8
        assert abs(result["mobilization_mass_balance_error_thousands"]["median"]) < 1e-8
        assert abs(result["civilian_population_balance_error_thousands"]["median"]) < 1e-8

    summary = {
        "model": "Operation Downfall V1.2 eight-year network, MFG, optimal-transport, game-MPC and stratified semi-Markov competing-risks model",
        "survival_model": {
            "family": "structure-stratified semi-Markov multi-state model with time-varying cause-specific Cox hazards",
            "transitions": [
                "decision-ready -> central surrender order",
                "decision-ready -> palace blocking -> delayed return to decision-ready",
                "central order -> coordinated surrender or fragmented surrender",
            ],
            "estimation_warning": "Transition coefficients are transparent counterfactual priors, not fitted event-history estimates.",
            "why_not_plain_cox": "Palace blocking is recurrent and nonterminal, so a one-event Cox model would remove paths from the risk set incorrectly.",
        },
        "warning": "Historical counterfactual conditional on model structures; not an objective forecast.",
        "historical_cases_role": "mechanism-prior calibration and leave-one-out validation only",
        "historical_cases": CASES,
        "posterior_parameter_summary": {
            "direct_combat_hazard": q(posterior[:, 0]),
            "isolation_hazard": q(posterior[:, 1]),
            "disease_hazard": q(posterior[:, 2]),
            "capture_propensity": q(posterior[:, 3]),
            "fortification_protection": q(posterior[:, 4]),
        },
        "leave_one_out_validation": loo,
        "structures": [s.name for s in STRUCTURES],
        "results": results,
        "ensemble": ensemble_summary(results),
    }
    (OUT / "没落行动V1.2_八年完整仿真摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    scenario_rows = []
    conditional_rows = []
    for r in results:
        u = r["unconditional"]
        scenario_rows.append({
            "scenario": r["scenario"], "structure": r["structure"],
            **r["outcome_distribution"],
            "allied_kia_median_thousands": u["allied_kia_thousands"]["median"],
            "allied_total_casualties_median_thousands": u["allied_total_casualties_thousands"]["median"],
            "japanese_home_dead_median_thousands": u["japanese_home_dead_thousands"]["median"],
            "japanese_military_total_dead_median_thousands": u["japanese_military_total_dead_thousands"]["median"],
            "japanese_total_dead_median_thousands": u["japanese_total_dead_thousands"]["median"],
            "death_ratio_median": u["japanese_to_allied_death_ratio"]["median"],
            "civilian_dead_median_thousands": u["japanese_civilian_dead_thousands"]["median"],
            "auxiliary_deployed_median_thousands": u["japanese_cumulative_auxiliary_deployed_thousands"]["median"],
            "nominal_pool_remaining_median_thousands": u["japanese_nominal_mobilization_pool_remaining_thousands"]["median"],
            "home_force_remaining_median_thousands": u["japanese_home_remaining_thousands"]["median"],
            "allied_reinforcement_median_thousands": u["allied_cumulative_reinforcement_thousands"]["median"],
            "coronet_launch_share": r["coronet_launch_share"],
            "surrender_order_share": r["central_surrender_order_share"],
            "surrender_order_week_median": r["central_surrender_order_week"]["median"],
            "atomic_strikes_median": r["atomic_strikes"]["median"],
        })
        for condition, data in r["conditional"].items():
            conditional_rows.append({
                "scenario": r["scenario"], "structure": r["structure"],
                "condition": condition, "paths": data["paths"], "share": data["share"],
                "allied_kia_median_thousands": data["allied_kia_thousands"]["median"],
                "allied_total_casualties_median_thousands": data["allied_total_casualties_thousands"]["median"],
                "japanese_home_dead_median_thousands": data["japanese_home_dead_thousands"]["median"],
                "japanese_total_dead_median_thousands": data["japanese_total_dead_thousands"]["median"],
                "death_ratio_median": data["japanese_to_allied_death_ratio"]["median"],
                "civilian_dead_median_thousands": data["japanese_civilian_dead_thousands"]["median"],
            })
    write_csv(OUT / "没落行动V1.2_八年多模型情景结果.csv", scenario_rows)
    write_csv(OUT / "没落行动V1.2_八年条件化结果.csv", conditional_rows)
    write_csv(OUT / "没落行动V1.2_历史战例留一验证.csv", [{
        "held_out": x["held_out"],
        "observed_dead_thousands": x["observed_dead_thousands"],
        "predicted_dead_p01": x["predicted_dead_thousands"]["p01"],
        "predicted_dead_median": x["predicted_dead_thousands"]["median"],
        "predicted_dead_p99": x["predicted_dead_thousands"]["p99"],
        "dead_covered_by_p01_p99": x["dead_covered_by_p01_p99"],
        "captured_covered_by_p01_p99": x["captured_covered_by_p01_p99"],
    } for x in loo])

    print(json.dumps({
        "ensemble": summary["ensemble"],
        "leave_one_out": loo,
        "verification": "PASS",
    }, ensure_ascii=False, indent=2))
    print("OPERATION_DOWNFALL_V12_VERIFICATION: PASS")


if __name__ == "__main__":
    main()
