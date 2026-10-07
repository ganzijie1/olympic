import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

WEEKS = 78
OUTCOMES = [
    "coordinated_surrender",
    "fragmented_surrender",
    "conditional_settlement",
    "allied_pause",
    "humanitarian_collapse",
    "war_continues_78_weeks",
]


@dataclass(frozen=True)
class AlliedPolicy:
    name: str
    tempo: float
    queue_brake: float
    casualty_brake: float
    logistics_investment: float
    medical_investment: float
    blockade_pressure: float
    information_gain: float


@dataclass(frozen=True)
class JapanesePolicy:
    name: str
    concentration: float
    fortification: float
    special_attack_release: float
    reserve_release: float
    hardline_shift: float
    command_fragility: float


ALLIED_POLICIES = [
    AlliedPolicy("baseline_plan_feedback", 0.82, 0.28, 0.18, 0.48, 0.36, 0.74, 0.42),
    AlliedPolicy("logistics_first", 0.70, 0.72, 0.30, 0.78, 0.48, 0.72, 0.56),
    AlliedPolicy("casualty_limiter", 0.66, 0.62, 0.78, 0.62, 0.76, 0.68, 0.58),
    AlliedPolicy("adaptive_balanced", 0.76, 0.64, 0.58, 0.70, 0.64, 0.72, 0.72),
    AlliedPolicy("blockade_then_assault", 0.58, 0.58, 0.46, 0.68, 0.54, 0.92, 0.66),
]

JAPANESE_POLICIES = [
    JapanesePolicy("ketsugo_concentrated", 1.18, 1.16, 1.20, 1.08, 0.12, 0.06),
    JapanesePolicy("special_attack_conserved", 1.05, 1.08, 0.78, 0.98, 0.08, 0.04),
    JapanesePolicy("distributed_defense", 0.94, 1.02, 0.96, 1.12, 0.02, 0.08),
    JapanesePolicy("brittle_command", 1.02, 1.05, 1.05, 0.94, 0.05, 0.22),
]

SCENARIOS = {
    "evidence_center": {},
    "typhoon_and_sealift_shock": {"weather_shock": 0.34, "sealift_scale": 0.82},
    "medical_capacity_shortfall": {"medical_scale": 0.72},
    "command_information_fracture": {"observation_noise": 1.65, "command_noise": 1.55},
    "prolonged_blockade": {"landing_delay": 12, "blockade_scale": 1.20},
}

EVIDENCE = {
    "olympic_initial_assault_force": {"value": 582500, "class": "physical/planning", "confidence": "A", "source": "NHHC H-Gram 057"},
    "planned_total_us_force_kyushu": {"value": 766700, "class": "physical/planning", "confidence": "A", "source": "NHHC H-Gram 057"},
    "japanese_force_kyushu_august": {"value": 900000, "qualifier": "over", "class": "observed", "confidence": "A", "source": "NHHC H-Gram 057"},
    "planned_vessels": {"value": 3033, "class": "planning", "confidence": "A", "source": "US Navy official report"},
    "medical_estimate_60d_total_casualties": {"value": 176663, "class": "historical estimate", "confidence": "B", "source": "US Army Medical Department history"},
    "medical_estimate_120d_total_casualties": {"value": 394859, "class": "historical estimate", "confidence": "B", "source": "US Army Medical Department history"},
    "medical_estimate_120d_battle_casualties": {"value": 125935, "class": "historical estimate", "confidence": "B", "source": "US Army Medical Department history"},
    "okinawa_days": {"value": 91, "class": "observed", "confidence": "A"},
    "okinawa_us_combat_casualties": {"value": 49151, "class": "observed", "confidence": "A"},
    "okinawa_us_nonbattle_casualties": {"value": 26000, "qualifier": "over", "class": "observed", "confidence": "A"},
    "okinawa_japanese_military_dead": {"value": 110000, "qualifier": "over", "class": "observed", "confidence": "A"},
    "okinawa_civilian_dead": {"value": 82000, "qualifier": "over", "class": "observed", "confidence": "B"},
    "okinawa_ships_sunk": {"value": 36, "class": "observed", "confidence": "A"},
    "okinawa_ships_damaged": {"value": 368, "class": "observed", "confidence": "A"},
    "louise_typhoon_planning_delay_days": {"value": 45, "class": "post-event estimate", "confidence": "B"},
    "starvation_mines_laid": {"value": 12135, "class": "observed", "confidence": "A"},
    "starvation_ships_sunk_or_badly_damaged": {"value": 670, "class": "observed", "confidence": "B"},
    "starvation_tonnage_million": {"value": 1.25, "class": "observed", "confidence": "B"},
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def quantiles(x):
    return {
        "p01": float(np.quantile(x, 0.01)),
        "p10": float(np.quantile(x, 0.10)),
        "median": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "mean": float(np.mean(x)),
    }


def simulate(policy, opponent, scenario_name="evidence_center", n=12000, seed=19451101):
    cfg = {
        "weather_shock": 0.0,
        "sealift_scale": 1.0,
        "medical_scale": 1.0,
        "observation_noise": 1.0,
        "command_noise": 1.0,
        "landing_delay": 0,
        "blockade_scale": 1.0,
    }
    cfg.update(SCENARIOS[scenario_name])
    rng = np.random.default_rng(seed)

    active = np.ones(n, dtype=bool)
    outcome = np.full(n, -1, dtype=np.int8)
    end_week = np.full(n, WEEKS, dtype=np.int16)

    allied = 582.5 * rng.lognormal(0.0, 0.06, n)
    japanese = 900.0 * opponent.concentration * rng.lognormal(0.0, 0.11, n)
    allied_reserve = 184.2 * rng.lognormal(0.0, 0.08, n)
    japanese_reserve = 105.0 * rng.lognormal(0.0, 0.16, n)
    japanese_initial_personnel = japanese + japanese_reserve
    progress = np.zeros(n)
    sea_air_control = np.clip(rng.normal(0.82, 0.05, n), 0.60, 0.95)
    shipping_damage = np.zeros(n)
    special_stock = np.clip(rng.normal(1.0, 0.14, n), 0.60, 1.35)

    # Explicit serial queues: theater arrival -> shore unloading -> inland distribution.
    inbound_queue = np.zeros(n)
    shore_queue = np.zeros(n)
    road_queue = np.zeros(n)
    supply_stock = np.clip(rng.normal(2.7, 0.35, n), 1.6, 3.8)
    medical_queue = np.zeros(n)
    medical_overload_patient_weeks = np.zeros(n)
    delivered_total = np.zeros(n)

    food = np.clip(rng.normal(0.52, 0.06, n), 0.34, 0.68)
    industry = np.clip(rng.normal(0.34, 0.05, n), 0.18, 0.50)
    governance = np.clip(rng.normal(0.76, 0.05, n), 0.58, 0.88)
    latent_resolve = np.clip(rng.beta(5.2, 2.4, n) + opponent.hardline_shift, 0.18, 0.99)
    latent_command = np.clip(rng.normal(0.82, 0.07, n), 0.50, 0.96)
    emperor_mediation = np.clip(rng.beta(2.0, 3.4, n), 0.02, 0.92)
    allied_tolerance = np.clip(rng.beta(4.8, 2.4, n), 0.25, 0.97)

    # Beliefs are updated from delayed, noisy observations, not from latent truth.
    belief_resolve = np.full(n, 0.68)
    belief_command = np.full(n, 0.80)
    observed_progress = np.zeros(n)
    observed_supply = np.full(n, 0.75)
    observed_casualty_rate = np.zeros(n)

    allied_kia = np.zeros(n)
    allied_wia = np.zeros(n)
    allied_nonbattle = np.zeros(n)
    japanese_dead = np.zeros(n)
    japanese_wounded_incident = np.zeros(n)
    japanese_wounded_pool = np.zeros(n)
    japanese_isolated = np.zeros(n)
    japanese_captured = np.zeros(n)
    japanese_surrendered = np.zeros(n)
    japanese_direct_combat_dead = np.zeros(n)
    japanese_wound_dead = np.zeros(n)
    japanese_isolation_dead = np.zeros(n)
    japanese_disease_dead = np.zeros(n)
    fortification_integrity = np.clip(
        rng.normal(0.88 * opponent.fortification, 0.05, n), 0.62, 1.0)
    # Structural heterogeneity is sampled before the campaign and never tuned
    # against a target exchange ratio. It governs whether fortified formations
    # remain connected, whether wounded survive, and whether isolated units
    # surrender or continue until destroyed.
    defense_network_redundancy = np.clip(rng.beta(4.2, 2.8, n), 0.18, 0.94)
    medical_resilience = np.clip(rng.beta(2.6, 3.8, n), 0.08, 0.86)
    annihilation_doctrine = np.clip(
        0.45 * latent_resolve + 0.55 * rng.beta(5.0, 2.0, n), 0.25, 0.99)
    civilian_dead = np.zeros(n)
    bombing_dead = np.zeros(n)
    atomic_strikes = np.zeros(n)
    cumulative_reward = np.zeros(n)
    action_changes = np.zeros(n)
    last_tempo = np.full(n, policy.tempo)

    snapshots = []
    for week in range(WEEKS):
        alive = active.copy()
        if not np.any(alive):
            break

        invaded = week >= cfg["landing_delay"]
        weeks_since = max(0, week - cfg["landing_delay"])
        seasonal = 0.80 + 0.13 * np.sin((week - 7) * np.pi / 26.0)
        weather = np.clip(rng.normal(seasonal, 0.10, n), 0.38, 1.02)
        if scenario_name == "typhoon_and_sealift_shock" and 4 <= weeks_since <= 9:
            weather *= 1.0 - cfg["weather_shock"]

        # Partial observations and a conservative Bayesian-style filter.
        obs_scale = cfg["observation_noise"] * (1.05 - policy.information_gain)
        observed_progress = np.clip(
            0.62 * observed_progress + 0.38 * progress
            + rng.normal(0.0, 0.035 * obs_scale, n), 0.0, 1.0)
        observed_supply = np.clip(
            0.58 * observed_supply + 0.42 * np.tanh(supply_stock / 3.0)
            + rng.normal(0.0, 0.045 * obs_scale, n), 0.0, 1.0)
        resolve_signal = np.clip(latent_resolve + rng.normal(0.0, 0.14 * obs_scale, n), 0.0, 1.0)
        command_signal = np.clip(latent_command + rng.normal(0.0, 0.12 * obs_scale, n), 0.0, 1.0)
        belief_resolve = np.clip(0.84 * belief_resolve + 0.16 * resolve_signal, 0.0, 1.0)
        belief_command = np.clip(0.84 * belief_command + 0.16 * command_signal, 0.0, 1.0)

        queue_ratio = np.clip((inbound_queue + shore_queue + road_queue) / 110.0, 0.0, 2.5)
        casualty_signal = np.clip(observed_casualty_rate / 18.0, 0.0, 2.0)
        tempo = np.clip(
            policy.tempo
            + 0.16 * (observed_supply - 0.62)
            + 0.10 * (0.55 - belief_resolve)
            - policy.queue_brake * 0.20 * queue_ratio
            - policy.casualty_brake * 0.16 * casualty_signal,
            0.28, 0.96)
        if policy.name == "blockade_then_assault" and weeks_since < 10:
            tempo *= 0.58
        action_changes += (np.abs(tempo - last_tempo) > 0.035) & alive
        last_tempo = tempo

        blockade = policy.blockade_pressure * cfg["blockade_scale"]
        food[alive] = np.clip(
            food[alive] - 0.0058 * blockade * sea_air_control[alive]
            - 0.0025 * (1.0 - industry[alive]) + rng.normal(0, 0.003, alive.sum()), 0.0, 0.72)
        industry[alive] = np.clip(
            industry[alive] - 0.0045 * blockade * sea_air_control[alive]
            + rng.normal(0, 0.0025, alive.sum()), 0.0, 0.50)
        governance[alive] = np.clip(
            governance[alive] - 0.0022 * (1.0 - food[alive])
            - 0.0018 * (1.0 - latent_command[alive]), 0.0, 0.90)

        conventional_intensity = blockade * (0.76 + 0.24 * (1.0 - tempo))
        weekly_bombing_dead = 0.34 * conventional_intensity * rng.lognormal(0.0, 0.22, n)
        bombing_dead[alive] += weekly_bombing_dead[alive]
        civilian_dead[alive] += weekly_bombing_dead[alive]

        # Aggregate continuation of the 1945 atomic-production and authorization process.
        strike_probability = 0.003 + 0.005 * (week > 12) + 0.006 * (allied_kia > 35.0)
        strike = alive & (rng.random(n) < strike_probability * (0.55 + 0.45 * belief_resolve))
        if np.any(strike):
            deaths = rng.lognormal(np.log(70.0), 0.30, strike.sum())
            atomic_strikes[strike] += 1
            civilian_dead[strike] += deaths
            industry[strike] *= 0.92
            governance[strike] *= 0.96
            rally = strike & (rng.random(n) < 0.23)
            latent_resolve[rally] = np.clip(latent_resolve[rally] + 0.04, 0.0, 1.0)

        if invaded:
            attack_window = np.exp(-weeks_since / 8.0)
            release = np.clip(
                opponent.special_attack_release * attack_window
                * (0.76 + 0.38 * tempo), 0.0, 1.4)
            special_effect = (0.055 * special_stock * release
                              * (1.10 - sea_air_control)
                              * rng.lognormal(0.0, 0.30, n))
            shipping_damage[alive] = np.clip(
                shipping_damage[alive] + special_effect[alive]
                - 0.012 * policy.logistics_investment, 0.0, 0.68)
            special_stock[alive] = np.maximum(0.0, special_stock[alive] - 0.065 * release[alive])

            arrival = (18.5 * cfg["sealift_scale"] * weather
                       * (1.0 - shipping_damage)
                       * (0.76 + 0.32 * policy.logistics_investment))
            inbound_queue[alive] += arrival[alive]
            unload_capacity = (15.0 * weather * (1.0 - 0.52 * shipping_damage)
                               * (0.72 + 0.42 * policy.logistics_investment))
            unloaded = np.minimum(inbound_queue, unload_capacity)
            inbound_queue -= unloaded
            shore_queue += unloaded

            road_capacity = (13.2 * (0.76 + 0.34 * policy.logistics_investment)
                             * (0.94 - 0.18 * progress))
            inland = np.minimum(shore_queue, road_capacity)
            shore_queue -= inland
            road_queue += inland
            distribution_capacity = 11.8 * (0.72 + 0.42 * policy.logistics_investment)
            delivered = np.minimum(road_queue, distribution_capacity)
            road_queue -= delivered
            supply_stock += delivered / 42.0
            delivered_total += delivered

            demand = (0.22 + 0.00078 * allied) * (0.66 + 0.48 * tempo)
            supply_stock = np.clip(supply_stock - demand, 0.0, 5.0)
            logistics = np.clip(
                0.30 + 0.52 * np.tanh(supply_stock / 2.1)
                - 0.20 * np.tanh(queue_ratio) - 0.24 * shipping_damage, 0.10, 0.98)

            if weeks_since < 14:
                reinforcement = np.minimum(allied_reserve, 10.8 * logistics)
                allied += reinforcement
                allied_reserve -= reinforcement
            if weeks_since < 10:
                j_reinforcement = np.minimum(
                    japanese_reserve, 8.0 * opponent.reserve_release
                    * (0.46 + 0.54 * food))
                japanese += j_reinforcement
                japanese_reserve -= j_reinforcement

            japanese_logistics = np.clip(
                0.10 + 0.52 * food + 0.20 * industry - 0.13 * progress, 0.06, 0.72)
            a_effective = allied * logistics * (0.84 + 0.36 * sea_air_control) * tempo
            isolated_combat_effect = (
                0.24 * japanese_isolated * latent_resolve * np.sqrt(food))
            j_effective = (japanese + isolated_combat_effect
                           + 82.0 * np.sqrt(food * industry) * latent_command)
            j_effective *= japanese_logistics * (0.72 + 0.38 * fortification_integrity)
            ratio = np.clip(a_effective / np.maximum(j_effective, 1.0), 0.12, 5.0)

            a_rate = (0.0100 * (0.72 + 0.38 * fortification_integrity)
                      * np.sqrt(np.maximum(j_effective, 1.0) / np.maximum(a_effective, 1.0))
                      * rng.lognormal(0.0, 0.20, n))
            j_rate = (0.0182 * (0.82 + 0.45 * sea_air_control) * tempo
                      * np.sqrt(np.maximum(a_effective, 1.0) / np.maximum(j_effective, 1.0))
                      * rng.lognormal(0.0, 0.19, n))
            a_loss = np.minimum(allied, allied * a_rate * active)
            j_loss = np.minimum(japanese, japanese * j_rate * active)
            allied -= a_loss
            japanese -= j_loss

            wounded = 0.77 * a_loss + 32.0 * special_effect
            immediate_kia = 0.23 * a_loss + 8.0 * special_effect
            medical_queue += wounded
            medical_capacity = (15.4 * cfg["medical_scale"]
                                * (0.70 + 0.44 * policy.medical_investment)
                                * (0.78 + 0.22 * logistics))
            treated = np.minimum(medical_queue, medical_capacity)
            medical_queue -= treated
            overflow = np.maximum(0.0, medical_queue - 1.8 * medical_capacity)
            medical_overload_patient_weeks += overflow
            overflow_deaths = 0.055 * overflow
            allied_kia += immediate_kia + overflow_deaths
            allied_wia += np.maximum(0.0, wounded - overflow_deaths)
            allied_nonbattle += (0.62 + 0.82 * (1.0 - logistics)
                                 + 0.44 * queue_ratio) * rng.lognormal(0.0, 0.16, n)
            # Japanese losses are resolved through explicit stocks and flows.
            # Direct combat loss first creates deaths, wounded, and a small
            # capture flow; wounded and isolated formations then evolve under
            # their own medical, food, command, and surrender mechanisms.
            direct_death_share = np.clip(
                0.34 + 0.16 * sea_air_control + 0.10 * tempo
                + 0.08 * (1.0 - japanese_logistics), 0.38, 0.68)
            direct_capture_share = np.clip(
                0.008 + 0.045 * progress * (1.0 - latent_resolve)
                + 0.025 * (1.0 - latent_command), 0.006, 0.08)
            direct_wounded_share = np.maximum(
                0.0, 1.0 - direct_death_share - direct_capture_share)
            direct_dead = direct_death_share * j_loss
            direct_captured = direct_capture_share * j_loss
            new_wounded = direct_wounded_share * j_loss
            japanese_direct_combat_dead += direct_dead
            japanese_dead += direct_dead
            japanese_captured += direct_captured
            japanese_wounded_incident += new_wounded
            japanese_wounded_pool += new_wounded

            # Strong fortifications reduce immediate exposure but create pockets
            # that can become cut off as transport and command fail.
            isolation_rate = np.clip(
                0.002 + 0.012 * progress + 0.016 * (1.0 - japanese_logistics)
                + 0.010 * (1.0 - latent_command), 0.002, 0.035)
            isolation_rate *= 1.25 - 0.55 * defense_network_redundancy
            newly_isolated = np.minimum(
                japanese, japanese * isolation_rate * fortification_integrity
                * (0.72 + 0.36 * tempo) * active)
            japanese -= newly_isolated
            japanese_isolated += newly_isolated

            isolated_death_rate = np.clip(
                0.010 + 0.040 * (1.0 - food)
                + 0.025 * sea_air_control * tempo
                + 0.025 * annihilation_doctrine, 0.01, 0.10)
            isolated_capture_rate = np.clip(
                0.003 + 0.030 * (1.0 - latent_resolve)
                + 0.020 * progress + 0.018 * (1.0 - latent_command), 0.003, 0.07)
            isolated_capture_rate *= 1.15 - 0.75 * annihilation_doctrine
            isolated_return_rate = np.clip(
                0.025 * japanese_logistics * latent_command
                * (1.0 - progress), 0.0, 0.025)
            isolated_dead = japanese_isolated * isolated_death_rate * active
            isolated_captured = japanese_isolated * isolated_capture_rate * active
            isolated_returned = japanese_isolated * isolated_return_rate * active
            isolated_total = isolated_dead + isolated_captured + isolated_returned
            isolated_scale = np.minimum(1.0, japanese_isolated / np.maximum(isolated_total, 1e-9))
            isolated_dead *= isolated_scale
            isolated_captured *= isolated_scale
            isolated_returned *= isolated_scale
            japanese_isolated -= isolated_dead + isolated_captured + isolated_returned
            japanese += isolated_returned
            japanese_isolation_dead += isolated_dead
            japanese_dead += isolated_dead
            japanese_captured += isolated_captured

            japanese_medical_capacity = np.clip(
                0.10 + 0.44 * japanese_logistics + 0.22 * food
                + 0.12 * latent_command + 0.16 * medical_resilience, 0.08, 0.82)
            wound_death_rate = np.clip(
                0.018 + 0.085 * (1.0 - japanese_medical_capacity)
                + 0.025 * (1.0 - food), 0.02, 0.12)
            wound_return_rate = np.clip(
                0.10 * japanese_medical_capacity * food, 0.005, 0.065)
            wound_capture_rate = np.clip(
                0.004 + 0.025 * progress * (1.0 - latent_command), 0.003, 0.03)
            wound_dead = japanese_wounded_pool * wound_death_rate * active
            wound_returned = japanese_wounded_pool * wound_return_rate * active
            wound_captured = japanese_wounded_pool * wound_capture_rate * active
            wound_total = wound_dead + wound_returned + wound_captured
            wound_scale = np.minimum(1.0, japanese_wounded_pool / np.maximum(wound_total, 1e-9))
            wound_dead *= wound_scale
            wound_returned *= wound_scale
            wound_captured *= wound_scale
            japanese_wounded_pool -= wound_dead + wound_returned + wound_captured
            japanese += wound_returned
            japanese_wound_dead += wound_dead
            japanese_dead += wound_dead
            japanese_captured += wound_captured

            disease_rate = np.clip(
                0.0006 + 0.0065 * (1.0 - food) ** 2
                + 0.0035 * (1.0 - japanese_logistics), 0.0005, 0.012)
            disease_loss = np.minimum(japanese, japanese * disease_rate * active)
            japanese -= disease_loss
            disease_dead = disease_loss * np.clip(0.48 + 0.30 * (1.0 - food), 0.48, 0.72)
            japanese_disease_dead += disease_dead
            japanese_dead += disease_dead
            japanese_wounded_pool += disease_loss - disease_dead
            japanese_wounded_incident += disease_loss - disease_dead

            fortification_integrity[alive] = np.clip(
                fortification_integrity[alive]
                - 0.007 * sea_air_control[alive] * tempo[alive]
                - 0.004 * progress[alive]
                - 0.003 * (1.0 - japanese_logistics[alive]), 0.08, 1.0)
            civilian_dead += (0.42 + 1.25 * tempo * (progress > 0.03)
                              + 0.92 * (1.0 - food)) * rng.lognormal(0.0, 0.18, n)

            delta = (0.022 + 0.028 * np.tanh(np.log(ratio))
                     + 0.016 * logistics - 0.006 * (opponent.fortification - 1.0))
            # A weekly disadvantage can stall a lodgment or cause a bounded local
            # setback, but it cannot erase the entire accumulated campaign state.
            delta = np.maximum(delta, -0.001)
            progress = np.clip(progress + delta * rng.lognormal(0.0, 0.16, n) * active, 0.0, 1.0)
            observed_casualty_rate = 0.62 * observed_casualty_rate + 0.38 * (a_loss + special_effect * 6.0)
        else:
            logistics = np.full(n, 0.0)
            allied_nonbattle += 0.12 * rng.lognormal(0.0, 0.12, n) * active
            civilian_dead += (0.38 + 1.18 * (1.0 - food)) * rng.lognormal(0.0, 0.16, n) * active

        latent_resolve[alive] = np.clip(
            latent_resolve[alive] + 0.010 * (0.62 - progress[alive])
            - 0.009 * (1.0 - food[alive]) - 0.006 * atomic_strikes[alive]
            + rng.normal(0.0, 0.008, alive.sum()), 0.02, 0.99)
        latent_command[alive] = np.clip(
            latent_command[alive] - opponent.command_fragility * 0.012
            - 0.005 * (1.0 - governance[alive])
            + rng.normal(0.0, 0.004 * cfg["command_noise"], alive.sum()), 0.02, 0.98)

        total_allied = allied_kia + allied_wia + allied_nonbattle
        if week >= 4:
            surrender_hazard = 0.082 * sigmoid(
                -3.7 + 1.45 * progress + 1.20 * (1.0 - food)
                + 0.82 * (1.0 - industry) + 0.70 * atomic_strikes
                + 0.90 * emperor_mediation - 1.55 * latent_resolve)
            compliance = sigmoid(
                -0.25 + 2.25 * latent_command + 1.00 * emperor_mediation
                - 0.90 * latent_resolve)
            conditional = 0.010 * sigmoid(
                -1.2 + 1.30 * (1.0 - food) + 0.75 * (1.0 - allied_tolerance)
                - 0.60 * progress)
            pause = 0.027 * sigmoid(
                -2.2 + total_allied / 125.0 + 1.35 * shipping_damage
                + 0.80 * medical_queue / 25.0 - 1.10 * progress - 0.80 * allied_tolerance)
            collapse = 0.030 * sigmoid(
                -4.0 + 4.2 * (0.20 - food) + 2.4 * (0.35 - governance))
            risks = np.column_stack([
                surrender_hazard * compliance,
                surrender_hazard * (1.0 - compliance),
                conditional, pause, collapse,
            ])
            risks[~active] = 0.0
            risks /= np.maximum(1.0, risks.sum(axis=1) / 0.20)[:, None]
            draw = rng.random(n)
            cumulative = np.cumsum(risks, axis=1)
            events = draw[:, None] < cumulative
            happened = active & events.any(axis=1)
            chosen = np.argmax(events, axis=1)
            outcome[happened] = chosen[happened]
            end_week[happened] = week + 1

            # Political termination reallocates surviving personnel; it does not
            # retroactively turn them into battlefield deaths. Coordinated
            # surrender disarms nearly all surviving stocks, while fragmented
            # surrender leaves a residual outside the terminal accounting window.
            surrender_fraction = np.zeros(n)
            surrender_fraction[happened & (chosen == 0)] = 0.98
            surrender_fraction[happened & (chosen == 1)] = np.clip(
                0.48 + 0.38 * compliance[happened & (chosen == 1)], 0.48, 0.84)
            surrender_fraction[happened & (chosen == 2)] = 0.76
            surrender_fraction[happened & (chosen == 4)] = 0.12
            for stock in (japanese, japanese_reserve, japanese_isolated, japanese_wounded_pool):
                disarmed = stock * surrender_fraction
                japanese_surrendered += disarmed
                stock -= disarmed
            active[happened] = False

        weekly_reward = (
            1.8 * progress - 0.010 * (allied_kia + allied_wia)
            - 0.004 * civilian_dead - 0.012 * medical_overload_patient_weeks
            - 0.20 * np.clip(queue_ratio - 1.0, 0.0, None))
        cumulative_reward[alive] += weekly_reward[alive]

        if week in (0, 7, 12, 16, 25, 38, 51, 77):
            snapshots.append({
                "week": week + 1,
                "active_share": float(active.mean()),
                "progress_median": float(np.median(progress)),
                "allied_total_casualties_median_thousands": float(np.median(total_allied)),
                "japanese_military_dead_median_thousands": float(np.median(japanese_dead)),
                "inbound_queue_median": float(np.median(inbound_queue)),
                "shore_queue_median": float(np.median(shore_queue)),
                "medical_queue_median": float(np.median(medical_queue)),
                "food_median": float(np.median(food)),
                "latent_resolve_median": float(np.median(latent_resolve)),
                "belief_resolve_median": float(np.median(belief_resolve)),
            })

    outcome[active] = 5
    total_casualties = allied_kia + allied_wia + allied_nonbattle
    japanese_accounted = (
        japanese + japanese_reserve + japanese_isolated + japanese_wounded_pool
        + japanese_dead + japanese_captured + japanese_surrendered)
    japanese_mass_balance_error = japanese_accounted - japanese_initial_personnel
    assert float(np.max(np.abs(japanese_mass_balance_error))) < 1e-6
    unacceptable = ((medical_overload_patient_weeks > 120.0) | (civilian_dead > 550.0)
                    | (outcome == 4)).astype(float)
    reward_tail = np.quantile(cumulative_reward, 0.10)
    return {
        "policy": policy.name,
        "opponent": opponent.name,
        "scenario": scenario_name,
        "trajectories": n,
        "outcome_distribution": {
            OUTCOMES[i]: float(np.mean(outcome == i)) for i in range(len(OUTCOMES))
        },
        "allied_kia_thousands": quantiles(allied_kia),
        "allied_total_casualties_thousands": quantiles(total_casualties),
        "japanese_military_dead_thousands": quantiles(japanese_dead),
        "japanese_wounded_incidents_thousands": quantiles(japanese_wounded_incident),
        "japanese_military_captured_thousands": quantiles(japanese_captured),
        "japanese_military_surrendered_thousands": quantiles(japanese_surrendered),
        "japanese_isolated_remaining_thousands": quantiles(japanese_isolated),
        "japanese_wounded_pool_remaining_thousands": quantiles(japanese_wounded_pool),
        "japanese_ready_remaining_thousands": quantiles(japanese + japanese_reserve),
        "japanese_irrecoverable_loss_thousands": quantiles(
            japanese_dead + japanese_captured + japanese_wounded_pool),
        "japanese_death_decomposition": {
            "direct_combat": quantiles(japanese_direct_combat_dead),
            "died_of_wounds": quantiles(japanese_wound_dead),
            "isolated_pockets": quantiles(japanese_isolation_dead),
            "disease_and_starvation": quantiles(japanese_disease_dead),
        },
        "japanese_mass_balance_error_thousands": quantiles(japanese_mass_balance_error),
        "japanese_to_allied_death_ratio": quantiles(
            japanese_dead / np.maximum(allied_kia, 0.5)),
        "japanese_dead_per_allied_total_casualty": quantiles(
            japanese_dead / np.maximum(total_casualties, 0.5)),
        "japanese_civilian_dead_thousands": quantiles(civilian_dead),
        "terminal_week": quantiles(end_week.astype(float)),
        "progress": quantiles(progress),
        "medical_overload_patient_weeks_thousands": quantiles(medical_overload_patient_weeks),
        "delivered_supply_index": quantiles(delivered_total),
        "atomic_strikes": quantiles(atomic_strikes),
        "food_index": quantiles(food),
        "action_changes": quantiles(action_changes),
        "belief_error_resolve": quantiles(np.abs(belief_resolve - latent_resolve)),
        "unacceptable_outcome_rate": float(np.mean(unacceptable)),
        "mean_reward": float(np.mean(cumulative_reward)),
        "lower_tail_reward": float(reward_tail),
        "robust_utility": float(np.mean(cumulative_reward) + 0.65 * reward_tail
                                - 80.0 * np.mean(unacceptable)),
        "history": snapshots,
    }


def policy_search():
    matrix = []
    for p_idx, policy in enumerate(ALLIED_POLICIES):
        for o_idx, opponent in enumerate(JAPANESE_POLICIES):
            matrix.append(simulate(
                policy, opponent, n=5500,
                seed=19451101 + 7919 * p_idx + 1543 * o_idx))
    worst_case = {}
    for policy in ALLIED_POLICIES:
        rows = [x for x in matrix if x["policy"] == policy.name]
        worst_case[policy.name] = min(x["robust_utility"] for x in rows)
    selected_name = max(worst_case, key=worst_case.get)
    return matrix, worst_case, selected_name


def historical_validation():
    # Validation targets are held out from policy scoring. The checks concern
    # scale and direction, because Olympic itself has no observed outcome.
    return {
        "okinawa_observed": {
            "duration_days": 91,
            "us_combat_casualties": 49151,
            "us_nonbattle_casualties": ">26000",
            "japanese_military_dead": ">110000",
            "civilian_dead": ">82000",
            "ships_sunk": 36,
            "ships_damaged": 368,
        },
        "iwo_jima_observed": {
            "japanese_defenders": 21000,
            "japanese_taken_prisoner_during_battle": 216,
            "us_killed": 6871,
            "us_wounded": 19217,
        },
        "luzon_observed": {
            "japanese_killed": 192000,
            "japanese_captured": 9700,
            "us_killed": 7933,
            "us_total_casualties": 40565,
        },
        "use_boundary": (
            "External post-run comparators only; none is a target exchange ratio "
            "or an optimization objective for the counterfactual Kyushu model."
        ),
        "olympic_medical_planning_envelope": {
            "60_day_total_casualties": 176663,
            "120_day_total_casualties": 394859,
            "120_day_battle_casualties": 125935,
            "interpretation": "planning estimates with different denominators, not ground truth",
        },
        "mechanism_checks": [
            "greater fortification must not reduce Allied casualties ceteris paribus",
            "lower sealift must increase queues or reduce delivered supply",
            "lower medical capacity must increase overflow mortality",
            "higher observation noise must increase latent-state belief error",
            "blockade delay must reduce food before it can reduce assault casualties",
        ],
    }


def sensitivity(selected_policy):
    # Morris-style one-at-a-time screening over high-impact, low-certainty factors.
    factors = {
        "japanese_concentration": (0.90, 1.18),
        "fortification": (0.90, 1.18),
        "special_attack_release": (0.75, 1.22),
        "hardline_shift": (-0.04, 0.14),
        "command_fragility": (0.03, 0.23),
    }
    baseline_opponent = JAPANESE_POLICIES[0]
    rows = []
    for i, (factor, (low, high)) in enumerate(factors.items()):
        kwargs = asdict(baseline_opponent)
        kwargs[factor.replace("japanese_", "")] = low
        low_opp = JapanesePolicy(**kwargs)
        kwargs = asdict(baseline_opponent)
        kwargs[factor.replace("japanese_", "")] = high
        high_opp = JapanesePolicy(**kwargs)
        low_result = simulate(selected_policy, low_opp, n=4500, seed=71237 + i * 101)
        high_result = simulate(selected_policy, high_opp, n=4500, seed=71237 + i * 101)
        rows.append({
            "factor": factor,
            "low": low,
            "high": high,
            "delta_allied_casualties_median_thousands": (
                high_result["allied_total_casualties_thousands"]["median"]
                - low_result["allied_total_casualties_thousands"]["median"]),
            "delta_coordinated_surrender_share": (
                high_result["outcome_distribution"]["coordinated_surrender"]
                - low_result["outcome_distribution"]["coordinated_surrender"]),
            "delta_unacceptable_rate": (
                high_result["unacceptable_outcome_rate"]
                - low_result["unacceptable_outcome_rate"]),
        })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    matrix, worst_case, selected_name = policy_search()
    selected = next(x for x in ALLIED_POLICIES if x.name == selected_name)

    evaluation = []
    for s_idx, scenario in enumerate(SCENARIOS):
        for o_idx, opponent in enumerate(JAPANESE_POLICIES):
            evaluation.append(simulate(
                selected, opponent, scenario_name=scenario, n=16000,
                seed=20261001 + s_idx * 10007 + o_idx * 2081))

    sensitivity_rows = sensitivity(selected)
    validation = historical_validation()

    # Directional regression checks required by the research plan.
    center = next(x for x in evaluation if x["scenario"] == "evidence_center"
                  and x["opponent"] == "ketsugo_concentrated")
    sealift = next(x for x in evaluation if x["scenario"] == "typhoon_and_sealift_shock"
                   and x["opponent"] == "ketsugo_concentrated")
    medical = next(x for x in evaluation if x["scenario"] == "medical_capacity_shortfall"
                   and x["opponent"] == "ketsugo_concentrated")
    info = next(x for x in evaluation if x["scenario"] == "command_information_fracture"
                and x["opponent"] == "ketsugo_concentrated")
    delay = next(x for x in evaluation if x["scenario"] == "prolonged_blockade"
                 and x["opponent"] == "ketsugo_concentrated")
    assert sealift["delivered_supply_index"]["median"] < center["delivered_supply_index"]["median"]
    assert (medical["medical_overload_patient_weeks_thousands"]["p90"]
            > center["medical_overload_patient_weeks_thousands"]["p90"])
    assert info["belief_error_resolve"]["median"] > center["belief_error_resolve"]["median"]
    assert delay["food_index"]["median"] < center["food_index"]["median"]
    for row in evaluation:
        assert abs(sum(row["outcome_distribution"].values()) - 1.0) < 1e-10

    summary = {
        "model": "Operation Downfall V2: evidence-graded partially observable queueing game",
        "interpretation": "conditional frequencies inside the model, not objective probabilities",
        "research_plan_adaptations": [
            "physical-observable, expert-judgment and unknowable parameter classes",
            "serial sealift, unloading, inland-distribution and medical queues",
            "latent resolve and command states with noisy delayed observations",
            "state-feedback policy library and adversary policy library",
            "risk-constrained maximin policy search rather than unvalidated deep RL",
            "historical mechanism validation, sensitivity screening and OOD shocks",
        ],
        "evidence": EVIDENCE,
        "selected_policy": selected_name,
        "selected_policy_parameters": asdict(selected),
        "policy_worst_case_utility": worst_case,
        "training_matrix": matrix,
        "evaluation": evaluation,
        "sensitivity": sensitivity_rows,
        "validation": validation,
    }
    (OUT / "没落行动V2_完整仿真摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    matrix_rows = []
    for row in matrix:
        matrix_rows.append({
            "policy": row["policy"],
            "opponent": row["opponent"],
            "robust_utility": row["robust_utility"],
            "allied_casualties_median_thousands": row["allied_total_casualties_thousands"]["median"],
            "coordinated_surrender": row["outcome_distribution"]["coordinated_surrender"],
            "allied_pause": row["outcome_distribution"]["allied_pause"],
            "unacceptable_outcome_rate": row["unacceptable_outcome_rate"],
        })
    write_csv(OUT / "没落行动V2_策略对手收益矩阵.csv", matrix_rows)

    eval_rows = []
    for row in evaluation:
        eval_rows.append({
            "scenario": row["scenario"],
            "opponent": row["opponent"],
            "selected_policy": row["policy"],
            "coordinated_surrender": row["outcome_distribution"]["coordinated_surrender"],
            "fragmented_surrender": row["outcome_distribution"]["fragmented_surrender"],
            "conditional_settlement": row["outcome_distribution"]["conditional_settlement"],
            "allied_pause": row["outcome_distribution"]["allied_pause"],
            "humanitarian_collapse": row["outcome_distribution"]["humanitarian_collapse"],
            "war_continues_78_weeks": row["outcome_distribution"]["war_continues_78_weeks"],
            "allied_kia_median_thousands": row["allied_kia_thousands"]["median"],
            "allied_total_casualties_median_thousands": row["allied_total_casualties_thousands"]["median"],
            "japanese_military_dead_median_thousands": row["japanese_military_dead_thousands"]["median"],
            "japanese_wounded_incidents_median_thousands": row[
                "japanese_wounded_incidents_thousands"]["median"],
            "japanese_military_captured_median_thousands": row[
                "japanese_military_captured_thousands"]["median"],
            "japanese_irrecoverable_loss_median_thousands": row[
                "japanese_irrecoverable_loss_thousands"]["median"],
            "japanese_to_allied_death_ratio_median": row[
                "japanese_to_allied_death_ratio"]["median"],
            "japanese_civilian_dead_median_thousands": row["japanese_civilian_dead_thousands"]["median"],
            "terminal_week_median": row["terminal_week"]["median"],
            "progress_median": row["progress"]["median"],
            "medical_overload_patient_weeks_median_thousands": row[
                "medical_overload_patient_weeks_thousands"]["median"],
            "belief_error_resolve_median": row["belief_error_resolve"]["median"],
            "unacceptable_outcome_rate": row["unacceptable_outcome_rate"],
        })
    write_csv(OUT / "没落行动V2_分布外情景结果.csv", eval_rows)
    write_csv(OUT / "没落行动V2_Morris敏感性.csv", sensitivity_rows)
    evidence_rows = []
    for key, item in EVIDENCE.items():
        evidence_rows.append({
            "parameter": key,
            "value": item.get("value"),
            "qualifier": item.get("qualifier", ""),
            "evidence_class": item.get("class", ""),
            "confidence": item.get("confidence", ""),
            "source": item.get("source", ""),
        })
    write_csv(OUT / "没落行动V2_证据与参数分级.csv", evidence_rows)

    print(json.dumps({
        "selected_policy": selected_name,
        "policy_worst_case_utility": worst_case,
        "center": center,
        "regression_checks": "PASS",
    }, ensure_ascii=False, indent=2))
    print("OPERATION_DOWNFALL_V2_VERIFICATION: PASS")


if __name__ == "__main__":
    main()
