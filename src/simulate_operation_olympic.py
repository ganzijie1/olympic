import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

N = 60_000
WEEKS = 78
FRONTS = ["Miyazaki", "Ariake", "Kushikino"]
REGIONS = [
    "Hokkaido", "Tohoku", "Kanto_Shinetu", "Tokai_Hokuriku",
    "Kinki", "Chugoku", "Shikoku", "Kyushu",
]
REGION_POPULATION = np.array([0.052, 0.103, 0.274, 0.133, 0.190, 0.091, 0.049, 0.108])
REGION_INDUSTRY = np.array([0.035, 0.068, 0.310, 0.176, 0.235, 0.074, 0.030, 0.072])
REGION_URBAN_EXPOSURE = np.array([0.035, 0.055, 0.315, 0.175, 0.250, 0.065, 0.025, 0.080])
OCCUPIED_REGIONS = ["China_Manchuria", "Taiwan", "Korea", "Southeast_Asia", "Pacific_Islands"]
OUTCOMES = [
    "coordinated_central_surrender",
    "central_surrender_with_fragmented_resistance",
    "conditional_settlement",
    "allied_operational_pause",
    "humanitarian_collapse",
    "coalition_war_continues_at_78_weeks",
]

SCENARIOS = {
    "olympic_planning_median": dict(
        invade_week=0, japanese_concentration=1.00, fortification=1.00,
        kamikaze=1.00, allied_adaptation=1.00, blockade=1.00,
        logistics_resilience=1.00, surrender_barrier=1.00,
        coalition_reinforcement=1.00, nuclear_authorization=0.18,
        total_war_mobilization=0.72, overseas_resistance=1.00,
    ),
    "ketsugo_high_concentration": dict(
        invade_week=0, japanese_concentration=1.18, fortification=1.16,
        kamikaze=1.28, allied_adaptation=0.94, blockade=1.00,
        logistics_resilience=1.05, surrender_barrier=1.18,
        coalition_reinforcement=1.00, nuclear_authorization=0.18,
        total_war_mobilization=1.00, overseas_resistance=1.12,
    ),
    "allied_sealift_disrupted": dict(
        invade_week=0, japanese_concentration=1.06, fortification=1.08,
        kamikaze=1.22, allied_adaptation=0.88, blockade=0.96,
        logistics_resilience=0.84, surrender_barrier=1.08,
        coalition_reinforcement=0.92, nuclear_authorization=0.18,
        total_war_mobilization=0.88, overseas_resistance=1.05,
    ),
    "allied_adaptive_superiority": dict(
        invade_week=0, japanese_concentration=0.98, fortification=0.96,
        kamikaze=0.92, allied_adaptation=1.18, blockade=1.12,
        logistics_resilience=1.13, surrender_barrier=0.94,
        coalition_reinforcement=1.22, nuclear_authorization=0.12,
        total_war_mobilization=0.68, overseas_resistance=0.92,
    ),
    "coalition_nuclear_escalation": dict(
        invade_week=0, japanese_concentration=1.08, fortification=1.08,
        kamikaze=1.08, allied_adaptation=1.12, blockade=1.18,
        logistics_resilience=1.08, surrender_barrier=1.08,
        coalition_reinforcement=1.28, nuclear_authorization=0.72,
        total_war_mobilization=0.92, overseas_resistance=1.00,
    ),
    "hundred_million_shattering_extreme": dict(
        invade_week=0, japanese_concentration=1.18, fortification=1.22,
        kamikaze=1.32, allied_adaptation=1.05, blockade=1.14,
        logistics_resilience=1.00, surrender_barrier=1.38,
        coalition_reinforcement=1.25, nuclear_authorization=0.62,
        total_war_mobilization=1.35, overseas_resistance=1.22,
    ),
    "blockade_delay_comparator": dict(
        invade_week=18, japanese_concentration=1.06, fortification=1.12,
        kamikaze=0.88, allied_adaptation=1.10, blockade=1.28,
        logistics_resilience=1.08, surrender_barrier=1.02,
        coalition_reinforcement=1.18, nuclear_authorization=0.30,
        total_war_mobilization=0.88, overseas_resistance=0.96,
    ),
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def q(values):
    return {
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "mean": float(np.mean(values)),
    }


def simulate(name, p, seed):
    rng = np.random.default_rng(seed)
    active = np.ones(N, dtype=bool)
    outcome = np.full(N, -1, dtype=np.int8)
    outcome_week = np.full(N, WEEKS, dtype=np.int16)

    # Thousands of personnel. The priors are centered on the documented planning totals.
    allied = np.tile(np.array([205.0, 185.0, 192.5]), (N, 1))
    allied *= rng.lognormal(0.0, 0.055, (N, 3))
    japanese = np.tile(np.array([292.0, 318.0, 290.0]), (N, 1))
    japanese *= p["japanese_concentration"] * rng.lognormal(0.0, 0.10, (N, 3))
    allied_reserve = np.full(N, 184.2) * rng.lognormal(0.0, 0.07, N)
    japanese_reserve = np.full(N, 105.0) * rng.lognormal(0.0, 0.18, N)

    progress = np.zeros((N, 3))
    sea_air_control = np.clip(rng.normal(0.82, 0.045, N), 0.62, 0.94)
    allied_logistics = np.clip(rng.normal(0.89, 0.04, N), 0.70, 0.98)
    japanese_logistics = np.clip(rng.normal(0.52, 0.07, N), 0.30, 0.72)
    shipping_damage = np.zeros(N)
    suicide_air_stock = np.clip(rng.normal(1.0, 0.14, N), 0.65, 1.35) * p["kamikaze"]
    suicide_sea_stock = np.clip(rng.normal(0.84, 0.16, N), 0.42, 1.28) * p["kamikaze"]
    combined_fleet_stock = np.clip(rng.normal(0.20, 0.045, N), 0.08, 0.32)
    naval_fuel = np.clip(rng.normal(0.16, 0.035, N), 0.07, 0.25)

    food = np.clip(rng.normal(0.52, 0.055, N), 0.36, 0.67)
    industry = np.clip(rng.normal(0.34, 0.05, N), 0.20, 0.50)
    governance = np.clip(rng.normal(0.76, 0.04, N), 0.62, 0.86)
    civilian_displacement = np.zeros(N)
    hardline = np.clip(rng.beta(5.2, 2.4, N) * p["surrender_barrier"], 0.25, 1.0)
    emperor_mediation = np.clip(rng.beta(2.0, 3.4, N), 0.02, 0.92)
    allied_tolerance = np.clip(rng.beta(4.8, 2.4, N), 0.28, 0.97)
    soviet_pressure = np.clip(rng.normal(0.64, 0.10, N), 0.35, 0.90)
    command_cohesion = np.clip(rng.normal(0.82, 0.06, N), 0.58, 0.94)
    mobilization_intent = np.clip(rng.normal(0.64, 0.08, N), 0.40, 0.84)
    effective_civilian_defense = np.zeros(N)

    # Overseas stocks in thousands. They are separate from Kyushu and cannot be
    # freely transferred through Allied sea and air control.
    china_manchuria_force = np.full(N, 2000.0) * rng.lognormal(0.0, 0.08, N)
    pacific_seasia_force = np.full(N, 1250.0) * rng.lognormal(0.0, 0.10, N)
    nationalist_pressure = np.clip(rng.normal(0.37, 0.06, N), 0.20, 0.54)
    communist_pressure = np.clip(rng.normal(0.31, 0.06, N), 0.16, 0.48)
    chinese_factional_friction = np.clip(rng.normal(0.42, 0.10, N), 0.16, 0.68)
    soviet_operational_pressure = np.clip(rng.normal(0.78, 0.07, N), 0.58, 0.94)
    commonwealth_pressure = np.clip(rng.normal(0.56, 0.06, N), 0.38, 0.72)

    base_damage = np.array([0.14, 0.24, 0.53, 0.48, 0.55, 0.41, 0.28, 0.39])
    regional_damage = np.clip(
        base_damage[None, :] + rng.normal(0.0, 0.045, (N, len(REGIONS))), 0.05, 0.72)
    pre_x_regional_damage = regional_damage.copy()
    regional_transport = np.clip(
        0.78 - 0.48 * regional_damage + rng.normal(0.0, 0.025, (N, len(REGIONS))), 0.24, 0.82)
    regional_food = np.clip(
        0.65 - 0.28 * regional_damage + rng.normal(0.0, 0.035, (N, len(REGIONS))), 0.25, 0.72)
    regional_governance = np.clip(
        0.86 - 0.22 * regional_damage + rng.normal(0.0, 0.025, (N, len(REGIONS))), 0.50, 0.88)
    occupied_infrastructure_damage = np.zeros((N, len(OCCUPIED_REGIONS)))
    denial_preparation = np.clip(
        np.array([0.42, 0.48, 0.46, 0.38, 0.34])[None, :]
        + rng.normal(0.0, 0.09, (N, len(OCCUPIED_REGIONS))), 0.10, 0.72)
    occupied_civilian_dead = np.zeros((N, len(OCCUPIED_REGIONS)))
    conventional_bombing_civilian_dead = np.zeros(N)

    # At X-day the model assumes Japan has already refused surrender. Weapons
    # potentially available later still require a fresh political authorization.
    nuclear_stock = rng.poisson(3.5, N).astype(float)
    nuclear_authorized = rng.random(N) < p["nuclear_authorization"]
    atomic_strikes = np.zeros(N)
    atomic_civilian_dead = np.zeros(N)
    urban_system_damage = np.clip(rng.normal(0.42, 0.05, N), 0.28, 0.57)
    coronet_launched = np.zeros(N, dtype=bool)
    coronet_progress = np.zeros(N)
    honshu_defenders = np.full(N, 1180.0) * rng.lognormal(0.0, 0.11, N)
    soviet_hokkaido_entry = np.zeros(N, dtype=bool)

    allied_kia = np.zeros(N)
    allied_wia = np.zeros(N)
    japanese_military_dead = np.zeros(N)
    japanese_civilian_dead = np.zeros(N)

    history = []
    for week in range(WEEKS):
        alive = active.copy()
        if not np.any(alive):
            break

        invaded = week >= p["invade_week"]
        coalition_growth = p["coalition_reinforcement"] * min(week / 30.0, 1.0)
        nationalist_pressure[alive] = np.clip(
            nationalist_pressure[alive] + 0.0038 * coalition_growth
            + rng.normal(0.0, 0.002, alive.sum()), 0.25, 0.92)
        communist_pressure[alive] = np.clip(
            communist_pressure[alive] + 0.0034 * coalition_growth
            + rng.normal(0.0, 0.002, alive.sum()), 0.18, 0.88)
        chinese_factional_friction[alive] = np.clip(
            chinese_factional_friction[alive] + rng.normal(0.0, 0.002, alive.sum()), 0.10, 0.78)
        soviet_operational_pressure[alive] = np.clip(
            soviet_operational_pressure[alive] + 0.0025 * coalition_growth
            + rng.normal(0.0, 0.0015, alive.sum()), 0.45, 0.98)
        commonwealth_pressure[alive] = np.clip(
            commonwealth_pressure[alive] + 0.0035 * coalition_growth
            + rng.normal(0.0, 0.0015, alive.sum()), 0.30, 0.90)

        # China/Manchuria and the remaining Pacific armies erode independently.
        # Only a negligible share can return to the home islands.
        combined_chinese_pressure = (
            nationalist_pressure + communist_pressure
            - 0.52 * chinese_factional_friction
            * nationalist_pressure * communist_pressure)
        china_loss_rate = (0.0046 * combined_chinese_pressure
                           + 0.0075 * soviet_operational_pressure
                           + 0.0020 * (1.0 - japanese_logistics))
        pacific_loss_rate = (0.0042 * commonwealth_pressure
                             + 0.0038 * sea_air_control
                             + 0.0026 * (1.0 - japanese_logistics))
        china_manchuria_force[alive] *= np.clip(
            1.0 - china_loss_rate[alive] * rng.lognormal(0.0, 0.12, alive.sum()), 0.94, 1.0)
        pacific_seasia_force[alive] *= np.clip(
            1.0 - pacific_loss_rate[alive] * rng.lognormal(0.0, 0.12, alive.sum()), 0.95, 1.0)

        loss_of_control = np.column_stack([
            1.0 - china_manchuria_force / 2000.0,
            np.clip(0.18 + 0.55 * sea_air_control + 0.15 * week / WEEKS, 0.0, 1.0),
            np.clip(0.20 + 0.65 * soviet_operational_pressure + 0.12 * week / WEEKS, 0.0, 1.0),
            1.0 - pacific_seasia_force / 1250.0,
            np.clip(0.38 + 0.58 * sea_air_control + 0.18 * week / WEEKS, 0.0, 1.0),
        ])
        # Documented surrender rules required intact preservation. Before a
        # surrender order, military denial and local unauthorized destruction
        # are modeled as separate, bounded risks; wholesale city destruction is
        # not assumed without a documentary anchor.
        military_denial = (0.0045 * denial_preparation * loss_of_control
                           * hardline[:, None])
        local_breakdown = (0.0020 * denial_preparation * loss_of_control
                           * (1.0 - command_cohesion[:, None]))
        damage_increment = (military_denial + local_breakdown) * rng.lognormal(
            0.0, 0.22, (N, len(OCCUPIED_REGIONS)))
        occupied_infrastructure_damage[alive] = np.clip(
            occupied_infrastructure_damage[alive] + damage_increment[alive], 0.0, 0.85)
        occupied_civilian_dead[alive] += (
            0.16 * damage_increment[alive]
            * np.array([4.2, 1.0, 1.8, 3.2, 0.5])[None, :]
            * rng.lognormal(0.0, 0.20, (alive.sum(), len(OCCUPIED_REGIONS))))

        # Continued production creates a stock, but use is a policy choice. No
        # target selection is modeled; effects are aggregate urban-system shocks.
        if week % 4 == 0:
            nuclear_stock[alive] += rng.poisson(0.95 + 0.035 * week, alive.sum())
        newly_authorized = alive & (~nuclear_authorized) & (
            rng.random(N) < 0.0035 * p["nuclear_authorization"])
        nuclear_authorized[newly_authorized] = True
        strike_now = (alive & nuclear_authorized & (nuclear_stock >= 1.0)
                      & (rng.random(N) < (0.060 + 0.020 * (week < 18))))
        if np.any(strike_now):
            strike_deaths = rng.lognormal(np.log(72.0), 0.30, strike_now.sum())
            strike_rows = np.flatnonzero(strike_now)
            strike_regions = rng.choice(len(REGIONS), size=strike_now.sum(), p=REGION_URBAN_EXPOSURE)
            atomic_strikes[strike_now] += 1.0
            nuclear_stock[strike_now] -= 1.0
            atomic_civilian_dead[strike_now] += strike_deaths
            japanese_civilian_dead[strike_now] += strike_deaths
            urban_system_damage[strike_now] = np.clip(
                urban_system_damage[strike_now] + rng.normal(0.055, 0.012, strike_now.sum()), 0.0, 1.0)
            industry[strike_now] = np.clip(industry[strike_now] - 0.035, 0.0, 0.50)
            governance[strike_now] = np.clip(governance[strike_now] - 0.018, 0.0, 0.90)
            command_cohesion[strike_now] = np.clip(command_cohesion[strike_now] - 0.012, 0.0, 1.0)
            regional_damage[strike_rows, strike_regions] = np.clip(
                regional_damage[strike_rows, strike_regions]
                + rng.normal(0.12, 0.025, strike_now.sum()), 0.0, 1.0)
            regional_governance[strike_rows, strike_regions] = np.clip(
                regional_governance[strike_rows, strike_regions] - 0.055, 0.0, 1.0)
            # A minority rally reaction raises hardline resolve temporarily.
            rally = strike_now & (rng.random(N) < 0.24)
            hardline[rally] = np.clip(hardline[rally] + 0.035, 0.0, 1.0)
        blockade_pressure = p["blockade"] * (0.64 + 0.36 * min(week / 20.0, 1.0))
        seasonal_weather = 0.78 + 0.15 * np.sin((week - 7) * np.pi / 26.0)
        weather = np.clip(rng.normal(seasonal_weather, 0.10, N), 0.42, 1.02)

        # Blockade and bombing affect flows and national endurance even before landing.
        japanese_logistics[alive] = np.clip(
            japanese_logistics[alive]
            * (1.0 - 0.010 * blockade_pressure * sea_air_control[alive])
            + rng.normal(0.0, 0.008, alive.sum()), 0.05, 0.75)
        food[alive] = np.clip(
            food[alive] - 0.0062 * blockade_pressure
            - 0.0030 * (1.0 - japanese_logistics[alive])
            + rng.normal(0.0, 0.0035, alive.sum()), 0.0, 0.75)
        industry[alive] = np.clip(
            industry[alive] - 0.0048 * blockade_pressure * sea_air_control[alive]
            + rng.normal(0.0, 0.003, alive.sum()), 0.0, 0.50)

        conventional_damage = (0.0019 * p["blockade"] * sea_air_control[:, None]
                               * REGION_URBAN_EXPOSURE[None, :] / REGION_URBAN_EXPOSURE.mean())
        realized_bombing_damage = conventional_damage[alive] * rng.lognormal(
            0.0, 0.15, (alive.sum(), len(REGIONS)))
        regional_damage[alive] = np.clip(
            regional_damage[alive] + realized_bombing_damage, 0.0, 1.0)
        conventional_bombing_civilian_dead[alive] += (
            170.0 * np.sum(realized_bombing_damage * REGION_POPULATION[None, :], axis=1)
            * rng.lognormal(0.0, 0.16, alive.sum()))
        japanese_civilian_dead[alive] += (
            170.0 * np.sum(realized_bombing_damage * REGION_POPULATION[None, :], axis=1))
        regional_transport[alive] = np.clip(
            regional_transport[alive] - 0.0035 * realized_bombing_damage
            / np.maximum(REGION_URBAN_EXPOSURE[None, :], 0.02)
            - 0.0018 * blockade_pressure, 0.05, 0.82)
        # Simplified interregional food network: surviving transport shares a
        # fraction of the national surplus, while damage and blockade reduce it.
        national_food_pool = np.sum(regional_food * REGION_POPULATION[None, :], axis=1)
        transport_mean = np.sum(regional_transport * REGION_POPULATION[None, :], axis=1)
        regional_food[alive] = np.clip(
            regional_food[alive]
            + 0.050 * transport_mean[alive, None]
            * (national_food_pool[alive, None] - regional_food[alive])
            - 0.0045 * blockade_pressure
            - 0.0035 * regional_damage[alive]
            + rng.normal(0.0, 0.0025, (alive.sum(), len(REGIONS))), 0.0, 0.78)
        regional_governance[alive] = np.clip(
            regional_governance[alive] - 0.0025 * regional_damage[alive]
            - 0.0020 * (1.0 - regional_food[alive])
            + rng.normal(0.0, 0.0015, (alive.sum(), len(REGIONS))), 0.0, 0.90)
        food[alive] = np.clip(
            0.55 * food[alive]
            + 0.45 * np.sum(regional_food[alive] * REGION_POPULATION[None, :], axis=1), 0.0, 0.75)
        governance[alive] = np.clip(
            0.58 * governance[alive]
            + 0.42 * np.sum(regional_governance[alive] * REGION_POPULATION[None, :], axis=1), 0.0, 0.90)
        industry[alive] = np.clip(
            industry[alive] - 0.0025
            * np.sum(realized_bombing_damage * REGION_INDUSTRY[None, :], axis=1), 0.0, 0.50)

        # "Hundred million shattered jewels" is an intention ceiling, not a
        # count of deployable troops. Equipment, calories and command determine
        # how much can be transformed into effective local defense.
        mobilization_intent[alive] = np.clip(
            mobilization_intent[alive] + 0.0035 * p["total_war_mobilization"]
            + 0.006 * strike_now[alive] - 0.006 * (1.0 - food[alive])
            + rng.normal(0.0, 0.002, alive.sum()), 0.0, 1.0)
        conversion = (mobilization_intent * p["total_war_mobilization"]
                      * np.sqrt(np.maximum(food, 0.0))
                      * np.sqrt(np.maximum(industry, 0.0) / 0.34)
                      * command_cohesion)
        effective_civilian_defense = np.clip(240.0 * conversion, 0.0, 310.0)

        if invaded:
            weeks_since = week - p["invade_week"]
            # Reinforcements arrive gradually and are limited by sealift damage.
            if weeks_since < 14:
                allied_reserve += 4.8 * p["coalition_reinforcement"]
                release = np.minimum(allied_reserve, 13.2 * allied_logistics
                                     * (1.0 + 0.16 * coalition_growth))
                allied += release[:, None] * np.array([0.36, 0.29, 0.35])[None, :]
                allied_reserve -= release
            if weeks_since < 10:
                j_release = np.minimum(japanese_reserve, 9.0 * japanese_logistics)
                japanese += j_release[:, None] * np.array([0.30, 0.37, 0.33])[None, :]
                japanese_reserve -= j_release

            # Concentrated special attacks damage shipping early, then decay as stocks are spent.
            attack_window = np.exp(-weeks_since / 7.5)
            allied_countermeasure = np.clip(
                0.74 + 0.19 * p["allied_adaptation"] * min(weeks_since / 12.0, 1.0), 0.68, 0.96)
            air_strike = (0.019 * suicide_air_stock * attack_window
                          * (1.10 - sea_air_control) * (1.18 - allied_countermeasure)
                          * rng.lognormal(0.0, 0.30, N))
            sea_strike = (0.014 * suicide_sea_stock * np.exp(-weeks_since / 10.0)
                          * (1.06 - sea_air_control) * (1.12 - allied_countermeasure)
                          * rng.lognormal(0.0, 0.32, N))
            fleet_sortie = (0.010 * combined_fleet_stock * naval_fuel
                            * np.exp(-weeks_since / 5.5) * (1.08 - sea_air_control)
                            * rng.lognormal(0.0, 0.28, N))
            strike = air_strike + sea_strike + fleet_sortie
            shipping_damage[alive] = np.clip(
                shipping_damage[alive] + strike[alive], 0.0, 0.62)
            suicide_air_stock[alive] = np.maximum(
                0.0, suicide_air_stock[alive] - 0.070 * attack_window)
            suicide_sea_stock[alive] = np.maximum(
                0.0, suicide_sea_stock[alive] - 0.052 * np.exp(-weeks_since / 10.0))
            combined_fleet_stock[alive] = np.maximum(
                0.0, combined_fleet_stock[alive] - 0.035 * np.exp(-weeks_since / 5.5))
            naval_fuel[alive] = np.maximum(
                0.0, naval_fuel[alive] - 0.018 * combined_fleet_stock[alive])
            allied_logistics[alive] = np.clip(
                p["logistics_resilience"] * (0.92 - shipping_damage[alive])
                * (0.72 + 0.28 * weather[alive]), 0.24, 0.98)
            sea_air_control[alive] = np.clip(
                sea_air_control[alive] + 0.006 * p["allied_adaptation"]
                - 0.012 * strike[alive] + rng.normal(0.0, 0.003, alive.sum()), 0.55, 0.98)

            fort = p["fortification"] * np.array([1.00, 1.14, 1.06])[None, :]
            a_eff = allied * allied_logistics[:, None] * (0.86 + 0.36 * sea_air_control[:, None])
            civil_front = effective_civilian_defense[:, None] * np.array([0.34, 0.36, 0.30])[None, :]
            j_eff = (japanese + civil_front) * japanese_logistics[:, None] * fort
            ratio = np.clip(a_eff / np.maximum(j_eff, 1.0), 0.15, 5.0)
            contested = (progress < 0.98) & (allied > 2.0) & (japanese > 2.0)

            a_rate = (0.0105 * fort * np.sqrt(np.maximum(j_eff, 1.0) / np.maximum(a_eff, 1.0))
                      * rng.lognormal(0.0, 0.18, (N, 3)))
            j_rate = (0.0190 * p["allied_adaptation"]
                      * (0.82 + 0.52 * sea_air_control[:, None])
                      * np.sqrt(np.maximum(a_eff, 1.0) / np.maximum(j_eff, 1.0))
                      * rng.lognormal(0.0, 0.18, (N, 3)))
            a_losses = np.minimum(allied, allied * a_rate * contested)
            j_losses = np.minimum(japanese, japanese * j_rate * contested)
            allied -= a_losses
            japanese -= j_losses
            allied_kia += 0.235 * a_losses.sum(axis=1) + 0.20 * strike * 6.0
            allied_wia += 0.765 * a_losses.sum(axis=1) + 0.80 * strike * 6.0
            japanese_military_dead += 0.62 * j_losses.sum(axis=1)
            mobilized_civil_dead = (0.0065 * effective_civilian_defense
                                    * np.mean(contested, axis=1)
                                    * rng.lognormal(0.0, 0.18, N))
            japanese_civilian_dead += mobilized_civil_dead

            delta = (0.020 + 0.031 * np.tanh(np.log(ratio))
                     + 0.014 * allied_logistics[:, None]
                     - 0.006 * (fort - 1.0))
            delta *= rng.lognormal(0.0, 0.14, (N, 3))
            progress = np.clip(progress + np.maximum(delta, -0.005) * contested, 0.0, 1.0)

            combat_exposure = np.mean((progress > 0.03) & (progress < 0.98), axis=1)
            civilian_displacement[alive] = np.clip(
                civilian_displacement[alive] + 0.015 * combat_exposure[alive]
                + rng.normal(0.0, 0.003, alive.sum()), 0.0, 1.0)
            regional_damage[:, 7] = np.clip(
                regional_damage[:, 7] + 0.006 * combat_exposure, 0.0, 1.0)
            regional_governance[:, 7] = np.clip(
                regional_governance[:, 7] - 0.004 * combat_exposure, 0.0, 1.0)
            civilian_weekly = (0.45 + 1.75 * combat_exposure + 1.10 * (1.0 - food)
                               + 0.55 * (1.0 - governance))
            japanese_civilian_dead += np.maximum(0.0, civilian_weekly * rng.lognormal(0.0, 0.20, N))
            governance[alive] = np.clip(
                governance[alive] - 0.006 * combat_exposure[alive]
                - 0.004 * (1.0 - food[alive]) + rng.normal(0, 0.0025, alive.sum()), 0.0, 0.90)
        else:
            # The comparator delays invasion; deaths arise from blockade, bombing, disease and scarcity.
            japanese_civilian_dead += np.maximum(
                0.0, (0.35 + 1.30 * (1.0 - food)) * rng.lognormal(0.0, 0.16, N))
            progress_mean = np.zeros(N)

        progress_mean = progress.mean(axis=1)
        # Follow-on Coronet is an aggregate campaign state, not a tactical plan.
        # It requires a Kyushu lodgment, coalition capacity, and a separate launch decision.
        coronet_eligible = (active & (week >= 17) & (progress_mean >= 0.86)
                            & (allied_logistics >= 0.52))
        coronet_start = coronet_eligible & (~coronet_launched) & (
            rng.random(N) < 0.055 * p["coalition_reinforcement"])
        coronet_launched[coronet_start] = True
        coronet_active = active & coronet_launched & (coronet_progress < 1.0)
        if np.any(coronet_active):
            c_ratio = np.clip(
                (520.0 + 7.5 * coalition_growth) * allied_logistics
                / np.maximum(honshu_defenders * japanese_logistics, 1.0), 0.15, 4.0)
            c_intensity = coronet_active * (1.0 - coronet_progress)
            c_allied_losses = (4.6 * c_intensity * p["fortification"]
                               / np.sqrt(np.maximum(c_ratio, 0.10))
                               * rng.lognormal(0.0, 0.20, N))
            c_japanese_losses = (10.8 * c_intensity * p["allied_adaptation"]
                                 * np.sqrt(np.maximum(c_ratio, 0.10))
                                 * rng.lognormal(0.0, 0.18, N))
            allied_kia += 0.24 * c_allied_losses
            allied_wia += 0.76 * c_allied_losses
            japanese_military_dead += 0.63 * c_japanese_losses
            honshu_defenders = np.maximum(0.0, honshu_defenders - c_japanese_losses)
            coronet_progress = np.clip(
                coronet_progress + c_intensity
                * (0.016 + 0.022 * np.tanh(np.log(np.maximum(c_ratio, 0.05))))
                * rng.lognormal(0.0, 0.15, N), 0.0, 1.0)
            regional_damage[:, 2] = np.clip(
                regional_damage[:, 2] + 0.007 * c_intensity, 0.0, 1.0)
            regional_governance[:, 2] = np.clip(
                regional_governance[:, 2] - 0.0045 * c_intensity, 0.0, 1.0)
            japanese_civilian_dead += (1.25 * c_intensity
                                       * rng.lognormal(0.0, 0.20, N))

        # Stalin requested a northern Hokkaido occupation zone, but Truman did
        # not approve it. Entry is therefore a low-probability prolonged-war branch.
        hokkaido_entry_now = (active & (week >= 16) & (~soviet_hokkaido_entry)
                              & (rng.random(N) < 0.0008 * p["coalition_reinforcement"]
                                 * soviet_operational_pressure))
        soviet_hokkaido_entry[hokkaido_entry_now] = True
        if np.any(soviet_hokkaido_entry & active):
            north_active = soviet_hokkaido_entry & active
            regional_damage[north_active, 0] = np.clip(
                regional_damage[north_active, 0] + 0.0045, 0.0, 1.0)
            regional_governance[north_active, 0] = np.clip(
                regional_governance[north_active, 0] - 0.0035, 0.0, 1.0)
            japanese_military_dead[north_active] += rng.lognormal(
                np.log(0.65), 0.20, north_active.sum())

        total_a_cas = allied_kia + allied_wia
        total_j_loss = japanese_military_dead / np.maximum(
            1.0, 0.62 * 1005.0 * p["japanese_concentration"])

        # Japan is conditioned not to surrender before X-day. Competing risks begin after four weeks.
        if week >= 4:
            surrender_score = (
                -5.25 + 1.65 * (1.0 - food) + 1.10 * (1.0 - industry)
                + 1.20 * soviet_pressure + 1.45 * progress_mean
                + 1.25 * total_j_loss + 0.95 * emperor_mediation
                + 0.22 * atomic_strikes + 0.55 * urban_system_damage
                + 0.90 * coronet_progress + 0.28 * soviet_hokkaido_entry
                + 0.40 * (1.0 - china_manchuria_force / 2000.0)
                + 0.30 * (1.0 - pacific_seasia_force / 1250.0)
                - 1.55 * hardline - 0.32 * mobilization_intent)
            surrender_hazard = 0.078 * sigmoid(surrender_score)
            conditional_hazard = 0.012 * sigmoid(
                -1.2 + 1.25 * (1.0 - food) + 0.75 * total_j_loss
                + 0.60 * (1.0 - allied_tolerance) - 0.75 * progress_mean)
            pause_hazard = np.zeros(N)
            if invaded:
                pause_hazard = 0.030 * sigmoid(
                    -2.0 + total_a_cas / 105.0 + 1.35 * shipping_damage
                    - 1.05 * progress_mean - 1.10 * allied_tolerance)
            collapse_hazard = 0.038 * sigmoid(
                -4.1 + 4.0 * (0.20 - food) + 2.2 * (0.34 - governance)
                + 1.1 * civilian_displacement)

            compliance = sigmoid(
                -0.35 + 2.0 * command_cohesion + 1.15 * emperor_mediation
                - 0.75 * hardline - 0.40 * p["overseas_resistance"])
            risks = np.column_stack([
                surrender_hazard * compliance,
                surrender_hazard * (1.0 - compliance),
                conditional_hazard, pause_hazard, collapse_hazard,
            ])
            risks[~active] = 0.0
            risk_sum = risks.sum(axis=1)
            scale = np.maximum(1.0, risk_sum / 0.22)
            risks /= scale[:, None]
            draw = rng.random(N)
            cumulative = np.cumsum(risks, axis=1)
            event = draw[:, None] < cumulative
            happened = active & event.any(axis=1)
            chosen = np.argmax(event, axis=1)
            outcome[happened] = chosen[happened]
            outcome_week[happened] = week + 1
            active[happened] = False

        if week in (0, 3, 7, 12, 25, 38, 51, 77):
            history.append({
                "week": week + 1,
                "active_share": float(active.mean()),
                "mean_control": float(progress_mean.mean()),
                "median_allied_casualties_thousands": float(np.median(total_a_cas)),
                "median_japanese_military_dead_thousands": float(np.median(japanese_military_dead)),
                "median_japanese_civilian_dead_thousands": float(np.median(japanese_civilian_dead)),
                "median_food_index": float(np.median(food)),
                "median_atomic_strikes": float(np.median(atomic_strikes)),
                "median_conventional_bombing_civilian_dead_thousands": float(
                    np.median(conventional_bombing_civilian_dead)),
                "median_china_manchuria_force_thousands": float(np.median(china_manchuria_force)),
                "median_pacific_seasia_force_thousands": float(np.median(pacific_seasia_force)),
                "median_effective_civilian_defense_thousands": float(np.median(effective_civilian_defense)),
                "median_suicide_air_stock": float(np.median(suicide_air_stock)),
                "median_suicide_sea_stock": float(np.median(suicide_sea_stock)),
                "median_combined_fleet_stock": float(np.median(combined_fleet_stock)),
                "coronet_launched_share": float(np.mean(coronet_launched)),
                "median_coronet_progress": float(np.median(coronet_progress)),
                "soviet_hokkaido_entry_share": float(np.mean(soviet_hokkaido_entry)),
                "median_nationalist_pressure": float(np.median(nationalist_pressure)),
                "median_communist_pressure": float(np.median(communist_pressure)),
                "median_regional_damage": {
                    REGIONS[i]: float(np.median(regional_damage[:, i])) for i in range(len(REGIONS))
                },
                "median_occupied_infrastructure_damage": {
                    OCCUPIED_REGIONS[i]: float(np.median(occupied_infrastructure_damage[:, i]))
                    for i in range(len(OCCUPIED_REGIONS))
                },
            })

    outcome[active] = 5
    distribution = {OUTCOMES[i]: float(np.mean(outcome == i)) for i in range(len(OUTCOMES))}
    secured = np.mean(progress >= 0.90, axis=1)
    result = {
        "scenario": name,
        "trajectories": N,
        "weeks": WEEKS,
        "condition": "Japan has not surrendered before the Olympic decision point",
        "outcome_distribution": distribution,
        "terminal_week": q(outcome_week.astype(float)),
        "allied_kia_thousands": q(allied_kia),
        "allied_total_casualties_thousands": q(allied_kia + allied_wia),
        "japanese_military_dead_thousands": q(japanese_military_dead),
        "japanese_civilian_dead_thousands": q(japanese_civilian_dead),
        "southern_kyushu_fronts_secured_share": q(secured),
        "food_index": q(food),
        "shipping_damage_index": q(shipping_damage),
        "suicide_air_stock_remaining": q(suicide_air_stock),
        "suicide_sea_stock_remaining": q(suicide_sea_stock),
        "combined_fleet_stock_remaining": q(combined_fleet_stock),
        "naval_fuel_remaining": q(naval_fuel),
        "additional_atomic_strikes": q(atomic_strikes),
        "atomic_civilian_dead_thousands": q(atomic_civilian_dead),
        "conventional_bombing_civilian_dead_thousands": q(conventional_bombing_civilian_dead),
        "effective_civilian_defense_thousands": q(effective_civilian_defense),
        "china_manchuria_force_remaining_thousands": q(china_manchuria_force),
        "pacific_seasia_force_remaining_thousands": q(pacific_seasia_force),
        "coronet_launched_share": float(np.mean(coronet_launched)),
        "coronet_progress": q(coronet_progress),
        "soviet_hokkaido_entry_share": float(np.mean(soviet_hokkaido_entry)),
        "nationalist_pressure": q(nationalist_pressure),
        "communist_pressure": q(communist_pressure),
        "chinese_factional_friction": q(chinese_factional_friction),
        "soviet_operational_pressure": q(soviet_operational_pressure),
        "commonwealth_pressure": q(commonwealth_pressure),
        "regional_damage_median": {
            REGIONS[i]: float(np.median(regional_damage[:, i])) for i in range(len(REGIONS))
        },
        "pre_x_regional_damage_median": {
            REGIONS[i]: float(np.median(pre_x_regional_damage[:, i])) for i in range(len(REGIONS))
        },
        "regional_transport_median": {
            REGIONS[i]: float(np.median(regional_transport[:, i])) for i in range(len(REGIONS))
        },
        "regional_food_median": {
            REGIONS[i]: float(np.median(regional_food[:, i])) for i in range(len(REGIONS))
        },
        "regional_governance_median": {
            REGIONS[i]: float(np.median(regional_governance[:, i])) for i in range(len(REGIONS))
        },
        "occupied_infrastructure_damage_median": {
            OCCUPIED_REGIONS[i]: float(np.median(occupied_infrastructure_damage[:, i]))
            for i in range(len(OCCUPIED_REGIONS))
        },
        "occupied_civilian_dead_thousands": {
            OCCUPIED_REGIONS[i]: q(occupied_civilian_dead[:, i]) for i in range(len(OCCUPIED_REGIONS))
        },
        "history": history,
    }
    return result


def main():
    results = []
    for index, (name, params) in enumerate(SCENARIOS.items()):
        results.append(simulate(name, params, 19451101 + 1009 * index))

    by_name = {row["scenario"]: row for row in results}
    median = by_name["olympic_planning_median"]
    hard = by_name["ketsugo_high_concentration"]
    disrupted = by_name["allied_sealift_disrupted"]
    adaptive = by_name["allied_adaptive_superiority"]
    nuclear = by_name["coalition_nuclear_escalation"]
    extreme = by_name["hundred_million_shattering_extreme"]
    delay = by_name["blockade_delay_comparator"]

    assert hard["allied_total_casualties_thousands"]["median"] > median["allied_total_casualties_thousands"]["median"]
    assert disrupted["outcome_distribution"]["allied_operational_pause"] > median["outcome_distribution"]["allied_operational_pause"]
    assert adaptive["southern_kyushu_fronts_secured_share"]["median"] >= median["southern_kyushu_fronts_secured_share"]["median"]
    assert nuclear["additional_atomic_strikes"]["median"] > median["additional_atomic_strikes"]["median"]
    assert extreme["effective_civilian_defense_thousands"]["median"] > median["effective_civilian_defense_thousands"]["median"]
    assert delay["food_index"]["median"] < median["food_index"]["median"]
    for result in results:
        assert abs(sum(result["outcome_distribution"].values()) - 1.0) < 1e-10

    summary = {
        "model": "stochastic semi-Markov campaign, logistics and national-endurance model",
        "warning": "Historical counterfactual stress test, not an observed forecast. Casualty outputs are conditional model ranges.",
        "historical_anchors": {
            "olympic_x_day": "1945-11-01",
            "planned_initial_assault_ground_force": 582500,
            "planned_total_us_force_on_kyushu": 766700,
            "japanese_strength_on_kyushu_by_surrender": "over 900000",
            "planned_allied_vessels": 3033,
            "overseas_japanese_forces": "several million across China, Manchuria, Korea, Southeast Asia and Pacific garrisons",
            "atomic_policy": "additional weapons require renewed presidential authorization",
            "defense_geography": "seven Ketsugo operational zones and eight regional civil governorates are modeled separately",
        },
        "results": results,
    }
    json_path = OUT / "奥林匹克行动反事实仿真摘要.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = OUT / "奥林匹克行动反事实情景结果.csv"
    fields = [
        "scenario", "coordinated_central_surrender", "central_surrender_with_fragmented_resistance",
        "conditional_settlement", "allied_operational_pause", "humanitarian_collapse",
        "coalition_war_continues_at_78_weeks",
        "allied_kia_p10_thousands", "allied_kia_median_thousands", "allied_kia_p90_thousands",
        "allied_casualties_p10_thousands", "allied_casualties_median_thousands", "allied_casualties_p90_thousands",
        "japanese_military_dead_median_thousands", "japanese_civilian_dead_median_thousands",
        "fronts_secured_median_share", "food_index_median", "additional_atomic_strikes_median",
        "atomic_civilian_dead_median_thousands", "conventional_bombing_civilian_dead_median_thousands",
        "effective_civilian_defense_median_thousands",
        "china_manchuria_force_remaining_median_thousands", "pacific_force_remaining_median_thousands",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            dist = result["outcome_distribution"]
            writer.writerow({
                "scenario": result["scenario"],
                **dist,
                "allied_kia_p10_thousands": result["allied_kia_thousands"]["p10"],
                "allied_kia_median_thousands": result["allied_kia_thousands"]["median"],
                "allied_kia_p90_thousands": result["allied_kia_thousands"]["p90"],
                "allied_casualties_p10_thousands": result["allied_total_casualties_thousands"]["p10"],
                "allied_casualties_median_thousands": result["allied_total_casualties_thousands"]["median"],
                "allied_casualties_p90_thousands": result["allied_total_casualties_thousands"]["p90"],
                "japanese_military_dead_median_thousands": result["japanese_military_dead_thousands"]["median"],
                "japanese_civilian_dead_median_thousands": result["japanese_civilian_dead_thousands"]["median"],
                "fronts_secured_median_share": result["southern_kyushu_fronts_secured_share"]["median"],
                "food_index_median": result["food_index"]["median"],
                "additional_atomic_strikes_median": result["additional_atomic_strikes"]["median"],
                "atomic_civilian_dead_median_thousands": result["atomic_civilian_dead_thousands"]["median"],
                "conventional_bombing_civilian_dead_median_thousands": result["conventional_bombing_civilian_dead_thousands"]["median"],
                "effective_civilian_defense_median_thousands": result["effective_civilian_defense_thousands"]["median"],
                "china_manchuria_force_remaining_median_thousands": result["china_manchuria_force_remaining_thousands"]["median"],
                "pacific_force_remaining_median_thousands": result["pacific_seasia_force_remaining_thousands"]["median"],
            })

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("OPERATION_OLYMPIC_COUNTERFACTUAL_VERIFICATION: PASS")


if __name__ == "__main__":
    main()
