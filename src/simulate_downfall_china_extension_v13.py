"""China-theater extension for the Operation Downfall V1.2 model.

This module does not reconstruct operational routes. It adds theater-level
stocks and flows for a historical counterfactual in which Japan rejects the
1945 surrender, while Chinese Nationalist and Communist forces remain the
principal ground forces in China and US/Soviet forces provide geographically
limited direct combat, sealift, air, logistics, and northern pressure.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SOURCE = OUT / "没落行动V1.2_八年完整仿真摘要.json"
WEEKS = 416
PATHS = 2500

THEATERS = ("manchuria", "north_china", "central_south_china", "korea", "taiwan")
MAINLAND = np.array([True, True, True, False, False])
TAIWAN = np.array([False, False, False, False, True])

# Thousand persons. The first three entries split the former 2.0 million
# China-and-Manchuria stock; Korea and Taiwan retain V1.2's separate anchors.
JAPANESE_INITIAL = np.array([700.0, 600.0, 700.0, 600.0, 320.0])

# Mobilizable field stocks, not national armed-forces totals (thousand persons).
KMT_INITIAL = np.array([40.0, 620.0, 1450.0, 0.0, 75.0])
CCP_INITIAL = np.array([240.0, 760.0, 360.0, 0.0, 0.0])
SOVIET_INITIAL = np.array([1500.0, 0.0, 0.0, 260.0, 0.0])
US_INITIAL = np.array([0.0, 65.0, 90.0, 55.0, 55.0])

# Japanese-sponsored Chinese forces.  The 1.2m center follows a 1945 US
# diplomatic estimate of roughly 0.3m in Manchuria and 0.9m in China proper.
# These are separate agents: they may fight, defect to either Chinese camp,
# disband/surrender, be wounded, or die.
COLLABORATOR_INITIAL = np.array([300.0, 390.0, 510.0, 0.0, 0.0])

# Civilian populations exposed to continued war (thousand persons). Mainland
# is deliberately below total Chinese population because interior regions are
# represented through displacement/famine spillovers rather than direct battle.
CHINESE_CIVILIAN_INITIAL = np.array([38_000.0, 105_000.0, 155_000.0, 0.0, 0.0])
TAIWAN_CIVILIAN_INITIAL = np.array([0.0, 0.0, 0.0, 0.0, 6_100.0])

STRUCTURE_SCALE = {
    "network_campaign": 1.00,
    "aggregate_lanchester": 1.18,
    "discrete_event_logistics": 0.96,
    "system_dynamics": 0.91,
    "differential_game": 0.94,
}

# The Chihchiang/West Hunan casualty records do not identify one common
# denominator.  The narrow Japanese postwar reconstruction reports about
# 6.5k Japanese battle casualties, broader Japanese accounts mix combat and
# illness, and Chinese operational claims report 28k--36k enemy casualties.
# We therefore use three observation regimes to widen the relative-loss prior
# instead of treating any one published ratio as ground truth.
CHIHCHIANG_REGIME_PROB = np.array([0.35, 0.40, 0.25])
CHIHCHIANG_RELATIVE_LOSS_CENTER = np.array([1.45, 1.00, 0.65])


def quantiles(x: np.ndarray) -> dict[str, float]:
    return {
        "p05": float(np.quantile(x, 0.05)),
        "median": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
        "mean": float(np.mean(x)),
    }


def sample_from_reported(rng: np.random.Generator, summary: dict, n: int,
                         floor: float = 0.0, cap: float | None = None) -> np.ndarray:
    """Approximate a reported distribution with a split normal around its median."""
    raw_med = summary.get("median")
    if raw_med is None:
        fallback = cap if cap is not None else floor
        return np.full(n, fallback, dtype=float)
    med = float(raw_med)
    raw_p10 = summary.get("p10", summary.get("p05", med))
    raw_p90 = summary.get("p90", summary.get("p95", med))
    p10 = med if raw_p10 is None else float(raw_p10)
    p90 = med if raw_p90 is None else float(raw_p90)
    z = rng.normal(size=n)
    left = max((med - p10) / 1.28155, 1e-6)
    right = max((p90 - med) / 1.28155, 1e-6)
    x = med + np.where(z < 0.0, z * left, z * right)
    if cap is None:
        return np.maximum(floor, x)
    return np.clip(x, floor, cap)


def simulate_cell(source: dict, seed: int) -> tuple[dict, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    n = PATHS
    structure = source["structure"]
    scenario = source["scenario"]
    scale = STRUCTURE_SCALE[structure]

    order_share = float(source["central_surrender_order_share"])
    order_week = sample_from_reported(
        rng, source["central_surrender_order_week"], n, floor=12.0, cap=WEEKS)
    has_order = rng.random(n) < order_share
    order_week = np.where(has_order, order_week, WEEKS + 1)

    j_initial = JAPANESE_INITIAL[None, :] * rng.lognormal(0.0, 0.10, (n, 5))
    japanese = j_initial.copy()
    j_dead = np.zeros_like(japanese)
    j_wounded_disabled = np.zeros_like(japanese)
    j_captured_surrendered = np.zeros_like(japanese)
    j_supply = np.clip(rng.normal([0.58, 0.62, 0.66, 0.60, 0.57], 0.07, (n, 5)), 0.30, 0.82)
    j_command = np.clip(rng.normal(0.78, 0.08, (n, 5)), 0.42, 0.95)

    kmt = KMT_INITIAL[None, :] * rng.lognormal(0.0, 0.13, (n, 5))
    ccp = CCP_INITIAL[None, :] * rng.lognormal(0.0, 0.14, (n, 5))
    soviet = SOVIET_INITIAL[None, :] * rng.lognormal(0.0, 0.08, (n, 5))
    us = US_INITIAL[None, :] * rng.lognormal(0.0, 0.10, (n, 5))
    collaborator_initial = (
        COLLABORATOR_INITIAL[None, :] * rng.lognormal(0.0, 0.16, (n, 5)))
    collaborator = collaborator_initial.copy()
    collaborator_dead = np.zeros(n)
    collaborator_wounded = np.zeros(n)
    collaborator_to_kmt = np.zeros(n)
    collaborator_to_ccp = np.zeros(n)
    collaborator_disbanded_captured = np.zeros(n)

    kmt_dead = np.zeros(n)
    ccp_dead = np.zeros(n)
    soviet_dead = np.zeros(n)
    us_dead = np.zeros(n)
    kmt_cas = np.zeros(n)
    ccp_cas = np.zeros(n)
    soviet_cas = np.zeros(n)
    us_cas = np.zeros(n)

    china_civ = CHINESE_CIVILIAN_INITIAL[None, :] * rng.lognormal(0.0, 0.035, (n, 5))
    taiwan_civ = TAIWAN_CIVILIAN_INITIAL[None, :] * rng.lognormal(0.0, 0.025, (n, 5))
    china_civ_dead = np.zeros(n)
    taiwan_civ_dead = np.zeros(n)
    china_civ_severe_nonfatal = np.zeros(n)
    taiwan_civ_severe_nonfatal = np.zeros(n)
    food = np.clip(rng.normal([0.58, 0.61, 0.62, 0.65, 0.69], 0.055, (n, 5)), 0.40, 0.82)
    displacement = np.zeros((n, 5))

    # Political friction reduces combined effectiveness without treating KMT
    # and CCP as fighting one another in this counterfactual.
    coordination = np.clip(rng.beta(4.0, 2.8, n), 0.28, 0.92)
    us_support = np.clip(rng.normal(0.68, 0.12, n), 0.30, 0.95)
    soviet_momentum = np.clip(rng.normal(0.91, 0.08, n), 0.62, 1.08)
    chihchiang_regime = rng.choice(3, size=n, p=CHIHCHIANG_REGIME_PROB)
    chinese_relative_loss = np.clip(
        CHIHCHIANG_RELATIVE_LOSS_CENTER[chihchiang_regime]
        * rng.lognormal(0.0, 0.18, n), 0.42, 2.05)
    # Latent attachment to the sponsor is only partially observable.  Wages,
    # coercion and Japanese local control sustain service; battlefield decline,
    # amnesty credibility and peer defections accelerate exit.
    collaborator_attachment = np.clip(rng.beta(2.2, 2.5, (n, 5)), 0.08, 0.92)
    amnesty_kmt = np.clip(rng.normal(0.72, 0.12, n), 0.30, 0.96)
    amnesty_ccp = np.clip(rng.normal(0.58, 0.15, n), 0.18, 0.94)
    taiwan_landing_week = np.clip(rng.normal(24.0, 10.0, n), 6.0, 60.0)
    coastal_entry_week = np.clip(rng.normal(10.0, 6.0, n), 1.0, 32.0)
    theater_resolution_week = np.full(n, WEEKS, dtype=float)
    previously_active = np.ones(n, dtype=bool)

    for week in range(WEEKS):
        active = (japanese + collaborator).sum(axis=1) > 1.0
        newly_resolved = previously_active & (~active)
        theater_resolution_week[newly_resolved] = week
        previously_active = active.copy()
        if not np.any(active):
            break

        # Gradual Chinese commitment and limited US reinforcement. US/Soviet
        # forces amplify logistics and selected fronts; they do not replace the
        # Chinese armies in the interior.
        commit = 1.0 - np.exp(-(week + 1.0) / 20.0)
        kmt[:, 0] += active * (0.55 + 0.65 * us_support) * np.exp(-week / 150.0)
        kmt[:, 1] += active * (1.25 + 1.00 * us_support) * np.exp(-week / 170.0)
        kmt[:, 2] += active * (1.80 + 1.15 * us_support) * np.exp(-week / 190.0)
        ccp[:, 0] += active * 0.65 * np.exp(-week / 170.0)
        ccp[:, 1] += active * 1.20 * np.exp(-week / 190.0)
        ccp[:, 2] += active * 0.55 * np.exp(-week / 180.0)
        us[:, 1] += active * (week >= coastal_entry_week) * 0.16 * np.exp(-week / 100.0)
        us[:, 2] += active * (week >= coastal_entry_week) * 0.24 * np.exp(-week / 110.0)
        us[:, 4] += active * (week >= taiwan_landing_week) * 0.13 * np.exp(-week / 75.0)

        # Theater activation: Soviet fronts begin immediately, Chinese-wide
        # counteroffensives ramp up, and Taiwan begins only after sealift forms.
        activated = np.column_stack([
            np.ones(n),
            np.minimum(1.0, 0.22 + week / 30.0) * (week >= coastal_entry_week),
            np.minimum(1.0, 0.16 + week / 38.0) * (week >= coastal_entry_week),
            np.ones(n),
            (week >= taiwan_landing_week).astype(float),
        ])

        combined_chinese = (kmt + ccp) * (0.52 + 0.48 * coordination[:, None])
        soviet_effective = soviet * soviet_momentum[:, None]
        us_effective = us * (0.45 + 0.55 * us_support[:, None])
        opponent = combined_chinese + soviet_effective + us_effective
        collaborator_effective = collaborator * (
            0.18 + 0.37 * collaborator_attachment) * j_command

        # Network isolation increases after Soviet penetration and coastal
        # entry. Supply cannot fall below zero and command decays with isolation.
        pressure_ratio = np.sqrt(
            (opponent + 30.0) / (japanese + 0.55 * collaborator_effective + 30.0))
        isolation = np.clip(
            0.10 + 0.38 * activated + 0.22 * (1.0 - j_supply)
            + np.array([0.18, 0.10, 0.08, 0.14, 0.12])[None, :], 0.0, 0.95)
        j_supply *= np.clip(1.0 - activated * (
            np.array([0.030, 0.014, 0.012, 0.020, 0.017])[None, :]
            * (0.65 + 0.35 * isolation)), 0.88, 1.0)
        j_command *= np.clip(1.0 - activated * (0.004 + 0.010 * isolation), 0.94, 1.0)

        base = np.array([0.020, 0.0055, 0.0047, 0.010, 0.0105])[None, :]
        man_korea_decay = np.array([
            0.35 + 0.65 * np.exp(-week / 9.0), 1.0, 1.0,
            0.45 + 0.55 * np.exp(-week / 12.0), 1.0,
        ])[None, :]
        hazard = np.clip(
            scale * base * activated * man_korea_decay * pressure_ratio
            * (0.78 + 0.55 * (1.0 - j_supply))
            * rng.lognormal(0.0, 0.16, (n, 5)), 0.0, 0.075)
        j_loss = np.minimum(japanese, japanese * hazard)

        # Rapid Soviet penetrations yield more capture; prolonged isolated
        # fighting in China/Taiwan yields more deaths and wounds.
        fatal = np.clip(
            np.array([0.30, 0.43, 0.45, 0.31, 0.42])[None, :]
            + 0.20 * (1.0 - j_supply) + 0.10 * (1.0 - j_command), 0.22, 0.72)
        captured = np.clip(
            np.array([0.48, 0.22, 0.20, 0.42, 0.24])[None, :]
            + 0.22 * isolation - 0.18 * j_command, 0.08, 0.68)
        captured = np.minimum(captured, 0.93 - fatal)
        j_dead += j_loss * fatal
        j_captured_surrendered += j_loss * captured
        j_wounded_disabled += j_loss * (1.0 - fatal - captured)
        japanese -= j_loss

        # Collaborator competing risks.  The same person can experience only
        # one exit in a week.  Defectors transfer at reduced effective strength
        # because vetting, retraining and political distrust delay reuse.
        theater_signal = np.log(
            (opponent + 35.0) / (japanese + collaborator_effective + 35.0))
        peer_exit = np.clip(
            (collaborator_to_kmt[:, None] + collaborator_to_ccp[:, None])
            / np.maximum(collaborator_initial.sum(axis=1)[:, None], 1.0), 0.0, 1.0)
        access_total = np.maximum(kmt + ccp, 1.0)
        kmt_access = kmt / access_total
        ccp_access = ccp / access_total
        wage_failure = 1.0 - j_supply

        h_to_kmt = activated * np.exp(np.clip(
            -6.75 + 0.85 * theater_signal + 1.25 * isolation
            + 1.10 * wage_failure + 0.90 * peer_exit
            + 0.95 * kmt_access + 0.75 * amnesty_kmt[:, None], -9.0, -2.0))
        h_to_ccp = activated * np.exp(np.clip(
            -7.05 + 0.90 * theater_signal + 1.30 * isolation
            + 1.05 * wage_failure + 1.05 * peer_exit
            + 1.05 * ccp_access + 0.80 * amnesty_ccp[:, None], -9.0, -2.0))
        h_disband = activated * np.exp(np.clip(
            -7.15 + 1.20 * theater_signal + 1.45 * isolation
            + 1.30 * wage_failure + 0.80 * peer_exit, -9.0, -1.8))
        h_dead = activated * np.clip(
            0.00035 + 0.0020 * pressure_ratio * (0.45 + 0.70 * j_supply), 0.0, 0.012)
        h_wounded = activated * np.clip(1.85 * h_dead, 0.0, 0.020)
        hazards = np.stack([h_to_kmt, h_to_ccp, h_disband, h_dead, h_wounded], axis=2)
        total_h = hazards.sum(axis=2)
        event_prob = 1.0 - np.exp(-total_h)
        exits = collaborator * np.minimum(event_prob, 0.22)
        shares = hazards / np.maximum(total_h[:, :, None], 1e-12)
        flows = exits[:, :, None] * shares
        to_kmt, to_ccp, disbanded, collab_dead, collab_wounded = [
            flows[:, :, idx] for idx in range(5)]
        collaborator -= exits
        collaborator_to_kmt += to_kmt.sum(axis=1)
        collaborator_to_ccp += to_ccp.sum(axis=1)
        collaborator_disbanded_captured += disbanded.sum(axis=1)
        collaborator_dead += collab_dead.sum(axis=1)
        collaborator_wounded += collab_wounded.sum(axis=1)
        kmt += 0.58 * to_kmt
        ccp += 0.52 * to_ccp
        collaborator_attachment *= np.clip(
            1.0 - 0.018 * isolation - 0.012 * peer_exit, 0.94, 1.0)

        # Cause-specific opponent losses depend on surviving Japanese combat
        # power. Losses are allocated by each participant's effective exposure.
        j_fire = np.sqrt(
            (japanese + 0.42 * collaborator_effective + 20.0) / (opponent + 40.0))
        opp_hazard = np.clip(
            scale * chinese_relative_loss[:, None]
            * np.array([0.0042, 0.0105, 0.0090, 0.0032, 0.0052])[None, :]
            * activated * j_fire * (0.55 + 0.75 * j_supply)
            * rng.lognormal(0.0, 0.18, (n, 5)), 0.0, 0.025)
        total_opp_loss = opponent * opp_hazard
        exposure_total = np.maximum(kmt + ccp + soviet + us, 1.0)
        kmt_loss = np.minimum(kmt, total_opp_loss * kmt / exposure_total)
        ccp_loss = np.minimum(ccp, total_opp_loss * ccp / exposure_total)
        sov_loss = np.minimum(soviet, total_opp_loss * soviet / exposure_total)
        us_loss = np.minimum(us, total_opp_loss * us / exposure_total)

        kmt -= kmt_loss
        ccp -= ccp_loss
        soviet -= sov_loss
        us -= us_loss
        kmt_cas += kmt_loss.sum(axis=1)
        ccp_cas += ccp_loss.sum(axis=1)
        soviet_cas += sov_loss.sum(axis=1)
        us_cas += us_loss.sum(axis=1)
        kmt_dead += (kmt_loss * np.array([0.34, 0.36, 0.37, 0.0, 0.34])).sum(axis=1)
        ccp_dead += (ccp_loss * np.array([0.37, 0.39, 0.40, 0.0, 0.0])).sum(axis=1)
        soviet_dead += (sov_loss * np.array([0.26, 0.0, 0.0, 0.25, 0.0])).sum(axis=1)
        us_dead += (us_loss * np.array([0.0, 0.24, 0.24, 0.22, 0.25])).sum(axis=1)

        # Civilian risk combines ground exposure, displacement, food-system
        # loss, and disease. It is bounded by the remaining population stock.
        displacement += activated * np.array([0.0008, 0.0010, 0.0009, 0.0, 0.0011])[None, :]
        food *= np.clip(1.0 - activated * (
            0.00045 + 0.0010 * isolation + 0.0008 * displacement), 0.985, 1.0)
        combat_intensity = np.clip(j_loss / np.maximum(j_initial, 1.0), 0.0, 0.08)
        civ_hazard = (
            np.array([0.000025, 0.000020, 0.000017, 0.0, 0.000030])[None, :]
            + 0.0018 * combat_intensity
            + 0.000055 * activated * (1.0 - food) ** 2
            + 0.000020 * displacement)
        china_weekly = np.minimum(china_civ, china_civ * civ_hazard * MAINLAND[None, :])
        taiwan_weekly = np.minimum(taiwan_civ, taiwan_civ * civ_hazard * TAIWAN[None, :])
        china_civ -= china_weekly
        taiwan_civ -= taiwan_weekly
        china_civ_dead += china_weekly.sum(axis=1)
        taiwan_civ_dead += taiwan_weekly.sum(axis=1)
        china_civ_severe_nonfatal += (
            china_weekly * rng.lognormal(np.log(1.45), 0.12, (n, 5))).sum(axis=1)
        taiwan_civ_severe_nonfatal += (
            taiwan_weekly * rng.lognormal(np.log(1.25), 0.10, (n, 5))).sum(axis=1)

        # Central order and local collapse transfer survivors to surrender stock.
        comply_prob = np.clip(0.40 + 0.35 * (1.0 - j_command) + 0.18 * isolation, 0.35, 0.92)
        order_now = week >= order_week
        local_collapse = (j_supply < 0.12) & (j_command < 0.25)
        surrender_mask = (order_now[:, None] & (rng.random((n, 5)) < comply_prob)) | local_collapse
        surrendered = japanese * surrender_mask * np.where(order_now[:, None], 0.78, 0.42)
        japanese -= surrendered
        j_captured_surrendered += surrendered

        # Once Japanese command collapses, remaining collaborator formations
        # seek a patron or dissolve rather than persist as an independent army.
        proxy_collapse = order_now[:, None] | local_collapse | (japanese < 2.0)
        proxy_exit = collaborator * proxy_collapse * 0.55
        patron_total = np.maximum(kmt + ccp, 1.0)
        proxy_to_kmt = proxy_exit * 0.58 * kmt / patron_total
        proxy_to_ccp = proxy_exit * 0.42 * ccp / patron_total
        proxy_disband = np.maximum(0.0, proxy_exit - proxy_to_kmt - proxy_to_ccp)
        collaborator -= proxy_exit
        collaborator_to_kmt += proxy_to_kmt.sum(axis=1)
        collaborator_to_ccp += proxy_to_ccp.sum(axis=1)
        collaborator_disbanded_captured += proxy_disband.sum(axis=1)
        kmt += 0.45 * proxy_to_kmt
        ccp += 0.42 * proxy_to_ccp

    # Preserve V1.2 Southeast-Asia/Pacific component as a separately sampled
    # residual. The explicit module replaces China-Manchuria, Korea, and Taiwan.
    old_overseas = sample_from_reported(
        rng, source["unconditional"]["japanese_overseas_dead_thousands"], n)
    old_china_share = rng.beta(5.0, 3.2, n)  # broad attribution, not a fact
    residual_sea_pacific_dead = old_overseas * (1.0 - old_china_share)
    explicit_theater_dead = j_dead.sum(axis=1)
    revised_overseas_dead = explicit_theater_dead + residual_sea_pacific_dead

    old_home_military = sample_from_reported(
        rng, source["unconditional"]["japanese_home_dead_thousands"], n)
    old_home_civilian = sample_from_reported(
        rng, source["unconditional"]["japanese_civilian_dead_thousands"], n)
    duration = np.minimum(order_week, WEEKS)
    diversion = np.clip(0.025 + 0.055 * us_support * (duration / WEEKS), 0.02, 0.08)
    northern_relief = np.clip(0.02 * soviet_momentum, 0.01, 0.025)
    revised_home_military = old_home_military * (1.0 - 0.35 * diversion)
    revised_home_civilian = old_home_civilian * (
        1.0 + 0.025 * diversion + 0.012 * np.log1p(duration / 52.0) - northern_relief)
    revised_japanese_military = revised_home_military + revised_overseas_dead
    revised_japanese_total = revised_japanese_military + revised_home_civilian

    chinese_military_dead = kmt_dead + ccp_dead
    chinese_military_cas = kmt_cas + ccp_cas
    chinese_total_dead_mainland = chinese_military_dead + china_civ_dead
    china_taiwan_total_dead = chinese_total_dead_mainland + taiwan_civ_dead
    chinese_military_wounded_disabled = np.maximum(0.0, chinese_military_cas - chinese_military_dead)
    mainland_chinese_total_casualties = (
        chinese_military_cas + china_civ_dead + china_civ_severe_nonfatal)

    arrays = {
        "kmt_dead_thousands": kmt_dead,
        "ccp_dead_thousands": ccp_dead,
        "chinese_military_dead_thousands": chinese_military_dead,
        "chinese_military_casualties_thousands": chinese_military_cas,
        "chinese_military_wounded_disabled_thousands": chinese_military_wounded_disabled,
        "mainland_chinese_civilian_dead_thousands": china_civ_dead,
        "mainland_chinese_civilian_severe_nonfatal_thousands": china_civ_severe_nonfatal,
        "taiwan_civilian_dead_thousands": taiwan_civ_dead,
        "taiwan_civilian_severe_nonfatal_thousands": taiwan_civ_severe_nonfatal,
        "mainland_chinese_total_dead_thousands": chinese_total_dead_mainland,
        "mainland_chinese_total_casualties_thousands": mainland_chinese_total_casualties,
        "china_plus_taiwan_total_dead_thousands": china_taiwan_total_dead,
        "soviet_dead_thousands": soviet_dead,
        "us_china_korea_taiwan_dead_thousands": us_dead,
        "collaborator_dead_thousands": collaborator_dead,
        "collaborator_total_casualties_thousands": collaborator_dead + collaborator_wounded,
        "collaborator_defected_to_kmt_thousands": collaborator_to_kmt,
        "collaborator_defected_to_ccp_thousands": collaborator_to_ccp,
        "collaborator_disbanded_captured_thousands": collaborator_disbanded_captured,
        "collaborator_remaining_thousands": collaborator.sum(axis=1),
        "japanese_explicit_theater_dead_thousands": explicit_theater_dead,
        "japanese_revised_overseas_dead_thousands": revised_overseas_dead,
        "japanese_revised_military_dead_thousands": revised_japanese_military,
        "japanese_revised_home_civilian_dead_thousands": revised_home_civilian,
        "japanese_revised_total_dead_thousands": revised_japanese_total,
        "japanese_theater_captured_surrendered_thousands": j_captured_surrendered.sum(axis=1),
        "japanese_theater_remaining_thousands": japanese.sum(axis=1),
        "china_theater_resolution_week": theater_resolution_week,
    }
    result = {
        "scenario": scenario,
        "structure": structure,
        "paths": n,
        "metrics": {key: quantiles(value) for key, value in arrays.items()},
        "theater_japanese_dead": {
            theater: quantiles(j_dead[:, idx]) for idx, theater in enumerate(THEATERS)
        },
        "mass_balance_error_thousands": quantiles(
            (japanese + j_dead + j_wounded_disabled + j_captured_surrendered - j_initial).sum(axis=1)),
        "collaborator_mass_balance_error_thousands": quantiles(
            collaborator.sum(axis=1) + collaborator_dead + collaborator_wounded
            + collaborator_to_kmt + collaborator_to_ccp
            + collaborator_disbanded_captured - collaborator_initial.sum(axis=1)),
    }
    return result, arrays


def ensemble_summary(results: list[dict]) -> dict:
    grouped: dict[str, dict] = {}
    for scenario in sorted({x["scenario"] for x in results}):
        cells = [x for x in results if x["scenario"] == scenario]
        metrics: dict[str, dict] = {}
        for key in cells[0]["metrics"]:
            medians = np.array([x["metrics"][key]["median"] for x in cells])
            p05s = np.array([x["metrics"][key]["p05"] for x in cells])
            p95s = np.array([x["metrics"][key]["p95"] for x in cells])
            metrics[key] = {
                "ensemble_median_of_medians": float(np.median(medians)),
                "model_min_p05": float(np.min(p05s)),
                "model_max_p95": float(np.max(p95s)),
            }
        grouped[scenario] = metrics
    return grouped


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    results = []
    for idx, cell in enumerate(source["results"]):
        result, _ = simulate_cell(cell, seed=19450809 + idx * 7919)
        assert abs(result["mass_balance_error_thousands"]["median"]) < 1e-7
        assert abs(result["collaborator_mass_balance_error_thousands"]["median"]) < 1e-7
        results.append(result)

    ensemble = ensemble_summary(results)
    payload = {
        "model": "Operation Downfall V1.3.1 China-theater and collaborator coupled extension",
        "warning": "Structural counterfactual with wide priors; not a fitted casualty forecast.",
        "source_model": "Operation Downfall V1.2",
        "paths_per_scenario_structure": PATHS,
        "theaters": THEATERS,
        "historical_observation_model": {
            "chihchiang_chinese_total_casualties_thousands": [18.0, 21.1],
            "japanese_narrow_battle_casualties_thousands": 6.5,
            "japanese_broader_or_claimed_casualties_thousands": [11.0, 36.4],
            "regime_probabilities": CHIHCHIANG_REGIME_PROB.tolist(),
            "relative_loss_centers": CHIHCHIANG_RELATIVE_LOSS_CENTER.tolist(),
            "note": "Different source denominators are modeled as observation regimes, not averaged into a fixed exchange ratio.",
            "collaborator_force_anchor_thousands": {
                "manchuria": 300.0,
                "china_proper": 900.0
            },
            "collaborator_1944_defection_report_thousands": 34.167,
            "collaborator_note": "US diplomatic reporting and partisan claims are uncertain political observations, not error-free censuses.",
        },
        "results": results,
        "ensemble": ensemble,
    }
    (OUT / "没落行动V1.3_中国战区耦合仿真摘要.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for scenario, metrics in ensemble.items():
        row = {"scenario": scenario}
        for key, values in metrics.items():
            row[f"{key}_median"] = values["ensemble_median_of_medians"]
            row[f"{key}_low"] = values["model_min_p05"]
            row[f"{key}_high"] = values["model_max_p95"]
        rows.append(row)
    csv_path = OUT / "没落行动V1.3_中国战区情景结果.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "verification": "PASS",
        "cells": len(results),
        "paths": len(results) * PATHS,
        "ensemble": ensemble,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
