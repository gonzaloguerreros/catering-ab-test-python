"""
main.py
=======
End-to-end runner for the SPRING20 A/B test analysis.

Steps
-----
1. Generate synthetic experiment dataset.
2. Run pre-experiment sanity checks (SRM, covariate balance).
3. Execute all registered statistical tests.
4. Compute retrospective power and MDE.
5. Run Bonferroni-corrected subgroup analysis.
6. Calculate campaign business impact and ROI.
7. Produce all charts.
8. Persist results to JSON for downstream consumption.

Usage
-----
From the repository root::

    python main.py

Outputs
-------
data/experiment_data.csv
outputs/01_conversion_rates.png  …  05_60d_gmv.png
outputs/results_summary.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must be set before any other matplotlib import

# Add src/ to the module search path (avoids editable-install requirement
# while keeping the package layout clean for a portfolio project).
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ab_test_analysis import (
    calculate_business_impact,
    calculate_power_analysis,
    check_covariate_balance,
    check_sample_ratio_mismatch,
    subgroup_analysis,
    analyze_60d_gmv,
    analyze_conversion_rate,
    analyze_first_order_gmv,
)
from config import DATA_DIR, OUTPUTS_DIR
from data_generator import generate_experiment_data
from experiment_design import (
    ExperimentSpec,
    required_sample_size_proportions,
    experiment_runtime_days,
    define_power_users,
)
from visualizations import (
    plot_60d_gmv,
    plot_conversion_rates,
    plot_cumulative_conversion,
    plot_gmv_distribution,
    plot_subgroup_lift,
)

# ---------------------------------------------------------------------------
# Logging — structured, level-controlled output replaces bare print()
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    """Emit a prominent section header to the log."""
    border = "=" * 65
    logger.info("\n%s\n  %s\n%s", border, title, border)


def _log_dict(d: dict, indent: int = 2) -> None:
    """Emit key-value pairs from a result dict at INFO level."""
    pad = " " * indent
    for key, val in d.items():
        logger.info("%s%-42s %s", pad, key, val)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_analysis() -> None:
    """Execute the full A/B test analysis pipeline."""

    # 0. PROSPECTIVE EXPERIMENT DESIGN (done before launch) ------------------
    _section("STEP 0 — Prospective Experiment Design (Pre-Launch)")

    spec = ExperimentSpec(
        name             = "SPRING20",
        hypothesis       = "20% first-order discount will increase 30-day conversion rate "
                           "for newly acquired mid-market accounts by reducing first-order friction.",
        unit             = "corporate_account",
        control_desc     = "Standard onboarding — no discount",
        treatment_desc   = "20% off first order",
        primary_metric   = "30-day conversion rate",
        secondary_metrics= ["First-order GMV (converters only)", "60-day GMV (all accounts)",
                            "Orders in first 60 days"],
        guardrail_metrics= ["Cancellation rate", "Caterer on-time delivery rate",
                            "Support ticket rate"],
        segment_cuts     = ["Account tier (SMB / mid-market / enterprise)",
                            "Industry vertical",
                            "Power users (≥ 4 orders in prior 90 days) vs. regular",
                            "New accounts (< 60 days) vs. established"],
        alpha            = 0.05,
        power            = 0.80,
        min_detectable_effect = 0.03,
    )
    logger.info("\n%s", spec.summary())

    # Required sample size — prospective calculation
    sizing = required_sample_size_proportions(
        baseline_rate=0.25,       # historical baseline from data warehouse
        min_detectable_effect=0.03,
    )
    logger.info(
        "[Prospective Sizing]  MDE=+3pp | N per group: %d | N total: %d",
        sizing["n_per_group"], sizing["n_total"],
    )

    runtime = experiment_runtime_days(
        n_required=sizing["n_total"],
        daily_eligible=30,        # ~30 new mid-market accounts/day historically
    )
    logger.info(
        "[Runtime Estimate]  %d days (%d weeks) to reach required N",
        runtime["recommended_days"], runtime["recommended_weeks"],
    )

    power_user_def = define_power_users(orders_threshold=4, lookback_days=90)
    logger.info("[Power User Segment]  %s", power_user_def["definition"])
    logger.info(
        "[Note] This experiment has only %d accounts per group — "
        "underpowered for the +3pp MDE target. Results are illustrative.",
        250,
    )

    # 1. Generate data -------------------------------------------------------
    _section("STEP 1 — Generating Experiment Dataset")
    df = generate_experiment_data()
    out_path = DATA_DIR / "experiment_data.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved dataset → %s", out_path)

    # 2. Pre-experiment checks -----------------------------------------------
    _section("STEP 2 — Pre-Experiment Sanity Checks")

    srm = check_sample_ratio_mismatch(df)
    logger.info("[SRM Check]")
    _log_dict(srm)
    if srm["srm_detected"]:
        logger.error("SRM detected — halt analysis until randomisation is audited.")
        sys.exit(1)

    balance = check_covariate_balance(df)
    logger.info(
        "[Industry Balance]  chi2_p=%.4f — %s",
        balance.attrs["chi2_p_value"],
        "Balanced" if balance.attrs["balanced"] else "IMBALANCED",
    )

    # 3. Statistical tests ---------------------------------------------------
    _section("STEP 3 — Statistical Tests")

    conv     = analyze_conversion_rate(df)
    gmv_fst  = analyze_first_order_gmv(df)
    gmv_60d  = analyze_60d_gmv(df)

    logger.info("[Conversion Rate]")
    _log_dict(conv)
    logger.info("[First-Order GMV]")
    _log_dict(gmv_fst)
    logger.info("[60-Day GMV]")
    _log_dict(gmv_60d)

    # 4. Power analysis ------------------------------------------------------
    _section("STEP 4 — Statistical Power & MDE")
    power = calculate_power_analysis(df)
    _log_dict(power)

    # 5. Subgroup analysis ---------------------------------------------------
    _section("STEP 5 — Subgroup Analysis (Bonferroni Corrected)")
    subgroups = subgroup_analysis(df)
    logger.info("\n%s", subgroups.to_string(index=False))

    # 6. Business impact -----------------------------------------------------
    _section("STEP 6 — Business Impact & Campaign ROI")
    impact = calculate_business_impact(df)
    _log_dict(impact)

    # 7. Visualisations ------------------------------------------------------
    _section("STEP 7 — Generating Charts")
    plot_conversion_rates(conv)
    plot_gmv_distribution(df)
    plot_cumulative_conversion(df)
    plot_subgroup_lift(subgroups)
    plot_60d_gmv(df, gmv_60d)

    # 8. Persist results -----------------------------------------------------
    _section("STEP 8 — Persisting Results")
    summary = {
        "experiment":      "SPRING20",
        "conversion":      conv,
        "first_order_gmv": gmv_fst,
        "gmv_60d":         gmv_60d,
        "power":           power,
        "business_impact": impact,
    }
    results_path = OUTPUTS_DIR / "results_summary.json"
    with results_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Results saved → %s", results_path)

    # Executive summary ------------------------------------------------------
    _section("EXECUTIVE SUMMARY — SPRING20 Recommendation")

    ship = conv["significant"] and gmv_60d["significant"] and impact["roi_positive"]
    recommendation = "SHIP 🚀" if ship else "DO NOT SHIP ✗"

    logger.info(
        "\n"
        "  Campaign      : SPRING20 — 20%% first-order discount (mid-market)\n"
        "  Sample        : %d accounts (250 control / 250 treatment)\n"
        "\n"
        "  Conversion    : %.1f%% → %.1f%% (%+.1f pp, p=%.4f, %s)\n"
        "  First-order $ : $%.0f → $%.0f (+$%.0f, p=%.4f, %s)\n"
        "  60-day GMV    : $%.0f → $%.0f (+$%.0f, p=%.4f, %s)\n"
        "  Campaign ROI  : %.2fx  ($%.0f discount → $%.0f incremental revenue)\n"
        "  Power         : %.1f%%\n"
        "\n"
        "  Recommendation: %s\n",
        len(df),
        conv["rate_control"] * 100, conv["rate_treatment"] * 100,
        conv["absolute_lift"] * 100, conv["p_value"],
        "SIG" if conv["significant"] else "NS",
        gmv_fst["mean_control"], gmv_fst["mean_treatment"],
        gmv_fst["absolute_mean_lift"], gmv_fst["p_value_welch"],
        "SIG" if gmv_fst["significant_welch"] else "NS",
        gmv_60d["mean_control"], gmv_60d["mean_treatment"],
        gmv_60d["absolute_mean_lift"], gmv_60d["p_value"],
        "SIG" if gmv_60d["significant"] else "NS",
        impact["campaign_roi"] or 0,
        impact["total_discount_cost_usd"],
        impact["incremental_platform_revenue"],
        power["achieved_power"] * 100,
        recommendation,
    )


if __name__ == "__main__":
    run_analysis()
