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
