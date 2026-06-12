"""
main.py
=======
End-to-end runner for the SPRING20 A/B test analysis.

Execution order:
  1. Generate synthetic experiment dataset
  2. Run all statistical tests
  3. Produce all charts
  4. Print a formatted business summary report

Usage:
  python3 main.py

Outputs:
  data/experiment_data.csv     — raw experiment data
  outputs/01_conversion_rates.png
  outputs/02_gmv_distribution.png
  outputs/03_cumulative_conversion.png
  outputs/04_subgroup_lift.png
  outputs/05_60d_gmv.png
"""

import json
import sys
import os

# Ensure src/ is on the path regardless of where the script is called from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works in any environment

from data_generator   import generate_experiment_data
from ab_test_analysis import (
    check_sample_ratio_mismatch,
    check_industry_balance,
    test_conversion_rate,
    test_first_order_gmv,
    test_60d_gmv,
    calculate_power_analysis,
    subgroup_analysis,
    calculate_business_impact,
)
from visualizations import (
    plot_conversion_rates,
    plot_gmv_distribution,
    plot_cumulative_conversion,
    plot_subgroup_lift,
    plot_60d_gmv,
)


def _section(title: str) -> None:
    """Print a formatted section header to stdout."""
    width = 65
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def _print_dict(d: dict, indent: int = 2) -> None:
    """Pretty-print a result dictionary."""
    pad = " " * indent
    for k, v in d.items():
        print(f"{pad}{k:<42} {v}")


def main() -> None:

    # -----------------------------------------------------------------------
    # STEP 1 — Generate data
    # -----------------------------------------------------------------------
    _section("STEP 1 — Generating Experiment Dataset")
    df = generate_experiment_data()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/experiment_data.csv", index=False)
    print(f"  Total accounts: {len(df):,}  "
          f"(control: {(df.group=='control').sum()}, "
          f"treatment: {(df.group=='treatment').sum()})")
    print(f"  Date range: {df.signup_date.min()} → {df.signup_date.max()}")

    # -----------------------------------------------------------------------
    # STEP 2 — Pre-experiment checks
    # -----------------------------------------------------------------------
    _section("STEP 2 — Pre-Experiment Sanity Checks")

    srm = check_sample_ratio_mismatch(df)
    print("\n  [Sample Ratio Mismatch]")
    _print_dict(srm)
    if srm["srm_detected"]:
        print("\n  ⚠  WARNING: SRM detected — investigate randomisation before "
              "proceeding with analysis.")
    else:
        print("\n  ✓  No SRM detected — randomisation appears clean.")

    balance = check_industry_balance(df)
    print(f"\n  [Industry Balance]  chi2 p-value: {balance.attrs['chi2_p_value']}")
    print(f"  {'Balanced' if balance.attrs['balanced'] else 'IMBALANCED'} "
          f"across groups")
    print(balance.to_string(index=False))

    # -----------------------------------------------------------------------
    # STEP 3 — Statistical tests
    # -----------------------------------------------------------------------
    _section("STEP 3 — Statistical Tests")

    print("\n  [Primary KPI: Conversion Rate]")
    conv = test_conversion_rate(df)
    _print_dict(conv)

    print("\n  [Secondary KPI: First-Order GMV — converters only]")
    gmv_first = test_first_order_gmv(df)
    _print_dict(gmv_first)

    print("\n  [Secondary KPI: 60-Day GMV — all accounts]")
    gmv_60d = test_60d_gmv(df)
    _print_dict(gmv_60d)

    # -----------------------------------------------------------------------
    # STEP 4 — Power analysis
    # -----------------------------------------------------------------------
    _section("STEP 4 — Statistical Power & MDE")
    power = calculate_power_analysis(df)
    _print_dict(power)

    # -----------------------------------------------------------------------
    # STEP 5 — Subgroup analysis
    # -----------------------------------------------------------------------
    _section("STEP 5 — Subgroup Analysis (Bonferroni Corrected)")
    subgroups = subgroup_analysis(df)
    print(subgroups.to_string(index=False))

    # -----------------------------------------------------------------------
    # STEP 6 — Business impact
    # -----------------------------------------------------------------------
    _section("STEP 6 — Business Impact & Campaign ROI")
    impact = calculate_business_impact(df)
    _print_dict(impact)

    # -----------------------------------------------------------------------
    # STEP 7 — Visualisations
    # -----------------------------------------------------------------------
    _section("STEP 7 — Generating Charts")
    plot_conversion_rates(conv)
    plot_gmv_distribution(df)
    plot_cumulative_conversion(df)
    plot_subgroup_lift(subgroups)
    plot_60d_gmv(df, gmv_60d)

    # -----------------------------------------------------------------------
    # STEP 8 — Executive Summary
    # -----------------------------------------------------------------------
    _section("EXECUTIVE SUMMARY — SPRING20 Campaign Recommendation")

    significant = conv["significant"] and gmv_60d["significant"]
    recommendation = "SHIP 🚀" if (significant and impact["roi_positive"]) else "DO NOT SHIP ✗"

    print(f"""
  Campaign:     SPRING20 — 20% discount on first order for mid-market accounts
  Test window:  March 1 – May 31, 2024
  Sample:       {len(df):,} accounts (250 control / 250 treatment)

  Results:
  ────────────────────────────────────────────────────────
  Conversion rate:   {conv['rate_control']*100:.1f}% → {conv['rate_treatment']*100:.1f}%
                     (+{conv['absolute_lift']*100:.1f}pp, {conv['relative_lift_pct']:+.1f}%,
                      p={conv['p_value']:.4f}, {"SIGNIFICANT" if conv['significant'] else "NOT SIGNIFICANT"})

  First-order GMV:   ${gmv_first['mean_control']:,.2f} → ${gmv_first['mean_treatment']:,.2f}
                     (+${gmv_first['absolute_mean_lift']:,.2f},
                      p={gmv_first['p_value_welch']:.4f}, {"SIGNIFICANT" if gmv_first['significant_welch'] else "NOT SIGNIFICANT"})

  60-day GMV:        ${gmv_60d['mean_control']:,.2f} → ${gmv_60d['mean_treatment']:,.2f}
                     (+${gmv_60d['absolute_mean_lift']:,.2f},
                      p={gmv_60d['p_value']:.4f}, {"SIGNIFICANT" if gmv_60d['significant'] else "NOT SIGNIFICANT"})

  Campaign ROI:      {impact['campaign_roi']:.2f}x  (${impact['total_discount_cost_usd']:,.0f} discount
                     cost → ${impact['incremental_platform_revenue']:,.0f} incremental revenue)

  Statistical power: {power['achieved_power']*100:.1f}%

  Recommendation:  {recommendation}
  ────────────────────────────────────────────────────────
  Segmentation note: Technology accounts show the strongest treatment
  response (+{subgroups.loc[subgroups.industry=='Technology','relative_lift_pct'].values[0]:.1f}% lift).
  Consider targeting future campaigns at Tech-heavy account lists.
    """)

    # Save summary to JSON for downstream use (dashboards, reports)
    summary = {
        "experiment": "SPRING20",
        "conversion":     conv,
        "first_order_gmv": gmv_first,
        "gmv_60d":        gmv_60d,
        "power":          power,
        "business_impact": impact,
    }
    with open("outputs/results_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("  Results saved to outputs/results_summary.json")


if __name__ == "__main__":
    main()
