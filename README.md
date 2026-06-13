# SPRING20 Promotional Campaign — A/B Test Analysis (Python)

**Role context:** Product Analyst portfolio project demonstrating applied statistics, experiment design, and business impact measurement for a B2B catering marketplace.

---

## Business Context

A B2B catering marketplace (modelled after ezCater) ran a promotional campaign — **SPRING20** — offering newly acquired mid-market corporate accounts a **20% discount on their first order**. The goal was to increase first-order conversion rates and drive higher average order values.

This project evaluates whether the campaign worked, whether it was worth the discount cost, and which customer segments responded most strongly.

---

## Experiment Design

| Parameter | Value |
|-----------|-------|
| Test window | March 1 – May 31, 2024 |
| Randomisation unit | Corporate account (not order-level) |
| Control | Standard onboarding — no discount |
| Treatment | 20% off first order |
| Primary KPI | Conversion rate (placed first order within 30 days) |
| Secondary KPIs | First-order GMV, 60-day GMV |
| Sample | 500 accounts (250 per group) |

Account-level randomisation was chosen over order-level to avoid **within-account contamination** — a known failure mode when the same account could see both variants.

---

## Results Summary

| Metric | Control | Treatment | Lift | p-value | Significant? |
|--------|---------|-----------|------|---------|-------------|
| Conversion Rate | 50.4% | 51.6% | +1.2pp (+2.4%) | 0.788 | ✗ No |
| First-Order GMV | $423.54 | $479.90 | +$56.37 (+13.3%) | 0.004 | ✓ Yes |
| 60-Day GMV | $592.26 | $697.55 | +$105.30 (+17.8%) | 0.109 | ✗ No |
| Campaign ROI | — | — | 0.43x | — | — |

**Recommendation: Do Not Ship in current form.**

The discount drives a meaningful lift in first-order size (+13%, statistically significant), but **does not convert more accounts**. Because conversion rate was flat, the 60-day GMV lift fails to reach significance — and the campaign ROI is 0.43x (every $1 of discount cost returns $0.43 in incremental platform revenue).

**Path forward:** Re-target the discount to **Finance and Legal** segments which showed 20%+ relative conversion lift (though underpowered individually), and consider a smaller discount (10%) to improve ROI while preserving the AOV lift effect.

---

## Statistical Methods

| Test | Purpose | Why This Test |
|------|---------|---------------|
| Two-proportion z-test | Conversion rate difference | Standard for binary outcomes at scale |
| Welch's t-test | GMV mean difference | Robust to unequal variances between groups |
| Mann-Whitney U | GMV distribution difference | Non-parametric cross-check for skewed order values |
| Wilson score CI | Rate confidence intervals | Superior to normal approximation near 0/1 |
| Chi-squared GoF | Sample ratio mismatch | Catches randomisation bugs before analysis |
| Bonferroni correction | Subgroup multiple comparisons | Controls family-wise Type I error rate |
| Retrospective power | Was the test adequately powered? | Explains why null results may be inconclusive |

---

## Project Structure

```
catering-ab-test-python/
├── src/
│   ├── data_generator.py      # Synthetic experiment data (500 accounts, 24 signals)
│   ├── ab_test_analysis.py    # All statistical tests and business impact metrics
│   └── visualizations.py     # matplotlib/seaborn chart functions
├── data/
│   └── experiment_data.csv    # Generated dataset
├── outputs/
│   ├── 01_conversion_rates.png
│   ├── 02_gmv_distribution.png
│   ├── 03_cumulative_conversion.png
│   ├── 04_subgroup_lift.png
│   ├── 05_60d_gmv.png
│   └── results_summary.json
├── main.py                    # End-to-end runner
└── requirements.txt
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run full analysis — generates data, runs tests, produces charts
python main.py
```

---

## Charts Generated

| Chart | Insight |
|-------|---------|
| `01_conversion_rates.png` | Side-by-side bar chart with 95% Wilson CI error bars |
| `02_gmv_distribution.png` | KDE + box plot of first-order GMV among converters |
| `03_cumulative_conversion.png` | Day-by-day conversion curves — tests for speed of conversion, not just rate |
| `04_subgroup_lift.png` | Forest-plot-style segment analysis with Bonferroni significance markers |
| `05_60d_gmv.png` | Violin plot of 60-day GMV including zero-value non-converters |

---

## Subgroup / Segment Analysis

A core part of any A/B test is cutting the data across dimensions to find **who the treatment actually works for** — and whether aggregate results are masking a strong effect in a specific segment.

This project implements two types of segment cuts:

| Segment | What it tests |
|---------|--------------|
| **Account tier** (SMB / Mid-market / Enterprise) | Does the discount resonate differently by company size? |
| **Industry vertical** (Finance, Legal, Tech, Healthcare, Retail) | Do certain industries place larger first orders? |
| **Power users** (accounts with ≥ 4 orders in prior 90 days) | Do already-engaged accounts respond differently to a discount incentive? |
| **Account age** (new < 60 days vs. established) | Is this a new-account acquisition tool or a re-engagement tool? |

All subgroup tests apply **Bonferroni correction** (α / k = 0.05 / 7 ≈ 0.007 per test) to control the family-wise Type I error rate when running multiple comparisons. Segments with fewer than 5 observations per arm are excluded automatically.

The power user cut is particularly relevant in a marketplace context: if the discount is primarily being captured by accounts that would have ordered anyway, it's pure margin erosion — not incremental growth.

**Key finding:** Finance and Legal verticals showed 20%+ relative conversion lift, though sample sizes were too small for individual significance. Recommended follow-up: re-run targeted at these verticals with a reduced discount (10%) to improve ROI.

---

## Key Analytical Decisions

- **Why account-level, not order-level randomisation?** Order-level randomisation in a B2B context creates contamination risk — the same account manager placing repeat orders could receive both treatment and control, inflating or masking the true effect.
- **Why Welch's t-test over Student's t-test?** Treatment accounts have a higher incentive to place larger orders, so we cannot assume equal variance. Welch's is the conservative and correct default.
- **Why include Mann-Whitney U?** Order values are right-skewed (a few large catering events dominate). The non-parametric test validates the parametric t-test result without distributional assumptions.
- **Why Bonferroni for subgroups?** Running 7 independent tests at α=0.05 gives a 30%+ chance of at least one false positive. Bonferroni controls this at the cost of reduced power — appropriate for a screening analysis.

---

*Dataset is fully synthetic. All figures and results are generated programmatically for portfolio demonstration purposes.*
