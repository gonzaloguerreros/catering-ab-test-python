"""
ab_test_analysis.py
===================
Core statistical engine for the A/B test evaluation.

Covers:
  1. Pre-experiment sanity checks (Sample Ratio Mismatch, covariate balance)
  2. Primary KPI:  conversion rate — two-proportion z-test
  3. Secondary KPI: first-order GMV — Welch's t-test + Mann-Whitney U
  4. Secondary KPI: 60-day GMV — Welch's t-test
  5. Statistical power and minimum detectable effect (MDE) calculation
  6. Subgroup / segment analysis with multiple-testing correction (Bonferroni)
  7. Practical significance: uplift $, discount ROI, net revenue impact

Design notes
------------
- Two-sided tests are used by default; one-sided are available via `alternative`.
- We report both p-values and 95% confidence intervals — p-values alone are
  insufficient for decision-making (see American Statistical Association, 2016).
- Effect sizes (Cohen's h for proportions, Cohen's d for means) are included
  so stakeholders can assess practical, not just statistical, significance.
- All functions return plain dicts so results can be serialised to JSON or
  loaded into a DataFrame for reporting.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.power import NormalIndPower, zt_ind_solve_power

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Alpha and power thresholds used throughout
# ---------------------------------------------------------------------------
ALPHA = 0.05       # significance level
POWER = 0.80       # desired statistical power for MDE calculations


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h: effect size for difference between two proportions.
    Interpretation: 0.2 small, 0.5 medium, 0.8 large.
    """
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def _cohens_d(mean1: float, mean2: float, std1: float, std2: float,
              n1: int, n2: int) -> float:
    """
    Pooled Cohen's d: standardised mean difference.
    Interpretation: 0.2 small, 0.5 medium, 0.8 large.
    """
    pooled_std = np.sqrt(
        ((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2)
    )
    return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0


# ---------------------------------------------------------------------------
# 1. Pre-experiment checks
# ---------------------------------------------------------------------------

def check_sample_ratio_mismatch(df: pd.DataFrame,
                                 expected_ratio: float = 0.5) -> dict:
    """
    Sample Ratio Mismatch (SRM) test.

    A significant deviation from the expected 50/50 split suggests a
    randomisation or logging bug — the experiment result should NOT be trusted
    if SRM is detected.  Uses a chi-squared goodness-of-fit test.

    Parameters
    ----------
    df             : experiment dataframe (must have 'group' column)
    expected_ratio : expected fraction in treatment group (default 0.5)

    Returns
    -------
    dict with counts, observed ratio, chi2 statistic, p-value, and SRM flag.
    """
    counts    = df["group"].value_counts()
    n_control   = counts.get("control", 0)
    n_treatment = counts.get("treatment", 0)
    n_total     = n_control + n_treatment

    # Chi-squared goodness-of-fit: observed vs. expected counts
    expected_treatment = n_total * expected_ratio
    expected_control   = n_total * (1 - expected_ratio)

    chi2, p_value = stats.chisquare(
        f_obs=[n_control, n_treatment],
        f_exp=[expected_control, expected_treatment]
    )

    return {
        "n_control":        n_control,
        "n_treatment":      n_treatment,
        "observed_ratio":   round(n_treatment / n_total, 4),
        "expected_ratio":   expected_ratio,
        "chi2_statistic":   round(chi2, 4),
        "p_value":          round(p_value, 4),
        "srm_detected":     p_value < ALPHA,
    }


def check_industry_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verify that the industry distribution is balanced between groups.
    Imbalance would be a confound — Technology-heavy treatment groups would
    show inflated lift because Tech accounts respond better to discounts.

    Returns a DataFrame with industry shares per group and a chi-squared
    p-value for the overall distribution difference.
    """
    # Pivot: rows = industry, cols = group, values = share within group
    balance = (
        df.groupby(["group", "industry"])
          .size()
          .reset_index(name="count")
    )
    total_per_group = df.groupby("group").size().rename("total")
    balance = balance.merge(total_per_group, on="group")
    balance["share_pct"] = (balance["count"] / balance["total"] * 100).round(1)

    # Chi-squared test on the contingency table
    contingency = pd.crosstab(df["industry"], df["group"])
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency)

    balance_pivot = balance.pivot(index="industry", columns="group",
                                  values="share_pct").reset_index()
    balance_pivot.attrs["chi2_p_value"] = round(p_val, 4)
    balance_pivot.attrs["balanced"]     = p_val >= ALPHA

    return balance_pivot


# ---------------------------------------------------------------------------
# 2. Primary KPI — Conversion Rate
# ---------------------------------------------------------------------------

def test_conversion_rate(df: pd.DataFrame,
                         alternative: Literal["two-sided","larger","smaller"]
                         = "two-sided") -> dict:
    """
    Two-proportion z-test for the difference in conversion rates.

    H0: conversion_rate_treatment == conversion_rate_control
    H1: conversion_rate_treatment != conversion_rate_control  (two-sided)

    Also computes:
    - 95% CI on the absolute rate difference (Newcombe method)
    - Relative uplift (%)
    - Cohen's h effect size
    """
    ctrl  = df[df["group"] == "control"]
    treat = df[df["group"] == "treatment"]

    n_ctrl,  conv_ctrl  = len(ctrl),  ctrl["converted"].sum()
    n_treat, conv_treat = len(treat), treat["converted"].sum()

    rate_ctrl  = conv_ctrl  / n_ctrl
    rate_treat = conv_treat / n_treat

    # Two-proportion z-test
    count   = np.array([conv_treat, conv_ctrl])
    nobs    = np.array([n_treat,    n_ctrl])
    z_stat, p_value = proportions_ztest(count, nobs, alternative=alternative)

    # 95% CI on each rate individually (Wilson score interval — preferred over
    # normal approximation for rates close to 0 or 1)
    ci_ctrl_lo,  ci_ctrl_hi  = proportion_confint(conv_ctrl,  n_ctrl,  alpha=ALPHA, method="wilson")
    ci_treat_lo, ci_treat_hi = proportion_confint(conv_treat, n_treat, alpha=ALPHA, method="wilson")

    # Absolute difference CI (Newcombe's hybrid score method)
    diff = rate_treat - rate_ctrl
    se_diff = np.sqrt(
        rate_treat * (1 - rate_treat) / n_treat +
        rate_ctrl  * (1 - rate_ctrl)  / n_ctrl
    )
    z_crit = stats.norm.ppf(1 - ALPHA / 2)
    ci_diff_lo = diff - z_crit * se_diff
    ci_diff_hi = diff + z_crit * se_diff

    return {
        "metric":               "Conversion Rate",
        "n_control":            n_ctrl,
        "n_treatment":          n_treat,
        "conversions_control":  int(conv_ctrl),
        "conversions_treatment":int(conv_treat),
        "rate_control":         round(rate_ctrl,  4),
        "rate_treatment":       round(rate_treat, 4),
        "absolute_lift":        round(diff, 4),
        "relative_lift_pct":    round(diff / rate_ctrl * 100, 2),
        "ci_95_diff":           (round(ci_diff_lo, 4), round(ci_diff_hi, 4)),
        "ci_95_control":        (round(ci_ctrl_lo, 4), round(ci_ctrl_hi, 4)),
        "ci_95_treatment":      (round(ci_treat_lo, 4), round(ci_treat_hi, 4)),
        "z_statistic":          round(z_stat, 4),
        "p_value":              round(p_value, 6),
        "significant":          p_value < ALPHA,
        "cohens_h":             round(_cohens_h(rate_treat, rate_ctrl), 4),
        "alternative":          alternative,
    }


# ---------------------------------------------------------------------------
# 3. Secondary KPI — First-Order GMV (among converters only)
# ---------------------------------------------------------------------------

def test_first_order_gmv(df: pd.DataFrame) -> dict:
    """
    Welch's t-test (unequal variances) + Mann-Whitney U for first-order GMV.

    Analysis is restricted to converted accounts — we're testing whether the
    discount nudges customers to order MORE food, not just whether it converts
    them (that's the conversion test above).

    Welch's t-test is used over Student's t-test because we cannot assume
    equal variances between groups — treatment accounts have a higher incentive
    to place larger orders.

    Mann-Whitney U is a non-parametric cross-check that is robust to the
    right-skewed distribution of order values.
    """
    converters = df[df["converted"] == 1]

    ctrl_vals  = converters.loc[converters["group"] == "control",   "first_order_value"].dropna()
    treat_vals = converters.loc[converters["group"] == "treatment",  "first_order_value"].dropna()

    # Welch's t-test
    t_stat, p_welch = stats.ttest_ind(treat_vals, ctrl_vals, equal_var=False)

    # 95% CI on mean difference (Welch's method)
    mean_diff = treat_vals.mean() - ctrl_vals.mean()
    se = np.sqrt(treat_vals.var(ddof=1)/len(treat_vals) + ctrl_vals.var(ddof=1)/len(ctrl_vals))
    # Welch-Satterthwaite degrees of freedom
    df_welch = (
        (treat_vals.var(ddof=1)/len(treat_vals) + ctrl_vals.var(ddof=1)/len(ctrl_vals))**2
        / (
            (treat_vals.var(ddof=1)/len(treat_vals))**2 / (len(treat_vals)-1)
          + (ctrl_vals.var(ddof=1)/len(ctrl_vals))**2   / (len(ctrl_vals)-1)
        )
    )
    t_crit = stats.t.ppf(1 - ALPHA/2, df=df_welch)
    ci_lo  = mean_diff - t_crit * se
    ci_hi  = mean_diff + t_crit * se

    # Mann-Whitney U (non-parametric)
    u_stat, p_mann = stats.mannwhitneyu(treat_vals, ctrl_vals, alternative="two-sided")

    return {
        "metric":                   "First-Order GMV (converters only)",
        "n_control":                len(ctrl_vals),
        "n_treatment":              len(treat_vals),
        "mean_control":             round(ctrl_vals.mean(), 2),
        "mean_treatment":           round(treat_vals.mean(), 2),
        "median_control":           round(ctrl_vals.median(), 2),
        "median_treatment":         round(treat_vals.median(), 2),
        "std_control":              round(ctrl_vals.std(ddof=1), 2),
        "std_treatment":            round(treat_vals.std(ddof=1), 2),
        "absolute_mean_lift":       round(mean_diff, 2),
        "relative_mean_lift_pct":   round(mean_diff / ctrl_vals.mean() * 100, 2),
        "ci_95_mean_diff":          (round(ci_lo, 2), round(ci_hi, 2)),
        "t_statistic_welch":        round(t_stat, 4),
        "p_value_welch":            round(p_welch, 6),
        "significant_welch":        p_welch < ALPHA,
        "u_statistic_mann_whitney": round(u_stat, 2),
        "p_value_mann_whitney":     round(p_mann, 6),
        "significant_mann_whitney": p_mann < ALPHA,
        "cohens_d":                 round(_cohens_d(
                                        treat_vals.mean(), ctrl_vals.mean(),
                                        treat_vals.std(ddof=1), ctrl_vals.std(ddof=1),
                                        len(treat_vals), len(ctrl_vals)), 4),
    }


# ---------------------------------------------------------------------------
# 4. Secondary KPI — 60-Day GMV (all accounts, zeros for non-converters)
# ---------------------------------------------------------------------------

def test_60d_gmv(df: pd.DataFrame) -> dict:
    """
    Welch's t-test on 60-day GMV including zero-value non-converters.

    This is the most important revenue metric for the business: how much
    incremental revenue does the campaign generate per account acquired,
    regardless of whether they converted on the first order?
    """
    ctrl_gmv  = df.loc[df["group"] == "control",   "gmv_60d"]
    treat_gmv = df.loc[df["group"] == "treatment",  "gmv_60d"]

    t_stat, p_value = stats.ttest_ind(treat_gmv, ctrl_gmv, equal_var=False)
    mean_diff = treat_gmv.mean() - ctrl_gmv.mean()

    # CI
    se = np.sqrt(treat_gmv.var(ddof=1)/len(treat_gmv) + ctrl_gmv.var(ddof=1)/len(ctrl_gmv))
    df_welch = (
        (treat_gmv.var(ddof=1)/len(treat_gmv) + ctrl_gmv.var(ddof=1)/len(ctrl_gmv))**2
        / (
            (treat_gmv.var(ddof=1)/len(treat_gmv))**2 / (len(treat_gmv)-1)
          + (ctrl_gmv.var(ddof=1)/len(ctrl_gmv))**2   / (len(ctrl_gmv)-1)
        )
    )
    t_crit = stats.t.ppf(1 - ALPHA/2, df=df_welch)

    return {
        "metric":                 "60-Day GMV (all accounts)",
        "n_control":              len(ctrl_gmv),
        "n_treatment":            len(treat_gmv),
        "mean_control":           round(ctrl_gmv.mean(), 2),
        "mean_treatment":         round(treat_gmv.mean(), 2),
        "absolute_mean_lift":     round(mean_diff, 2),
        "relative_mean_lift_pct": round(mean_diff / ctrl_gmv.mean() * 100, 2),
        "ci_95_mean_diff":        (round(mean_diff - t_crit*se, 2),
                                   round(mean_diff + t_crit*se, 2)),
        "t_statistic":            round(t_stat, 4),
        "p_value":                round(p_value, 6),
        "significant":            p_value < ALPHA,
        "cohens_d":               round(_cohens_d(
                                      treat_gmv.mean(), ctrl_gmv.mean(),
                                      treat_gmv.std(ddof=1), ctrl_gmv.std(ddof=1),
                                      len(treat_gmv), len(ctrl_gmv)), 4),
    }


# ---------------------------------------------------------------------------
# 5. Statistical Power & MDE
# ---------------------------------------------------------------------------

def calculate_power_analysis(df: pd.DataFrame) -> dict:
    """
    Retrospective power analysis and minimum detectable effect (MDE).

    Reports:
    - Achieved power for the observed conversion rate effect
    - MDE: smallest effect size detectable at 80% power given sample size
    - Whether the experiment was adequately powered
    """
    n = len(df) // 2    # assume equal split

    # Effect size observed
    rate_ctrl  = df.loc[df["group"] == "control",   "converted"].mean()
    rate_treat = df.loc[df["group"] == "treatment",  "converted"].mean()
    observed_h = abs(_cohens_h(rate_treat, rate_ctrl))

    # Achieved power
    power_analysis = NormalIndPower()
    achieved_power = power_analysis.solve_power(
        effect_size=observed_h,
        nobs1=n,
        alpha=ALPHA,
        ratio=1.0,
        alternative="two-sided"
    )

    # MDE at 80% power
    mde_h = zt_ind_solve_power(
        effect_size=None,
        nobs1=n,
        alpha=ALPHA,
        power=POWER,
        ratio=1.0,
        alternative="two-sided"
    )
    # Convert Cohen's h MDE back to an approximate absolute rate difference
    # (approximation valid for rates near 0.5)
    mde_rate_diff = mde_h / 2

    return {
        "n_per_group":          n,
        "alpha":                ALPHA,
        "target_power":         POWER,
        "observed_effect_h":    round(observed_h, 4),
        "achieved_power":       round(achieved_power, 4),
        "adequately_powered":   achieved_power >= POWER,
        "mde_cohens_h":         round(mde_h, 4),
        "mde_approx_rate_diff": round(mde_rate_diff, 4),
    }


# ---------------------------------------------------------------------------
# 6. Subgroup Analysis with Bonferroni Correction
# ---------------------------------------------------------------------------

def subgroup_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the conversion rate test separately for each industry segment.

    Multiple comparisons inflate Type I error.  Bonferroni correction is
    applied: adjusted alpha = 0.05 / number_of_segments.  Only segments
    with p < adjusted_alpha are flagged as significant.

    This analysis answers: "Does the promo work uniformly, or are there
    segments where it's more/less effective?"  Heterogeneous treatment
    effects inform targeting decisions for future campaigns.
    """
    segments       = df["industry"].unique()
    adjusted_alpha = ALPHA / len(segments)   # Bonferroni adjustment

    rows = []
    for segment in sorted(segments):
        seg_df   = df[df["industry"] == segment]
        ctrl     = seg_df[seg_df["group"] == "control"]
        treat    = seg_df[seg_df["group"] == "treatment"]

        if len(ctrl) < 5 or len(treat) < 5:
            # Too few observations for a reliable test — skip
            continue

        n_ctrl,  conv_ctrl  = len(ctrl),  ctrl["converted"].sum()
        n_treat, conv_treat = len(treat), treat["converted"].sum()

        rate_ctrl  = conv_ctrl  / n_ctrl  if n_ctrl  > 0 else 0
        rate_treat = conv_treat / n_treat if n_treat > 0 else 0

        try:
            _, p_val = proportions_ztest(
                [conv_treat, conv_ctrl], [n_treat, n_ctrl], alternative="two-sided"
            )
        except Exception:
            p_val = np.nan

        rows.append({
            "industry":              segment,
            "n_control":             n_ctrl,
            "n_treatment":           n_treat,
            "rate_control":          round(rate_ctrl,  4),
            "rate_treatment":        round(rate_treat, 4),
            "absolute_lift":         round(rate_treat - rate_ctrl, 4),
            "relative_lift_pct":     round((rate_treat - rate_ctrl) / rate_ctrl * 100, 1)
                                     if rate_ctrl > 0 else np.nan,
            "p_value":               round(p_val, 4) if not np.isnan(p_val) else np.nan,
            "bonferroni_alpha":      round(adjusted_alpha, 4),
            "significant_bonferroni":bool(p_val < adjusted_alpha) if not np.isnan(p_val) else False,
        })

    return pd.DataFrame(rows).sort_values("relative_lift_pct", ascending=False)


# ---------------------------------------------------------------------------
# 7. Business Impact — Discount ROI
# ---------------------------------------------------------------------------

def calculate_business_impact(df: pd.DataFrame,
                               platform_take_rate: float = 0.20) -> dict:
    """
    Translate statistical results into business language.

    Calculates:
    - Total discount cost of the campaign
    - Incremental GMV attributable to the treatment
    - Incremental platform revenue (take_rate × incremental_GMV)
    - Net campaign ROI = incremental_revenue / discount_cost

    Parameters
    ----------
    platform_take_rate : fraction of GMV the platform retains as revenue
                         (ezCater's model is ~15-20%)
    """
    treat = df[df["group"] == "treatment"]
    ctrl  = df[df["group"] == "control"]

    total_discount_cost   = treat["discount_applied"].sum()
    mean_gmv_treat        = treat["gmv_60d"].mean()
    mean_gmv_ctrl         = ctrl["gmv_60d"].mean()
    incremental_gmv_per_account = mean_gmv_treat - mean_gmv_ctrl

    # Scale incremental GMV to full treatment cohort
    total_incremental_gmv = incremental_gmv_per_account * len(treat)
    incremental_revenue   = total_incremental_gmv * platform_take_rate

    roi = incremental_revenue / total_discount_cost if total_discount_cost > 0 else np.nan

    return {
        "n_treatment_accounts":          len(treat),
        "total_discount_cost_usd":       round(total_discount_cost, 2),
        "avg_discount_per_account_usd":  round(total_discount_cost / len(treat), 2),
        "incremental_gmv_per_account":   round(incremental_gmv_per_account, 2),
        "total_incremental_gmv_usd":     round(total_incremental_gmv, 2),
        "platform_take_rate":            platform_take_rate,
        "incremental_platform_revenue":  round(incremental_revenue, 2),
        "campaign_roi":                  round(roi, 2),
        "roi_positive":                  roi > 1.0 if not np.isnan(roi) else False,
    }
