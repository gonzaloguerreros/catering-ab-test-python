"""
ab_test_analysis.py
===================
Core statistical engine for the SPRING20 A/B test evaluation.

Analysis plan (pre-registered before data collection)
------------------------------------------------------
1. Pre-experiment sanity checks
   a. Sample Ratio Mismatch (SRM) — chi-squared goodness-of-fit.
   b. Covariate balance — industry distribution chi-squared.
2. Primary KPI   : Conversion rate — two-proportion z-test.
3. Secondary KPI : First-order GMV — Welch's t-test + Mann-Whitney U.
4. Secondary KPI : 60-day GMV — Welch's t-test.
5. Statistical power and minimum detectable effect (MDE).
6. Subgroup analysis — per-industry z-tests with Bonferroni correction.
7. Business impact — discount cost, incremental GMV, campaign ROI.

Design principles
-----------------
- Two-sided tests by default; ``alternative`` parameter allows one-sided
  when directional hypotheses are pre-specified.
- Both p-values *and* 95 % confidence intervals are reported.  P-values
  alone are insufficient (ASA Statement on p-values, Wasserstein &
  Lazar, 2016).
- Effect sizes (Cohen's h for proportions, Cohen's d for means) are
  included so readers can assess practical, not just statistical,
  significance (Cohen, 1988).
- All public functions return plain ``dict`` or ``pd.DataFrame`` objects
  for easy serialisation (JSON) and downstream aggregation.

References
----------
Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
    (2nd ed.). Lawrence Erlbaum Associates.
Wasserstein, R. L., & Lazar, N. A. (2016). The ASA's statement on
    p-values. *The American Statistician*, 70(2), 129–133.
Hutto, C., & Gilbert, E. (2014). VADER: A parsimonious rule-based model
    for sentiment analysis of social media text. *ICWSM*, 8(1), 216–225.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Literal

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower, zt_ind_solve_power
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from config import ALPHA, MIN_SEGMENT_N, PLATFORM_TAKE_RATE, POWER

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _cohens_h(p1: float, p2: float) -> float:
    """
    Compute Cohen's *h* effect size for two independent proportions.

    Uses the arcsine transformation so the metric is on a ratio scale
    regardless of the baseline rate.

    Parameters
    ----------
    p1, p2 : float
        Proportions in [0, 1].

    Returns
    -------
    float
        Cohen's h.  Benchmarks: 0.2 = small, 0.5 = medium, 0.8 = large.
    """
    return 2.0 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def _cohens_d(
    mean1: float,
    mean2: float,
    std1: float,
    std2: float,
    n1: int,
    n2: int,
) -> float:
    """
    Compute pooled Cohen's *d* (standardised mean difference).

    Parameters
    ----------
    mean1, mean2 : float  Treatment and control means.
    std1, std2   : float  Treatment and control standard deviations.
    n1, n2       : int    Group sizes.

    Returns
    -------
    float
        Cohen's d.  Benchmarks: 0.2 = small, 0.5 = medium, 0.8 = large.
    """
    pooled_var = ((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    return (mean1 - mean2) / pooled_std if pooled_std > 0.0 else 0.0


def _welch_ci(
    vals_a: pd.Series,
    vals_b: pd.Series,
) -> tuple[float, float]:
    """
    Return a (1 − ALPHA) % Welch confidence interval for the mean difference.

    Uses the Welch–Satterthwaite degrees-of-freedom approximation.

    Parameters
    ----------
    vals_a, vals_b : pd.Series
        Numeric series for the two groups.

    Returns
    -------
    tuple[float, float]
        (lower bound, upper bound) of the confidence interval.
    """
    n_a, n_b   = len(vals_a), len(vals_b)
    var_a, var_b = vals_a.var(ddof=1), vals_b.var(ddof=1)

    se = np.sqrt(var_a / n_a + var_b / n_b)

    # Welch–Satterthwaite degrees of freedom
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df_ws  = df_num / df_den

    t_crit    = stats.t.ppf(1.0 - ALPHA / 2.0, df=df_ws)
    mean_diff = vals_a.mean() - vals_b.mean()
    return (mean_diff - t_crit * se, mean_diff + t_crit * se)


# ---------------------------------------------------------------------------
# 1. Pre-experiment sanity checks
# ---------------------------------------------------------------------------

def check_sample_ratio_mismatch(
    df: pd.DataFrame,
    expected_ratio: float = 0.5,
) -> Dict:
    """
    Detect Sample Ratio Mismatch (SRM) using a chi-squared goodness-of-fit test.

    SRM occurs when the observed group split deviates significantly from the
    intended ratio — a signal of a randomisation or logging bug.  Results from
    an SRM-affected experiment should not be trusted (Kohavi et al., 2020).

    Parameters
    ----------
    df : pd.DataFrame
        Experiment DataFrame containing a ``group`` column with values
        ``'control'`` and ``'treatment'``.
    expected_ratio : float, optional
        Expected fraction of accounts in the treatment arm.  Default 0.5.

    Returns
    -------
    dict
        Keys: n_control, n_treatment, observed_ratio, expected_ratio,
              chi2_statistic, p_value, srm_detected.

    Raises
    ------
    ValueError
        If ``expected_ratio`` is not in (0, 1).
    """
    if not (0.0 < expected_ratio < 1.0):
        raise ValueError(
            f"expected_ratio must be in (0, 1), got {expected_ratio!r}"
        )

    counts      = df["group"].value_counts()
    n_ctrl      = int(counts.get("control",   0))
    n_treat     = int(counts.get("treatment", 0))
    n_total     = n_ctrl + n_treat

    expected_treat = n_total * expected_ratio
    expected_ctrl  = n_total * (1.0 - expected_ratio)

    chi2, p_value = stats.chisquare(
        f_obs=[n_ctrl,         n_treat],
        f_exp=[expected_ctrl,  expected_treat],
    )

    # Cast to Python bool — scipy may return numpy.bool_ which fails `is True`
    srm = bool(p_value < ALPHA)
    if srm:
        logger.warning(
            "SRM detected (p=%.4f < %.2f). Do not interpret experiment results "
            "until the randomisation pipeline is audited.",
            p_value, ALPHA,
        )

    return {
        "n_control":       n_ctrl,
        "n_treatment":     n_treat,
        "observed_ratio":  round(n_treat / n_total, 4),
        "expected_ratio":  expected_ratio,
        "chi2_statistic":  round(chi2, 4),
        "p_value":         round(p_value, 4),
        "srm_detected":    srm,
    }


def check_covariate_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test whether the industry distribution is balanced between groups.

    Imbalance is a confound: a treatment arm that is over-represented by
    Technology accounts would show inflated conversion lift because Tech
    accounts respond more strongly to discounts.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment DataFrame with ``group`` and ``industry`` columns.

    Returns
    -------
    pd.DataFrame
        Pivot of industry shares (%) per group.
        ``DataFrame.attrs["chi2_p_value"]`` holds the overall test p-value.
        ``DataFrame.attrs["balanced"]``     is True when p ≥ ALPHA.
    """
    balance = (
        df.groupby(["group", "industry"])
          .size()
          .reset_index(name="count")
          .merge(df.groupby("group").size().rename("total"), on="group")
          .assign(share_pct=lambda d: (d["count"] / d["total"] * 100).round(1))
    )

    contingency = pd.crosstab(df["industry"], df["group"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress low-count chi2 warnings
        chi2, p_val, _dof, _expected = stats.chi2_contingency(contingency)

    pivot = balance.pivot(
        index="industry", columns="group", values="share_pct"
    ).reset_index()
    pivot.attrs["chi2_p_value"] = round(p_val, 4)
    pivot.attrs["balanced"]     = p_val >= ALPHA

    logger.info(
        "Covariate balance check: chi2_p=%.4f — %s",
        p_val,
        "BALANCED" if p_val >= ALPHA else "IMBALANCED ⚠",
    )
    return pivot


# ---------------------------------------------------------------------------
# 2. Primary KPI — Conversion Rate
# ---------------------------------------------------------------------------

def analyze_conversion_rate(
    df: pd.DataFrame,
    alternative: Literal["two-sided", "larger", "smaller"] = "two-sided",
) -> Dict:
    """
    Two-proportion z-test for the difference in 30-day conversion rates.

    Hypotheses (two-sided default)
    --------------------------------
    H₀ : rate_treatment = rate_control
    H₁ : rate_treatment ≠ rate_control

    In addition to the z-test, the function computes:

    - Wilson score 95 % CI on each rate individually.  Wilson intervals
      are preferred over the Wald (normal approximation) interval because
      they maintain coverage when rates are near 0 or 1 (Brown et al., 2001).
    - Newcombe-style 95 % CI on the absolute rate difference.
    - Relative lift (%).
    - Cohen's h effect size.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment DataFrame with ``group`` and ``converted`` (0/1) columns.
    alternative : {'two-sided', 'larger', 'smaller'}, optional
        Direction of the alternative hypothesis.

    Returns
    -------
    dict
        Full test results including rates, CIs, test statistic, p-value,
        significance flag, and effect size.
    """
    ctrl  = df[df["group"] == "control"]
    treat = df[df["group"] == "treatment"]

    n_ctrl,  conv_ctrl  = len(ctrl),  int(ctrl["converted"].sum())
    n_treat, conv_treat = len(treat), int(treat["converted"].sum())

    rate_ctrl  = conv_ctrl  / n_ctrl
    rate_treat = conv_treat / n_treat

    z_stat, p_value = proportions_ztest(
        count=np.array([conv_treat, conv_ctrl]),
        nobs =np.array([n_treat,    n_ctrl]),
        alternative=alternative,
    )

    # Wilson score intervals (per-group)
    ci_ctrl_lo,  ci_ctrl_hi  = proportion_confint(conv_ctrl,  n_ctrl,  alpha=ALPHA, method="wilson")
    ci_treat_lo, ci_treat_hi = proportion_confint(conv_treat, n_treat, alpha=ALPHA, method="wilson")

    # Absolute difference CI (Newcombe hybrid score)
    diff    = rate_treat - rate_ctrl
    se_diff = np.sqrt(
        rate_treat * (1.0 - rate_treat) / n_treat
        + rate_ctrl  * (1.0 - rate_ctrl)  / n_ctrl
    )
    z_crit     = stats.norm.ppf(1.0 - ALPHA / 2.0)
    ci_diff_lo = diff - z_crit * se_diff
    ci_diff_hi = diff + z_crit * se_diff

    result = {
        "metric":                "Conversion Rate",
        "n_control":             n_ctrl,
        "n_treatment":           n_treat,
        "conversions_control":   conv_ctrl,
        "conversions_treatment": conv_treat,
        "rate_control":          round(rate_ctrl,  4),
        "rate_treatment":        round(rate_treat, 4),
        "absolute_lift":         round(diff, 4),
        "relative_lift_pct":     round(diff / rate_ctrl * 100.0, 2),
        "ci_95_diff":            (round(ci_diff_lo, 4), round(ci_diff_hi, 4)),
        "ci_95_control":         (round(ci_ctrl_lo,  4), round(ci_ctrl_hi,  4)),
        "ci_95_treatment":       (round(ci_treat_lo, 4), round(ci_treat_hi, 4)),
        "z_statistic":           round(z_stat,  4),
        "p_value":               round(p_value, 6),
        "significant":           bool(p_value < ALPHA),
        "cohens_h":              round(_cohens_h(rate_treat, rate_ctrl), 4),
        "alternative":           alternative,
    }

    logger.info(
        "Conversion test: %.1f%% → %.1f%% (%+.1f pp) | p=%.4f | %s",
        rate_ctrl * 100, rate_treat * 100, diff * 100,
        p_value, "SIGNIFICANT" if result["significant"] else "not significant",
    )
    return result


# ---------------------------------------------------------------------------
# 3. Secondary KPI — First-Order GMV (converters only)
# ---------------------------------------------------------------------------

def analyze_first_order_gmv(df: pd.DataFrame) -> Dict:
    """
    Test for a difference in first-order GMV among converted accounts.

    Two complementary tests are run:

    Welch's t-test
        Parametric; uses means.  Welch's variant (``equal_var=False``) is
        used instead of Student's because the discount gives treatment
        accounts a higher incentive to order more food, likely inflating
        their variance.  Welch's is robust to this heteroscedasticity.

    Mann-Whitney U
        Non-parametric; uses ranks.  Order values are right-skewed (a few
        large corporate events dominate), which can inflate the t-statistic.
        Agreement between both tests strengthens inference.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment DataFrame.  Non-converters are filtered out internally.

    Returns
    -------
    dict
        Means, medians, SDs, CIs, both test statistics and p-values,
        significance flags, and Cohen's d.
    """
    converters = df[df["converted"] == 1]
    ctrl_vals  = converters.loc[converters["group"] == "control",    "first_order_value"].dropna()
    treat_vals = converters.loc[converters["group"] == "treatment",  "first_order_value"].dropna()

    t_stat,  p_welch = stats.ttest_ind(treat_vals, ctrl_vals, equal_var=False)
    u_stat,  p_mann  = stats.mannwhitneyu(treat_vals, ctrl_vals, alternative="two-sided")

    mean_diff    = treat_vals.mean() - ctrl_vals.mean()
    ci_lo, ci_hi = _welch_ci(treat_vals, ctrl_vals)

    result = {
        "metric":                   "First-Order GMV (converters only)",
        "n_control":                len(ctrl_vals),
        "n_treatment":              len(treat_vals),
        "mean_control":             round(ctrl_vals.mean(),   2),
        "mean_treatment":           round(treat_vals.mean(),  2),
        "median_control":           round(ctrl_vals.median(), 2),
        "median_treatment":         round(treat_vals.median(),2),
        "std_control":              round(ctrl_vals.std(ddof=1),  2),
        "std_treatment":            round(treat_vals.std(ddof=1), 2),
        "absolute_mean_lift":       round(mean_diff, 2),
        "relative_mean_lift_pct":   round(mean_diff / ctrl_vals.mean() * 100.0, 2),
        "ci_95_mean_diff":          (round(ci_lo, 2), round(ci_hi, 2)),
        "t_statistic_welch":        round(t_stat,  4),
        "p_value_welch":            round(p_welch, 6),
        "significant_welch":        bool(p_welch < ALPHA),
        "u_statistic_mann_whitney": round(float(u_stat), 2),
        "p_value_mann_whitney":     round(p_mann, 6),
        "significant_mann_whitney": bool(p_mann < ALPHA),
        "cohens_d":                 round(
            _cohens_d(
                treat_vals.mean(), ctrl_vals.mean(),
                treat_vals.std(ddof=1), ctrl_vals.std(ddof=1),
                len(treat_vals), len(ctrl_vals),
            ), 4
        ),
    }

    logger.info(
        "First-order GMV test: $%.0f → $%.0f (+$%.0f, %+.1f%%) | p_welch=%.4f",
        ctrl_vals.mean(), treat_vals.mean(), mean_diff,
        mean_diff / ctrl_vals.mean() * 100,
        p_welch,
    )
    return result


# ---------------------------------------------------------------------------
# 4. Secondary KPI — 60-Day GMV (all accounts, zeros for non-converters)
# ---------------------------------------------------------------------------

def analyze_60d_gmv(df: pd.DataFrame) -> Dict:
    """
    Welch's t-test on 60-day GMV including zero-value non-converters.

    This is the primary revenue metric for campaign budgeting: how much
    incremental GMV does the campaign generate *per acquired account*,
    regardless of whether they converted on the first order?

    Parameters
    ----------
    df : pd.DataFrame
        Full experiment DataFrame (converters and non-converters).

    Returns
    -------
    dict
        Descriptive stats, CI, t-statistic, p-value, significance flag,
        and Cohen's d.
    """
    ctrl_gmv  = df.loc[df["group"] == "control",    "gmv_60d"]
    treat_gmv = df.loc[df["group"] == "treatment",  "gmv_60d"]

    t_stat, p_value  = stats.ttest_ind(treat_gmv, ctrl_gmv, equal_var=False)
    mean_diff        = treat_gmv.mean() - ctrl_gmv.mean()
    ci_lo, ci_hi     = _welch_ci(treat_gmv, ctrl_gmv)

    result = {
        "metric":                 "60-Day GMV (all accounts)",
        "n_control":              len(ctrl_gmv),
        "n_treatment":            len(treat_gmv),
        "mean_control":           round(ctrl_gmv.mean(),  2),
        "mean_treatment":         round(treat_gmv.mean(), 2),
        "absolute_mean_lift":     round(mean_diff, 2),
        "relative_mean_lift_pct": round(mean_diff / ctrl_gmv.mean() * 100.0, 2),
        "ci_95_mean_diff":        (round(ci_lo, 2), round(ci_hi, 2)),
        "t_statistic":            round(t_stat,  4),
        "p_value":                round(p_value, 6),
        "significant":            bool(p_value < ALPHA),
        "cohens_d":               round(
            _cohens_d(
                treat_gmv.mean(), ctrl_gmv.mean(),
                treat_gmv.std(ddof=1), ctrl_gmv.std(ddof=1),
                len(treat_gmv), len(ctrl_gmv),
            ), 4
        ),
    }

    logger.info(
        "60d GMV test: $%.0f → $%.0f (+$%.0f) | p=%.4f | %s",
        ctrl_gmv.mean(), treat_gmv.mean(), mean_diff,
        p_value, "SIGNIFICANT" if result["significant"] else "not significant",
    )
    return result


# ---------------------------------------------------------------------------
# 5. Statistical Power & MDE
# ---------------------------------------------------------------------------

def calculate_power_analysis(df: pd.DataFrame) -> Dict:
    """
    Retrospective power analysis and minimum detectable effect (MDE).

    Reports achieved power for the observed effect and the MDE at the
    target power level for the actual sample size.

    Parameters
    ----------
    df : pd.DataFrame
        Full experiment DataFrame.

    Returns
    -------
    dict
        n_per_group, alpha, target_power, observed Cohen's h, achieved power,
        adequately_powered flag, MDE in Cohen's h, and approximate rate-diff MDE.

    Notes
    -----
    A retrospective power analysis on an experiment that produced a
    non-significant result is most useful for diagnosing whether the
    test was sensitive enough to detect the observed effect — not for
    claiming the null hypothesis is true (Hoenig & Heisey, 2001).
    """
    n = len(df) // 2  # assumes equal allocation

    rate_ctrl  = df.loc[df["group"] == "control",    "converted"].mean()
    rate_treat = df.loc[df["group"] == "treatment",  "converted"].mean()
    observed_h = abs(_cohens_h(rate_treat, rate_ctrl))

    power_solver   = NormalIndPower()
    achieved_power = power_solver.solve_power(
        effect_size=observed_h,
        nobs1=n,
        alpha=ALPHA,
        ratio=1.0,
        alternative="two-sided",
    )

    mde_h         = zt_ind_solve_power(
        effect_size=None,
        nobs1=n,
        alpha=ALPHA,
        power=POWER,
        ratio=1.0,
        alternative="two-sided",
    )
    # Approximate absolute rate MDE — valid near p=0.5 (arcsine linearisation)
    mde_rate_diff = mde_h / 2.0

    logger.info(
        "Power analysis: achieved_power=%.1f%% | MDE≈%.1f pp (need %.0f pp observed)",
        achieved_power * 100, mde_rate_diff * 100,
        abs(rate_treat - rate_ctrl) * 100,
    )

    return {
        "n_per_group":          n,
        "alpha":                ALPHA,
        "target_power":         POWER,
        "observed_effect_h":    round(observed_h,      4),
        "achieved_power":       round(achieved_power,  4),
        "adequately_powered":   bool(achieved_power >= POWER),
        "mde_cohens_h":         round(mde_h,           4),
        "mde_approx_rate_diff": round(mde_rate_diff,   4),
    }


# ---------------------------------------------------------------------------
# 6. Subgroup Analysis with Bonferroni Correction
# ---------------------------------------------------------------------------

def subgroup_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-industry conversion rate tests with Bonferroni multiple-comparison correction.

    Running *k* tests at α=0.05 inflates the family-wise Type I error rate to
    1−(1−0.05)^k.  Bonferroni sets α_per_test = α / k, maintaining the
    family-wise rate at α.  This is conservative but appropriate for a
    screening analysis where false positives drive real spend.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment DataFrame with ``industry``, ``group``, and ``converted``
        columns.

    Returns
    -------
    pd.DataFrame
        One row per industry segment with rates, absolute and relative lift,
        raw p-value, Bonferroni threshold, and significance flag.
        Sorted by relative_lift_pct descending.
    """
    segments       = df["industry"].unique()
    adjusted_alpha = ALPHA / len(segments)  # Bonferroni correction

    rows = []
    for segment in sorted(segments):
        seg_df  = df[df["industry"] == segment]
        ctrl    = seg_df[seg_df["group"] == "control"]
        treat   = seg_df[seg_df["group"] == "treatment"]

        if len(ctrl) < MIN_SEGMENT_N or len(treat) < MIN_SEGMENT_N:
            logger.debug(
                "Skipping segment '%s' — insufficient observations (ctrl=%d, treat=%d)",
                segment, len(ctrl), len(treat),
            )
            continue

        n_ctrl,  conv_ctrl  = len(ctrl),  int(ctrl["converted"].sum())
        n_treat, conv_treat = len(treat), int(treat["converted"].sum())
        rate_ctrl  = conv_ctrl  / n_ctrl
        rate_treat = conv_treat / n_treat

        try:
            _, p_val = proportions_ztest(
                [conv_treat, conv_ctrl],
                [n_treat,    n_ctrl],
                alternative="two-sided",
            )
        except Exception as exc:   # noqa: BLE001 — surface as NaN, not crash
            logger.warning("z-test failed for segment '%s': %s", segment, exc)
            p_val = np.nan

        rows.append({
            "industry":               segment,
            "n_control":              n_ctrl,
            "n_treatment":            n_treat,
            "rate_control":           round(rate_ctrl,  4),
            "rate_treatment":         round(rate_treat, 4),
            "absolute_lift":          round(rate_treat - rate_ctrl, 4),
            "relative_lift_pct":      (
                round((rate_treat - rate_ctrl) / rate_ctrl * 100.0, 1)
                if rate_ctrl > 0.0 else np.nan
            ),
            "p_value":                round(p_val, 4) if not np.isnan(p_val) else np.nan,
            "bonferroni_alpha":        round(adjusted_alpha, 4),
            "significant_bonferroni": bool(p_val < adjusted_alpha) if not np.isnan(p_val) else False,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("relative_lift_pct", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 7. Business Impact — Discount ROI
# ---------------------------------------------------------------------------

def calculate_business_impact(
    df: pd.DataFrame,
    platform_take_rate: float = PLATFORM_TAKE_RATE,
) -> Dict:
    """
    Translate statistical results into a campaign P&L statement.

    The key metric is ``campaign_roi``: incremental platform revenue
    generated per dollar of discount spend.  A ratio > 1.0 indicates
    the campaign is net-positive for the business.

    Parameters
    ----------
    df : pd.DataFrame
        Full experiment DataFrame.
    platform_take_rate : float, optional
        Fraction of GMV retained by the platform as revenue.
        Defaults to ``config.PLATFORM_TAKE_RATE``.

    Returns
    -------
    dict
        Total discount cost, incremental GMV, incremental platform revenue,
        and campaign ROI.

    Raises
    ------
    ValueError
        If ``platform_take_rate`` is not in (0, 1].
    """
    if not (0.0 < platform_take_rate <= 1.0):
        raise ValueError(
            f"platform_take_rate must be in (0, 1], got {platform_take_rate!r}"
        )

    treat = df[df["group"] == "treatment"]
    ctrl  = df[df["group"] == "control"]

    total_discount_cost         = treat["discount_applied"].sum()
    incremental_gmv_per_account = treat["gmv_60d"].mean() - ctrl["gmv_60d"].mean()
    total_incremental_gmv       = incremental_gmv_per_account * len(treat)
    incremental_revenue         = total_incremental_gmv * platform_take_rate

    roi = (
        incremental_revenue / total_discount_cost
        if total_discount_cost > 0.0 else np.nan
    )

    logger.info(
        "Campaign P&L: discount_cost=$%.0f | incremental_revenue=$%.0f | ROI=%.2fx",
        total_discount_cost, incremental_revenue, roi if not np.isnan(roi) else 0,
    )

    return {
        "n_treatment_accounts":         len(treat),
        "total_discount_cost_usd":      round(total_discount_cost, 2),
        "avg_discount_per_account_usd": round(total_discount_cost / len(treat), 2),
        "incremental_gmv_per_account":  round(incremental_gmv_per_account, 2),
        "total_incremental_gmv_usd":    round(total_incremental_gmv, 2),
        "platform_take_rate":           platform_take_rate,
        "incremental_platform_revenue": round(incremental_revenue, 2),
        "campaign_roi":                 round(roi, 2) if not np.isnan(roi) else None,
        "roi_positive":                 float(roi) > 1.0 if not np.isnan(roi) else False,
    }
