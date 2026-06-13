"""
experiment_design.py
====================
Prospective experiment design tools — the work done BEFORE an experiment
launches, not after.

This is the critical distinction the retrospective power analysis misses:
- Retrospective: "How much power did we have?" (after the fact)
- Prospective:   "How many accounts do we need?" (before launch)

At a marketplace like ezCater, an analyst is expected to:
  1. Define primary and secondary metrics *before* touching data
  2. Calculate the required sample size given a target MDE and power
  3. Estimate how long the experiment needs to run based on traffic
  4. Define the segment cuts (all users, power users, new accounts, etc.)
  5. Document this in an experiment spec that stakeholders sign off on

This module covers all five.

References
----------
Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled
Experiments*. Cambridge University Press. (The industry bible.)
Deng, A., Lu, J., & Chen, S. (2016). Continuous monitoring of A/B tests
without pain. *IEEE ICDM*.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower, zt_ind_solve_power

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment Specification — the pre-registration document
# ---------------------------------------------------------------------------

@dataclass
class ExperimentSpec:
    """
    Formal experiment specification — filled out BEFORE any data is collected.

    Pre-registration forces analysts to commit to metrics and hypotheses
    before seeing results, which is the primary defence against p-hacking
    and HARKing (Hypothesising After Results are Known).

    Parameters
    ----------
    name           : Unique experiment identifier (e.g., 'SPRING20').
    hypothesis     : The causal mechanism being tested in plain English.
    unit           : The randomisation unit ('account', 'order', 'session').
    control_desc   : What control group receives.
    treatment_desc : What treatment group receives.
    primary_metric : The ONE metric the ship/no-ship decision hinges on.
                     Only one — multiple primary metrics inflate Type I error.
    secondary_metrics : Supporting metrics that provide context but do NOT
                        drive the decision on their own.
    guardrail_metrics : Metrics that must NOT degrade. If any guardrail
                        moves negatively the experiment is stopped regardless
                        of primary metric results.
    segment_cuts   : Planned subgroup analyses. Must be defined upfront.
                     Post-hoc segment discovery is p-hacking.
    """
    name:              str
    hypothesis:        str
    unit:              str
    control_desc:      str
    treatment_desc:    str
    primary_metric:    str
    secondary_metrics: List[str]   = field(default_factory=list)
    guardrail_metrics: List[str]   = field(default_factory=list)
    segment_cuts:      List[str]   = field(default_factory=list)
    alpha:             float       = 0.05
    power:             float       = 0.80
    min_detectable_effect: Optional[float] = None

    def summary(self) -> str:
        """Return a formatted experiment brief for stakeholder sign-off."""
        lines = [
            f"EXPERIMENT SPEC: {self.name}",
            "=" * 55,
            f"Hypothesis     : {self.hypothesis}",
            f"Unit           : {self.unit}",
            f"Control        : {self.control_desc}",
            f"Treatment      : {self.treatment_desc}",
            "",
            f"PRIMARY METRIC (ship decision): {self.primary_metric}",
            "  α = {:.2f}  |  target power = {:.0%}  |  MDE = {}".format(
                self.alpha, self.power,
                f"{self.min_detectable_effect:+.1%}" if self.min_detectable_effect else "TBD"
            ),
            "",
            "SECONDARY METRICS (context only, not ship criteria):",
        ]
        for m in self.secondary_metrics:
            lines.append(f"  • {m}")
        lines.append("")
        lines.append("GUARDRAIL METRICS (auto-stop if any degrade):")
        for m in self.guardrail_metrics:
            lines.append(f"  • {m}")
        lines.append("")
        lines.append("PRE-PLANNED SEGMENT CUTS:")
        for s in self.segment_cuts:
            lines.append(f"  • {s}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sample Size Calculator — proportions (conversion rate)
# ---------------------------------------------------------------------------

def required_sample_size_proportions(
    baseline_rate:      float,
    min_detectable_effect: float,
    alpha:              float = 0.05,
    power:              float = 0.80,
    alternative:        str   = "two-sided",
    allocation_ratio:   float = 1.0,
) -> Dict:
    """
    Calculate the required sample size per group for a conversion rate test.

    This is the core experiment sizing calculation. You run this BEFORE
    launching, then wait until you have enough accounts before reading results.
    Reading too early inflates Type I error (peeking problem).

    Parameters
    ----------
    baseline_rate          : Current conversion rate (control).
                             Source: historical data from your data warehouse.
    min_detectable_effect  : Smallest absolute lift worth detecting.
                             Rule of thumb: set to the smallest effect that
                             would change a business decision.
    alpha                  : Significance level. 0.05 is standard; use 0.01
                             for high-stakes decisions or repeated testing.
    power                  : 1 - Type II error rate. 0.80 is the convention.
                             Use 0.90 when missing a real effect is costly.
    alternative            : 'two-sided' (default), 'larger', or 'smaller'.
    allocation_ratio       : n_treatment / n_control. 1.0 = equal split (optimal
                             for power); deviate only when treatment is costly.

    Returns
    -------
    dict with per-group sample size, total N, and sensitivity analysis
    across a range of MDE values.

    Examples
    --------
    >>> # ezCater scenario: 30-day conversion rate currently 25%.
    >>> # Want to detect a +3pp lift (10% relative) — is that worth detecting?
    >>> result = required_sample_size_proportions(
    ...     baseline_rate=0.25, min_detectable_effect=0.03
    ... )
    >>> print(result['n_per_group'])  # sample size needed per arm
    """
    if not (0.0 < baseline_rate < 1.0):
        raise ValueError(f"baseline_rate must be in (0, 1), got {baseline_rate}")
    if min_detectable_effect <= 0:
        raise ValueError(f"min_detectable_effect must be positive, got {min_detectable_effect}")

    treatment_rate = baseline_rate + min_detectable_effect

    # Cohen's h: effect size for proportions
    def cohens_h(p1: float, p2: float) -> float:
        return 2.0 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))

    h = abs(cohens_h(treatment_rate, baseline_rate))

    # Solve for sample size
    n_per_group = math.ceil(
        zt_ind_solve_power(
            effect_size=h,
            nobs1=None,
            alpha=alpha,
            power=power,
            ratio=allocation_ratio,
            alternative=alternative,
        )
    )
    n_total = math.ceil(n_per_group * (1 + allocation_ratio))

    # Sensitivity table: how does N change with different MDEs?
    sensitivity = []
    for mde in [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
        if baseline_rate + mde >= 1.0:
            continue
        h_i = abs(cohens_h(baseline_rate + mde, baseline_rate))
        try:
            n_i = math.ceil(
                zt_ind_solve_power(
                    effect_size=h_i,
                    nobs1=None,
                    alpha=alpha,
                    power=power,
                    ratio=allocation_ratio,
                    alternative=alternative,
                )
            )
            sensitivity.append({
                "mde_absolute":  mde,
                "mde_relative":  round(mde / baseline_rate, 3),
                "n_per_group":   n_i,
                "n_total":       math.ceil(n_i * (1 + allocation_ratio)),
            })
        except Exception:
            pass

    return {
        "baseline_rate":          baseline_rate,
        "min_detectable_effect":  min_detectable_effect,
        "treatment_rate":         round(treatment_rate, 4),
        "relative_mde":           round(min_detectable_effect / baseline_rate, 3),
        "cohens_h":               round(h, 4),
        "alpha":                  alpha,
        "power":                  power,
        "alternative":            alternative,
        "n_per_group":            n_per_group,
        "n_total":                n_total,
        "sensitivity_table":      sensitivity,
    }


# ---------------------------------------------------------------------------
# Sample Size Calculator — means (GMV, revenue)
# ---------------------------------------------------------------------------

def required_sample_size_means(
    baseline_mean:       float,
    baseline_std:        float,
    min_detectable_effect: float,
    alpha:               float = 0.05,
    power:               float = 0.80,
    alternative:         str   = "two-sided",
) -> Dict:
    """
    Calculate required sample size for a difference-in-means test (GMV, AOV).

    Use this for continuous metrics like average order value or 60-day GMV.
    Requires an estimate of the standard deviation — use historical data.

    Parameters
    ----------
    baseline_mean          : Current mean value (from historical data).
    baseline_std           : Standard deviation of the metric (from historical data).
    min_detectable_effect  : Absolute difference in means worth detecting.
    """
    if baseline_std <= 0:
        raise ValueError(f"baseline_std must be positive, got {baseline_std}")

    # Cohen's d
    d = min_detectable_effect / baseline_std

    solver = NormalIndPower()
    n_per_group = math.ceil(
        solver.solve_power(
            effect_size=d,
            nobs1=None,
            alpha=alpha,
            power=power,
            ratio=1.0,
            alternative=alternative,
        )
    )

    return {
        "baseline_mean":          baseline_mean,
        "baseline_std":           baseline_std,
        "min_detectable_effect":  min_detectable_effect,
        "relative_mde":           round(min_detectable_effect / baseline_mean, 3),
        "cohens_d":               round(d, 4),
        "alpha":                  alpha,
        "power":                  power,
        "n_per_group":            n_per_group,
        "n_total":                n_per_group * 2,
    }


# ---------------------------------------------------------------------------
# Runtime Calculator — how long does the experiment need to run?
# ---------------------------------------------------------------------------

def experiment_runtime_days(
    n_required:       int,
    daily_eligible:   int,
    allocation_pct:   float = 1.0,
    min_runtime_days: int   = 14,
) -> Dict:
    """
    Estimate how many days the experiment needs to run.

    Parameters
    ----------
    n_required      : Total accounts needed (from required_sample_size_*).
    daily_eligible  : How many eligible accounts are acquired per day
                      (from historical data in your warehouse).
    allocation_pct  : Fraction of eligible traffic enrolled (0.0–1.0).
                      Use < 1.0 to ramp up gradually or reserve a holdout.
    min_runtime_days: Minimum run time regardless of N (captures weekly
                      seasonality — always run at least 2 full weeks).

    Notes
    -----
    Two-week minimum is not arbitrary: B2B catering orders are weekly-cadence.
    Running for less than 2 weeks risks day-of-week confounding — Tuesday
    office lunches behave very differently from Friday team meals.
    """
    if daily_eligible <= 0:
        raise ValueError("daily_eligible must be positive")
    if not (0.0 < allocation_pct <= 1.0):
        raise ValueError("allocation_pct must be in (0, 1]")

    daily_enrolled  = daily_eligible * allocation_pct
    days_for_n      = math.ceil(n_required / daily_enrolled)
    recommended_days = max(days_for_n, min_runtime_days)

    # Round up to complete weeks (avoids day-of-week bias)
    weeks = math.ceil(recommended_days / 7)
    recommended_days_rounded = weeks * 7

    return {
        "n_required":              n_required,
        "daily_eligible":          daily_eligible,
        "daily_enrolled":          round(daily_enrolled),
        "allocation_pct":          allocation_pct,
        "days_to_reach_n":         days_for_n,
        "min_runtime_days":        min_runtime_days,
        "recommended_days":        recommended_days_rounded,
        "recommended_weeks":       weeks,
        "note": (
            f"Run for {recommended_days_rounded} days ({weeks} complete weeks) "
            f"to ensure full weekly seasonality is captured."
        ),
    }


# ---------------------------------------------------------------------------
# Power User Segment Definition
# ---------------------------------------------------------------------------

def define_power_users(
    orders_threshold:    int   = 4,
    lookback_days:       int   = 90,
    gmv_threshold:       Optional[float] = None,
) -> Dict:
    """
    Return the SQL WHERE clause and definition for 'power users'.

    Power users are high-frequency / high-value accounts. Analysing experiment
    results within this segment answers: 'Does the treatment work differently
    for our best customers?' — which is often the most important business question.

    Parameters
    ----------
    orders_threshold : Minimum orders in the lookback window to qualify.
    lookback_days    : How far back to measure ordering history.
    gmv_threshold    : Optional minimum lifetime GMV to qualify as power user.

    Returns a dict containing the segment definition and SQL snippet for
    use in Snowflake queries.
    """
    conditions = [
        f"orders_last_{lookback_days}d >= {orders_threshold}"
    ]
    if gmv_threshold:
        conditions.append(f"lifetime_gmv >= {gmv_threshold}")

    sql_where = " AND ".join(conditions)

    sql_snippet = f"""
-- Power user segment filter
-- Apply this WHERE clause to restrict analysis to high-frequency accounts
-- Definition: accounts with >= {orders_threshold} orders in the last {lookback_days} days
WHERE {sql_where}
"""
    return {
        "segment_name":       "Power Users",
        "orders_threshold":   orders_threshold,
        "lookback_days":      lookback_days,
        "gmv_threshold":      gmv_threshold,
        "definition":         f">= {orders_threshold} orders in last {lookback_days} days",
        "sql_where_clause":   sql_where,
        "sql_snippet":        sql_snippet,
    }


# ---------------------------------------------------------------------------
# Full Experiment Design Report
# ---------------------------------------------------------------------------

def generate_experiment_brief(
    spec:               ExperimentSpec,
    baseline_conv_rate: float,
    baseline_gmv_mean:  float,
    baseline_gmv_std:   float,
    mde_conv:           float,
    mde_gmv:            float,
    daily_new_accounts: int,
) -> str:
    """
    Generate a full pre-experiment brief — the document you share with
    stakeholders before launching.

    This combines: metric registration, sample size, runtime, and
    segment definitions into one artifact.
    """
    conv_sizing = required_sample_size_proportions(
        baseline_rate=baseline_conv_rate,
        min_detectable_effect=mde_conv,
        alpha=spec.alpha,
        power=spec.power,
    )
    gmv_sizing = required_sample_size_means(
        baseline_mean=baseline_gmv_mean,
        baseline_std=baseline_gmv_std,
        min_detectable_effect=mde_gmv,
        alpha=spec.alpha,
        power=spec.power,
    )
    # Binding constraint: must satisfy both primary and secondary metric sizing
    n_required = max(conv_sizing["n_total"], gmv_sizing["n_total"])

    runtime = experiment_runtime_days(
        n_required=n_required,
        daily_eligible=daily_new_accounts,
    )
    power_users = define_power_users()

    lines = [
        spec.summary(),
        "",
        "─" * 55,
        "SAMPLE SIZE REQUIREMENTS",
        "─" * 55,
        f"Primary metric (conversion rate, baseline={baseline_conv_rate:.1%}):",
        f"  MDE = {mde_conv:+.1%} absolute ({mde_conv/baseline_conv_rate:.0%} relative)",
        f"  Required n per group: {conv_sizing['n_per_group']:,}",
        f"  Required n total:     {conv_sizing['n_total']:,}",
        "",
        f"Secondary metric (60d GMV, baseline=${baseline_gmv_mean:,.0f}, σ=${baseline_gmv_std:,.0f}):",
        f"  MDE = ${mde_gmv:,.0f} absolute ({mde_gmv/baseline_gmv_mean:.0%} relative)",
        f"  Required n per group: {gmv_sizing['n_per_group']:,}",
        "",
        f"Binding constraint: {n_required:,} total accounts",
        "",
        "─" * 55,
        "EXPERIMENT RUNTIME",
        "─" * 55,
        f"Daily eligible new accounts: {daily_new_accounts:,}",
        f"Days to reach required N:    {runtime['days_to_reach_n']}",
        f"Recommended runtime:         {runtime['recommended_days']} days ({runtime['recommended_weeks']} weeks)",
        f"Note: {runtime['note']}",
        "",
        "─" * 55,
        "SEGMENT CUTS (PLANNED)",
        "─" * 55,
    ]
    for seg in spec.segment_cuts:
        lines.append(f"  • {seg}")
    lines += [
        "",
        f"Power user definition: {power_users['definition']}",
        f"SQL filter: {power_users['sql_where_clause']}",
        "",
        "─" * 55,
        "MDE SENSITIVITY TABLE (conversion rate)",
        "─" * 55,
        f"  {'MDE (abs)':>10}  {'MDE (rel)':>10}  {'n/group':>10}  {'n total':>10}",
    ]
    for row in conv_sizing["sensitivity_table"]:
        lines.append(
            f"  {row['mde_absolute']:>+10.1%}  {row['mde_relative']:>+10.0%}  "
            f"{row['n_per_group']:>10,}  {row['n_total']:>10,}"
        )
    lines.append("")
    lines.append("STATUS: DRAFT — Awaiting stakeholder sign-off before experiment launch.")
    return "\n".join(lines)
