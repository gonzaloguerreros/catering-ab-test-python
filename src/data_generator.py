"""
data_generator.py
=================
Generates a synthetic A/B test dataset for a promotional campaign on a
B2B catering marketplace.

Experiment design
-----------------
- Platform:     Corporate catering marketplace (ezCater-style)
- Campaign:     "SPRING20" — 20% off first order for newly acquired mid-market
                corporate accounts. Objective: increase conversion rate and
                drive a higher first-order value.
- Test window:  March 1 – May 31, 2024  (92 days)
- Unit:         Corporate account (account-level randomisation, not order-level,
                to avoid within-account contamination)
- Control:      No discount — standard onboarding experience
- Treatment:    20% discount on the account's first order
- Primary KPI:  Conversion rate (placed first order within 30 days of signup)
- Secondary KPIs: First-order GMV, 60-day GMV, orders in first 60 days

Assumptions baked into the simulation
--------------------------------------
- True conversion rate lift:  +8 percentage points (control ~52%, treatment ~60%)
- True AOV lift:              +12% on first order (treatment orders slightly larger
                              because the discount lowers the psychological barrier
                              to ordering more food)
- True 60-day GMV uplift:     +9% (diminishes over time as promo effect fades)
- Segment heterogeneity:      Technology accounts respond more strongly; Finance
                              accounts show little effect (robustness check).
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import random

# ---------------------------------------------------------------------------
# Reproducibility — fix seed so results are consistent across runs
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_CONTROL   = 250   # accounts assigned to control group
N_TREATMENT = 250   # accounts assigned to treatment group
TEST_START  = date(2024, 3, 1)
TEST_END    = date(2024, 5, 31)

# True underlying effect sizes (ground truth — available because this is
# synthetic data; in production these would be unknown)
TRUE_CONV_RATE_CONTROL   = 0.52    # 52% of control accounts place a first order
TRUE_CONV_RATE_TREATMENT = 0.60    # 60% of treatment accounts place a first order
TRUE_AOV_CONTROL_MEAN    = 420.0   # average first-order value, control ($)
TRUE_AOV_TREATMENT_MEAN  = 470.4   # ~12% lift in treatment ($)
TRUE_AOV_STD             = 140.0   # order value standard deviation (both groups)

# Segment-level heterogeneity in treatment effect (for subgroup analysis)
SEGMENT_CONV_LIFT = {
    "Technology":   0.12,   # strong responders
    "Finance":      0.02,   # weak responders
    "Healthcare":   0.08,
    "Consulting":   0.07,
    "Legal":        0.06,
    "Media":        0.09,
    "Other":        0.08,
}

INDUSTRIES = list(SEGMENT_CONV_LIFT.keys())


def _assign_signup_date(n: int) -> list:
    """Return n random signup dates uniformly spread across the test window."""
    days_in_window = (TEST_END - TEST_START).days
    return [
        TEST_START + timedelta(days=int(rng.integers(0, days_in_window)))
        for _ in range(n)
    ]


def _sample_first_order_value(group: str, n_converts: int) -> np.ndarray:
    """
    Draw first-order GMV from a right-skewed log-normal distribution.
    Log-normal is standard for purchase amounts: most orders are in the
    $200-$600 range with a long tail of large enterprise orders.
    """
    if group == "control":
        mu    = TRUE_AOV_CONTROL_MEAN
    else:
        mu    = TRUE_AOV_TREATMENT_MEAN

    # Convert (mean, std) of the underlying normal to log-normal params
    sigma = TRUE_AOV_STD
    mu_ln    = np.log(mu**2 / np.sqrt(sigma**2 + mu**2))
    sigma_ln = np.sqrt(np.log(1 + (sigma / mu) ** 2))

    return rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=n_converts).round(2)


def generate_experiment_data() -> pd.DataFrame:
    """
    Build the full account-level experiment dataframe.

    Returns
    -------
    pd.DataFrame
        One row per account.  Columns:
        account_id, group, industry, signup_date, converted,
        first_order_value, first_order_date, orders_60d, gmv_60d,
        discount_applied
    """
    records = []

    for group, n_accounts in [("control", N_CONTROL), ("treatment", N_TREATMENT)]:
        signup_dates = _assign_signup_date(n_accounts)

        for i in range(n_accounts):
            industry    = random.choice(INDUSTRIES)
            signup_date = signup_dates[i]

            # ------------------------------------------------------------------
            # Determine conversion (did the account place a first order?)
            # Treatment lift is modulated by industry segment
            # ------------------------------------------------------------------
            base_conv_rate = TRUE_CONV_RATE_CONTROL
            if group == "treatment":
                lift = SEGMENT_CONV_LIFT.get(industry, 0.08)
                conv_rate = min(base_conv_rate + lift, 0.99)
            else:
                conv_rate = base_conv_rate

            converted = bool(rng.random() < conv_rate)

            # ------------------------------------------------------------------
            # If converted, assign a first-order date within 30 days of signup
            # ------------------------------------------------------------------
            first_order_date  = None
            first_order_value = None
            discount_applied  = 0.0
            orders_60d        = 0
            gmv_60d           = 0.0

            if converted:
                days_to_first = int(rng.integers(1, 31))   # 1–30 days after signup
                first_order_date  = signup_date + timedelta(days=days_to_first)
                first_order_value = float(_sample_first_order_value(group, 1)[0])

                # Treatment accounts receive the 20% discount
                if group == "treatment":
                    discount_applied = round(first_order_value * 0.20, 2)

                # ------------------------------------------------------------------
                # Simulate repeat orders in the 60-day window post-signup
                # Treatment accounts reorder slightly more often (LTV halo effect)
                # ------------------------------------------------------------------
                reorder_prob = 0.55 if group == "treatment" else 0.48
                n_reorders   = int(rng.binomial(n=4, p=reorder_prob))
                orders_60d   = 1 + n_reorders   # first order + reorders

                # Reorder values drawn from the same distribution as control
                # (discount only applies to first order)
                if n_reorders > 0:
                    reorder_values = _sample_first_order_value("control", n_reorders)
                    gmv_60d = round(first_order_value + reorder_values.sum(), 2)
                else:
                    gmv_60d = first_order_value

            records.append({
                "account_id":         f"{group[0].upper()}{i+1:04d}",
                "group":              group,
                "industry":           industry,
                "signup_date":        signup_date,
                "converted":          int(converted),          # 1/0 for easy aggregation
                "first_order_date":   first_order_date,
                "first_order_value":  first_order_value,
                "discount_applied":   discount_applied,
                "orders_60d":         orders_60d,
                "gmv_60d":            gmv_60d,
            })

    df = pd.DataFrame(records)

    # Sort by signup_date for readability
    df = df.sort_values("signup_date").reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = generate_experiment_data()
    output_path = "data/experiment_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset generated: {len(df):,} accounts → {output_path}")
    print(df.groupby("group")[["converted", "first_order_value", "gmv_60d"]].mean().round(2))
