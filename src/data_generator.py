"""
data_generator.py
=================
Generates a synthetic, reproducible A/B test dataset for the SPRING20
promotional campaign on a B2B catering marketplace.

Experiment design
-----------------
- Platform    : Corporate catering marketplace (ezCater-style B2B).
- Campaign    : "SPRING20" — 20 % off first order for newly acquired
                mid-market corporate accounts.
- Objective   : Increase 30-day conversion rate and first-order GMV.
- Test window : March 1 – May 31, 2024 (92 days).
- Unit        : Corporate account (account-level randomisation prevents
                within-account spillover contamination).
- Control     : Standard onboarding — no discount.
- Treatment   : 20 % discount applied to the account's first order.

Primary KPI   : Conversion rate — placed first order within 30 days.
Secondary KPIs: First-order GMV, 60-day GMV, orders in first 60 days.

Simulation ground-truth (transparent because the data is synthetic)
-------------------------------------------------------------------
- Conversion lift : +8 pp (control 52 %, treatment 60 %).
- AOV lift        : +12 % on first order (discount lowers quantity friction).
- 60-day GMV lift : +9 % (halo effect diminishes as promo effect fades).
- Heterogeneity   : Technology accounts respond more strongly; Finance
                    accounts show near-zero lift (robustness check).

References
----------
Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled
Experiments*. Cambridge University Press.
"""

from __future__ import annotations

import logging
import random
from datetime import date, timedelta
from typing import List

import numpy as np
import pandas as pd

from config import (
    DISCOUNT_RATE,
    N_CONTROL,
    N_TREATMENT,
    RANDOM_SEED,
    SEGMENT_CONV_LIFT,
    TEST_END,
    TEST_START,
    TRUE_AOV_CONTROL_MEAN,
    TRUE_AOV_STD,
    TRUE_AOV_TREATMENT_MEAN,
    TRUE_CONV_RATE_CONTROL,
)

# ---------------------------------------------------------------------------
# Module-level logger — callers configure the handler; this module only emits
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility — seeded at module level so every import gives the same data
# ---------------------------------------------------------------------------
_rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)

INDUSTRIES: List[str] = list(SEGMENT_CONV_LIFT.keys())


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _assign_signup_dates(n: int) -> List[date]:
    """
    Return *n* signup dates drawn uniformly over the test window.

    Parameters
    ----------
    n : int
        Number of dates to generate.  Must be > 0.

    Returns
    -------
    List[date]
        List of ``datetime.date`` objects within [TEST_START, TEST_END).

    Raises
    ------
    ValueError
        If *n* is not a positive integer.
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n!r}")

    days_in_window = (TEST_END - TEST_START).days
    return [
        TEST_START + timedelta(days=int(_rng.integers(0, days_in_window)))
        for _ in range(n)
    ]


def _sample_order_values(group: str, n: int) -> np.ndarray:
    """
    Draw *n* order values from a log-normal distribution.

    Log-normal is the standard model for purchase amounts: it is right-
    skewed (a few very large corporate orders), strictly positive, and
    its parameters are directly interpretable (Crow & Shimizu, 1988).

    Parameters
    ----------
    group : {'control', 'treatment'}
        Determines the mean of the underlying log-normal.
    n : int
        Number of samples.

    Returns
    -------
    np.ndarray
        Array of order values rounded to 2 decimal places.

    Raises
    ------
    ValueError
        If *group* is not 'control' or 'treatment'.
    """
    valid_groups = {"control", "treatment"}
    if group not in valid_groups:
        raise ValueError(f"group must be one of {valid_groups}, got {group!r}")

    mu = TRUE_AOV_CONTROL_MEAN if group == "control" else TRUE_AOV_TREATMENT_MEAN
    sigma = TRUE_AOV_STD

    # Convert arithmetic (mean, std) → log-normal (mu_ln, sigma_ln)
    # Formula: mu_ln = log(mu² / sqrt(sigma² + mu²))
    mu_ln    = np.log(mu ** 2 / np.sqrt(sigma ** 2 + mu ** 2))
    sigma_ln = np.sqrt(np.log(1 + (sigma / mu) ** 2))

    return _rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=n).round(2)


def _build_account_record(
    group: str,
    index: int,
    industry: str,
    signup_date: date,
) -> dict:
    """
    Simulate a single account's experiment outcome.

    Parameters
    ----------
    group      : 'control' or 'treatment'.
    index      : Account index within the group (used to build account_id).
    industry   : Industry segment label.
    signup_date: Date the account signed up.

    Returns
    -------
    dict
        Account-level record with all outcome fields.
    """
    # --- Conversion probability -------------------------------------------
    conv_rate = TRUE_CONV_RATE_CONTROL
    if group == "treatment":
        lift = SEGMENT_CONV_LIFT.get(industry, 0.08)
        conv_rate = min(conv_rate + lift, 0.99)  # cap at 99 % to stay realistic

    converted = bool(_rng.random() < conv_rate)

    # --- Default outcome values (non-converters) -------------------------
    first_order_date:  date | None  = None
    first_order_value: float | None = None
    discount_applied:  float        = 0.0
    orders_60d:        int          = 0
    gmv_60d:           float        = 0.0

    if converted:
        # First order placed 1–30 days after signup
        days_to_first     = int(_rng.integers(1, 31))
        first_order_date  = signup_date + timedelta(days=days_to_first)
        first_order_value = float(_sample_order_values(group, 1)[0])

        if group == "treatment":
            discount_applied = round(first_order_value * DISCOUNT_RATE, 2)

        # Repeat orders within 60 days — treatment shows a small LTV halo
        reorder_prob = 0.55 if group == "treatment" else 0.48
        n_reorders   = int(_rng.binomial(n=4, p=reorder_prob))
        orders_60d   = 1 + n_reorders

        if n_reorders > 0:
            reorder_values = _sample_order_values("control", n_reorders)
            gmv_60d = round(first_order_value + reorder_values.sum(), 2)
        else:
            gmv_60d = first_order_value

    return {
        "account_id":        f"{group[0].upper()}{index + 1:04d}",
        "group":             group,
        "industry":          industry,
        "signup_date":       signup_date,
        "converted":         int(converted),        # 1/0 for easy aggregation
        "first_order_date":  first_order_date,
        "first_order_value": first_order_value,
        "discount_applied":  discount_applied,
        "orders_60d":        orders_60d,
        "gmv_60d":           gmv_60d,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_experiment_data() -> pd.DataFrame:
    """
    Build the full account-level experiment DataFrame.

    Generates accounts for both control and treatment arms, simulates
    conversion and revenue outcomes, and returns a tidy DataFrame sorted
    by signup date.

    Returns
    -------
    pd.DataFrame
        One row per account.  Schema:

        =========================================================
        Column             Dtype       Description
        -----------------  ----------  --------------------------
        account_id         str         Unique account identifier
        group              str         'control' | 'treatment'
        industry           str         Industry segment
        signup_date        date        Date account was acquired
        converted          int         1 if placed first order ≤ 30 d
        first_order_date   date|None   Date of first order, if any
        first_order_value  float|None  GMV of first order ($)
        discount_applied   float       Discount $ given (treatment only)
        orders_60d         int         Total orders in 60-day window
        gmv_60d            float       Total GMV in 60-day window ($)
        =========================================================

    Examples
    --------
    >>> df = generate_experiment_data()
    >>> len(df)
    500
    >>> set(df["group"].unique()) == {"control", "treatment"}
    True
    """
    logger.info(
        "Generating experiment data: %d control + %d treatment accounts",
        N_CONTROL,
        N_TREATMENT,
    )

    records = []
    for group, n_accounts in (("control", N_CONTROL), ("treatment", N_TREATMENT)):
        signup_dates = _assign_signup_dates(n_accounts)
        for i in range(n_accounts):
            record = _build_account_record(
                group=group,
                index=i,
                industry=random.choice(INDUSTRIES),
                signup_date=signup_dates[i],
            )
            records.append(record)

    df = (
        pd.DataFrame(records)
        .sort_values("signup_date")
        .reset_index(drop=True)
    )

    logger.info(
        "Dataset ready: %d accounts | conversion rates: control=%.1f%%, treatment=%.1f%%",
        len(df),
        df.loc[df["group"] == "control",    "converted"].mean() * 100,
        df.loc[df["group"] == "treatment",  "converted"].mean() * 100,
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from config import DATA_DIR

    df = generate_experiment_data()
    out = DATA_DIR / "experiment_data.csv"
    df.to_csv(out, index=False)
    logger.info("Saved to %s", out)
    print(df.groupby("group")[["converted", "first_order_value", "gmv_60d"]].mean().round(2))
