"""
tests/test_ab_test_analysis.py
===============================
Unit tests for the statistical analysis module.

Testing philosophy (Harvard CS109 / Google Engineering Practices)
-----------------------------------------------------------------
- Each test validates ONE behaviour.
- Tests use descriptive names that explain what they verify.
- Edge cases (zero variance, zero discount, SRM) are explicitly covered.
- Fixtures are isolated — tests do not share mutable state.

Run with::

    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from src/ without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ab_test_analysis import (
    _cohens_d,
    _cohens_h,
    calculate_business_impact,
    check_sample_ratio_mismatch,
    analyze_conversion_rate,
    analyze_first_order_gmv,
)
from config import ALPHA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def balanced_df() -> pd.DataFrame:
    """250 control + 250 treatment accounts with a known conversion lift."""
    rng = np.random.default_rng(0)
    n   = 250

    def _make_group(label: str, conv_rate: float, mean_val: float) -> pd.DataFrame:
        converted  = (rng.random(n) < conv_rate).astype(int)
        raw_vals   = rng.lognormal(np.log(mean_val), 0.3, n)
        # Use 0.0 for non-converters so arithmetic stays valid (no None in arrays)
        fov        = np.where(converted, raw_vals, 0.0)
        gmv        = np.where(converted, fov * 1.5, 0.0)
        discount   = np.where((converted == 1) & (label == "treatment"),
                              fov * 0.20, 0.0)
        # NaN for non-converters using standard float64 (object dtype breaks scipy)
        fov_with_nan = np.where(converted, fov, np.nan).astype(float)
        return pd.DataFrame({
            "group":             label,
            "converted":         converted,
            "first_order_value": fov_with_nan,
            "gmv_60d":           gmv.astype(float),
            "discount_applied":  discount.astype(float),
        })

    ctrl  = _make_group("control",   conv_rate=0.50, mean_val=400.0)
    treat = _make_group("treatment", conv_rate=0.62, mean_val=450.0)
    return pd.concat([ctrl, treat], ignore_index=True)


@pytest.fixture()
def srm_df() -> pd.DataFrame:
    """Heavily imbalanced split (80/20) that should trigger SRM."""
    ctrl  = pd.DataFrame({"group": ["control"]   * 400})
    treat = pd.DataFrame({"group": ["treatment"] * 100})
    return pd.concat([ctrl, treat], ignore_index=True)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestCohensH:
    def test_equal_proportions_returns_zero(self):
        assert _cohens_h(0.5, 0.5) == pytest.approx(0.0, abs=1e-9)

    def test_sign_reflects_direction(self):
        """Cohen's h should be positive when p1 > p2."""
        assert _cohens_h(0.6, 0.4) > 0.0

    def test_known_value(self):
        """h(0.6, 0.5) ≈ 0.201 (verified against Cohen 1988, Table 6.2.2)."""
        assert _cohens_h(0.6, 0.5) == pytest.approx(0.2013, abs=1e-3)


class TestCohensD:
    def test_equal_means_returns_zero(self):
        assert _cohens_d(100.0, 100.0, 20.0, 20.0, 50, 50) == pytest.approx(0.0)

    def test_direction_positive_when_mean1_greater(self):
        assert _cohens_d(120.0, 100.0, 20.0, 20.0, 50, 50) > 0.0

    def test_zero_pooled_std_returns_zero(self):
        """When all values are identical, d is undefined — return 0."""
        assert _cohens_d(5.0, 5.0, 0.0, 0.0, 10, 10) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SRM detection
# ---------------------------------------------------------------------------

class TestSampleRatioMismatch:
    def test_balanced_split_not_flagged(self, balanced_df):
        result = check_sample_ratio_mismatch(balanced_df)
        assert result["srm_detected"] is False

    def test_imbalanced_split_flagged(self, srm_df):
        result = check_sample_ratio_mismatch(srm_df)
        assert result["srm_detected"] is True

    def test_counts_are_correct(self, balanced_df):
        result = check_sample_ratio_mismatch(balanced_df)
        assert result["n_control"]   == 250
        assert result["n_treatment"] == 250

    def test_invalid_ratio_raises(self, balanced_df):
        with pytest.raises(ValueError, match="expected_ratio"):
            check_sample_ratio_mismatch(balanced_df, expected_ratio=1.5)


# ---------------------------------------------------------------------------
# Conversion rate test
# ---------------------------------------------------------------------------

class TestConversionRate:
    def test_detects_large_lift(self, balanced_df):
        """A 12 pp lift with n=250 per group should be detectable."""
        result = analyze_conversion_rate(balanced_df)
        assert result["significant"] is True

    def test_absolute_lift_sign_correct(self, balanced_df):
        """Treatment rate should exceed control rate in the fixture."""
        result = analyze_conversion_rate(balanced_df)
        assert result["absolute_lift"] > 0.0

    def test_ci_contains_absolute_lift(self, balanced_df):
        """95 % CI on the difference must bracket the point estimate."""
        result   = analyze_conversion_rate(balanced_df)
        lo, hi   = result["ci_95_diff"]
        lift     = result["absolute_lift"]
        assert lo <= lift <= hi

    def test_rates_are_probabilities(self, balanced_df):
        result = analyze_conversion_rate(balanced_df)
        assert 0.0 <= result["rate_control"]   <= 1.0
        assert 0.0 <= result["rate_treatment"] <= 1.0

    def test_p_value_in_valid_range(self, balanced_df):
        result = analyze_conversion_rate(balanced_df)
        assert 0.0 <= result["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# First-order GMV test
# ---------------------------------------------------------------------------

class TestFirstOrderGmv:
    def test_returns_expected_keys(self, balanced_df):
        required = {
            "mean_control", "mean_treatment", "absolute_mean_lift",
            "p_value_welch", "significant_welch", "cohens_d",
        }
        result = analyze_first_order_gmv(balanced_df)
        assert required.issubset(result.keys())

    def test_mean_lift_positive(self, balanced_df):
        """Treatment mean GMV should be higher in the fixture."""
        result = analyze_first_order_gmv(balanced_df)
        assert result["absolute_mean_lift"] > 0.0

    def test_welch_and_mann_whitney_agree_direction(self, balanced_df):
        """Both tests should agree on significance direction for a clear effect."""
        result = analyze_first_order_gmv(balanced_df)
        assert result["significant_welch"] == result["significant_mann_whitney"]


# ---------------------------------------------------------------------------
# Business impact
# ---------------------------------------------------------------------------

class TestBusinessImpact:
    def test_roi_is_numeric(self, balanced_df):
        result = calculate_business_impact(balanced_df)
        assert isinstance(result["campaign_roi"], (float, type(None)))

    def test_invalid_take_rate_raises(self, balanced_df):
        with pytest.raises(ValueError, match="platform_take_rate"):
            calculate_business_impact(balanced_df, platform_take_rate=1.5)

    def test_discount_cost_is_non_negative(self, balanced_df):
        result = calculate_business_impact(balanced_df)
        assert result["total_discount_cost_usd"] >= 0.0
