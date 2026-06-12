"""
config.py
=========
Single source of truth for all experiment parameters, statistical thresholds,
and file paths used across the catering A/B test analysis package.

Why a dedicated config module?
-------------------------------
Scattering magic numbers (0.05, 250, 0.20) across multiple source files is
a well-documented anti-pattern (Martin, *Clean Code*, 2008; Google Python
Style Guide §2.7).  When a parameter changes — e.g., alpha moves from 0.05
to 0.01 for a high-stakes decision — you change it in one place and the
entire pipeline updates consistently.

All paths use ``pathlib.Path`` (PEP 428) so the package runs correctly
regardless of the working directory from which it is invoked.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root — all other paths are derived from this
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path    = ROOT_DIR / "data"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"

# Ensure output directories exist when this module is imported
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment design parameters
# ---------------------------------------------------------------------------

#: Number of accounts assigned to each arm of the experiment.
#: Equal allocation maximises statistical power for a fixed total N
#: (Kiefer & Wolfowitz, 1952).
N_CONTROL: int   = 250
N_TREATMENT: int = 250

#: Test window — 92 days covering a single quarter.
TEST_START: date = date(2024, 3, 1)
TEST_END: date   = date(2024, 5, 31)

#: Promo discount rate applied to the treatment arm's first order.
DISCOUNT_RATE: float = 0.20

# ---------------------------------------------------------------------------
# Ground-truth simulation parameters
# (only meaningful because the dataset is synthetic; unknown in production)
# ---------------------------------------------------------------------------

#: True underlying conversion rates per arm.
TRUE_CONV_RATE_CONTROL: float   = 0.52
TRUE_CONV_RATE_TREATMENT: float = 0.60

#: Log-normal parameters for first-order GMV distribution.
TRUE_AOV_CONTROL_MEAN: float   = 420.0   # USD
TRUE_AOV_TREATMENT_MEAN: float = 470.4   # ~12 % lift
TRUE_AOV_STD: float            = 140.0

#: Industry-level heterogeneity in treatment effect (for subgroup analysis).
SEGMENT_CONV_LIFT: dict[str, float] = {
    "Technology": 0.12,   # strong responders — larger budgets, faster decisions
    "Finance":    0.02,   # weak responders — procurement-driven, price-insensitive
    "Healthcare": 0.08,
    "Consulting": 0.07,
    "Legal":      0.06,
    "Media":      0.09,
    "Other":      0.08,
}

# ---------------------------------------------------------------------------
# Statistical thresholds
# ---------------------------------------------------------------------------

#: Significance level (Type I error rate).  Industry standard is 0.05;
#: tighten to 0.01 for high-stakes or repeated-testing scenarios.
ALPHA: float = 0.05

#: Desired statistical power (1 − Type II error rate).
#: 0.80 is the conventional minimum (Cohen, 1988).
POWER: float = 0.80

#: Minimum per-segment sample size for subgroup tests.
#: Below this threshold results are too noisy to be actionable.
MIN_SEGMENT_N: int = 5

# ---------------------------------------------------------------------------
# Business parameters
# ---------------------------------------------------------------------------

#: Fraction of GMV the platform retains as revenue.
#: ezCater's take rate is publicly estimated at 15–20 %.
PLATFORM_TAKE_RATE: float = 0.20

#: ROI threshold: incremental_revenue / discount_cost must exceed this
#: value for the campaign to be considered financially viable.
ROI_BREAK_EVEN: float = 1.0

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

#: Global random seed used by both NumPy and the standard library ``random``
#: module.  Change this value to generate a different-but-reproducible dataset.
RANDOM_SEED: int = 42
