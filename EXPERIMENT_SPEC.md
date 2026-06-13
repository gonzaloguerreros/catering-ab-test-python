# Experiment Spec — SPRING20 Campaign
**Status:** DRAFT — Pre-registration document (filled out before data collection)  
**Author:** Gonzalo Guerreros · Product Analyst  
**Date:** March 1, 2024

---

> **Why this document exists:**  
> Pre-registering the experiment design before launch is the primary defence
> against p-hacking and HARKing (Hypothesising After Results are Known).
> Metrics, segment cuts, and the ship criteria are locked here before any
> results are visible. Changing them afterwards requires a new experiment.

---

## Hypothesis

**If** we offer newly acquired mid-market corporate accounts a 20% discount on their first order, **then** their 30-day conversion rate will increase because the discount reduces the psychological and financial barrier to placing a first order on a new platform.

---

## Experiment Design

| Parameter | Value |
|-----------|-------|
| Unit of randomisation | Corporate account (not order-level — prevents within-account spillover) |
| Control | Standard onboarding — no discount |
| Treatment | 20% off first order |
| Target population | Newly acquired mid-market accounts |
| Allocation | 50/50 (maximises statistical power for a given total N) |
| Test window | March 1 – May 31, 2024 |

---

## Metrics

### Primary Metric (ship/no-ship decision)
**30-day conversion rate** — % of enrolled accounts that place at least one delivered order within 30 days of account creation.

- Why this metric? It directly measures the mechanism: does the discount get accounts over the "first order" hurdle?
- MDE: +3 percentage points absolute (from a ~25% baseline → ~28%)
- Required sample size: ~2,400 accounts per group at 80% power, α = 0.05

### Secondary Metrics (context only — do NOT drive the ship decision)
1. **First-order GMV** — average order value for the first order (converters only). Tests whether discount nudges accounts to order more food.
2. **60-day GMV** — total revenue per account in the 60-day window, including non-converters (zeros). The most complete revenue picture.
3. **Orders in 60 days** — order frequency. Does the discount create repeat ordering or just a one-time purchase?

### Guardrail Metrics (auto-stop if any degrade significantly)
1. **Cancellation rate** — promo should not cause accounts to over-order and cancel
2. **Caterer on-time delivery rate** — higher order volume should not stress caterer capacity
3. **Support ticket rate** — discount confusion or promo abuse should not spike CS contacts

---

## Pre-Planned Segment Cuts

These cuts are registered upfront. Any segment analysis not listed here is exploratory and must be Bonferroni-corrected.

| Segment | Why |
|---------|-----|
| Account tier (SMB / mid-market / enterprise) | Promo is targeted at mid-market — verify it works there and doesn't have spillover effects |
| Industry (Technology, Finance, Healthcare, etc.) | Heterogeneous treatment effects may inform future targeting |
| **Power users vs. regular users** | Accounts with ≥ 4 orders in prior 90 days may respond differently to a discount |
| New accounts (< 60 days old) vs. established | First-order discount may matter more to brand-new accounts |
| Geography (Boston metro vs. other) | Market saturation differs |

---

## Sample Size Requirements

Computed prospectively using `src/experiment_design.py → required_sample_size_proportions()`.

| MDE (absolute) | MDE (relative) | N per group | N total |
|---------------|----------------|-------------|---------|
| +1.0 pp | +4% | 14,752 | 29,504 |
| +2.0 pp | +8% | 3,814 | 7,628 |
| **+3.0 pp** | **+12%** | **1,727** | **3,454** |
| +5.0 pp | +20% | 641 | 1,282 |
| +7.0 pp | +28% | 336 | 672 |

> **Decision:** Targeting the +3pp MDE as the minimum effect worth acting on.
> At 500 accounts/group (this experiment's actual sample), we can only reliably detect effects ≥ ~12.5 pp — the experiment is **underpowered for the target MDE**.
> Recommended: run for a full quarter before reading results, or expand enrollment to all newly acquired accounts.

---

## Runtime Estimate

At ~30 new mid-market accounts per day:
- Days to reach 3,454 accounts: **~115 days** (~16.5 weeks)
- Recommended runtime: **16 weeks** (full quarter, captures monthly billing cycles)

---

## Analysis Plan

1. Run SRM check before looking at any outcomes
2. Check covariate balance (industry, tier) between groups
3. Test primary metric (conversion rate) with two-proportion z-test
4. Test secondary metrics (GMV) with Welch's t-test + Mann-Whitney cross-check
5. Apply Bonferroni correction to all registered segment cuts
6. Calculate business impact (discount cost vs. incremental platform revenue)

**Ship criteria:** Primary metric significant (p < 0.05, two-sided) AND campaign ROI > 1.0x AND no guardrail metrics degraded.

---

*This document should be reviewed and signed off by the PM, Data Science lead, and Finance before the experiment launches.*
