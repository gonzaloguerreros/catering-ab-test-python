"""
visualizations.py
=================
All matplotlib/seaborn charts for the A/B test report.

Each function saves a PNG to outputs/ and returns the figure object so it can
also be rendered inline in a notebook.  Charts follow a clean, minimal style
that reads well in both light-mode dashboards and PDF exports.
"""

from __future__ import annotations
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Global style — consistent palette across all charts
# ---------------------------------------------------------------------------
CONTROL_COLOR   = "#5B8DB8"   # muted blue
TREATMENT_COLOR = "#E07B54"   # warm orange
NEUTRAL_COLOR   = "#6C757D"
BACKGROUND      = "#F8F9FA"

sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "figure.facecolor":  BACKGROUND,
    "axes.facecolor":    BACKGROUND,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
})

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> None:
    """Save figure to outputs/ at 150 DPI (crisp without being huge)."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → Saved: {path}")


# ---------------------------------------------------------------------------
# 1. Conversion Rate Comparison with CI bars
# ---------------------------------------------------------------------------

def plot_conversion_rates(conv_result: dict) -> plt.Figure:
    """
    Bar chart of conversion rates with 95% Wilson CI error bars.
    Annotates the absolute and relative lift.
    """
    groups  = ["Control", "Treatment"]
    rates   = [conv_result["rate_control"], conv_result["rate_treatment"]]
    ci_ctrl = conv_result["ci_95_control"]
    ci_treat= conv_result["ci_95_treatment"]
    errors  = [
        [rates[0] - ci_ctrl[0],  ci_ctrl[1]  - rates[0]],
        [rates[1] - ci_treat[0], ci_treat[1] - rates[1]],
    ]
    yerr = np.array(errors).T   # shape (2, 2) for asymmetric bars

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        groups, [r * 100 for r in rates],
        color=[CONTROL_COLOR, TREATMENT_COLOR],
        width=0.45, zorder=3,
        yerr=[[e * 100 for e in errors[0]], [e * 100 for e in errors[1]]],
        capsize=8, error_kw={"color": NEUTRAL_COLOR, "linewidth": 1.5}
    )

    # Annotate bar tops with rate values
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{rate*100:.1f}%",
            ha="center", va="bottom", fontweight="bold", fontsize=12
        )

    # Lift annotation between the bars
    lift_pct = conv_result["relative_lift_pct"]
    abs_lift = conv_result["absolute_lift"] * 100
    p_val    = conv_result["p_value"]
    sig_str  = "✓ Statistically significant" if conv_result["significant"] else "✗ Not significant"
    ax.annotate(
        f"+{abs_lift:.1f}pp ({lift_pct:+.1f}%)\np = {p_val:.4f}  {sig_str}",
        xy=(0.5, max(rates) * 100 + 5),
        xycoords=("data", "data"),
        ha="center", fontsize=10,
        color="green" if conv_result["significant"] else NEUTRAL_COLOR
    )

    ax.set_ylabel("Conversion Rate (%)")
    ax.set_title("Conversion Rate: Control vs. Treatment\n(95% Wilson CI error bars)")
    ax.set_ylim(0, max(rates) * 100 * 1.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    fig.tight_layout()
    _save(fig, "01_conversion_rates.png")
    return fig


# ---------------------------------------------------------------------------
# 2. First-Order GMV Distribution (KDE + box plots)
# ---------------------------------------------------------------------------

def plot_gmv_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Side-by-side KDE curves and box plots for first-order GMV.
    Converters only — shows how the discount shifts the order value
    distribution, not just the average.
    """
    converters = df[df["converted"] == 1].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # KDE plot
    for group, color, label in [
        ("control",   CONTROL_COLOR,   "Control"),
        ("treatment", TREATMENT_COLOR, "Treatment"),
    ]:
        vals = converters.loc[converters["group"] == group, "first_order_value"]
        sns.kdeplot(vals, ax=axes[0], color=color, label=label,
                    fill=True, alpha=0.35, linewidth=2)
        axes[0].axvline(vals.mean(), color=color, linestyle="--",
                        linewidth=1.5, label=f"{label} mean: ${vals.mean():,.0f}")

    axes[0].set_xlabel("First-Order GMV ($)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("First-Order GMV Distribution")
    axes[0].legend(fontsize=9)

    # Box plot
    plot_data = converters[["group", "first_order_value"]].copy()
    plot_data["group_label"] = plot_data["group"].map(
        {"control": "Control", "treatment": "Treatment"}
    )
    sns.boxplot(
        data=plot_data, x="group_label", y="first_order_value",
        palette={"Control": CONTROL_COLOR, "Treatment": TREATMENT_COLOR},
        ax=axes[1], width=0.45, flierprops={"marker": "o", "markersize": 4}
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("First-Order GMV ($)")
    axes[1].set_title("First-Order GMV Box Plot")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    fig.suptitle("First-Order GMV: Control vs. Treatment (Converters Only)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "02_gmv_distribution.png")
    return fig


# ---------------------------------------------------------------------------
# 3. Cumulative Conversion Over Time
# ---------------------------------------------------------------------------

def plot_cumulative_conversion(df: pd.DataFrame) -> plt.Figure:
    """
    Day-by-day cumulative conversion curves for each group.
    Shows whether the treatment converts faster (earlier) or just more.
    """
    df = df.copy()
    df["signup_date"]      = pd.to_datetime(df["signup_date"])
    df["first_order_date"] = pd.to_datetime(df["first_order_date"])
    df["days_to_convert"]  = (
        df["first_order_date"] - df["signup_date"]
    ).dt.days.fillna(np.inf)

    fig, ax = plt.subplots(figsize=(10, 5))

    for group, color, label in [
        ("control",   CONTROL_COLOR,   "Control"),
        ("treatment", TREATMENT_COLOR, "Treatment"),
    ]:
        g_df  = df[df["group"] == group]
        n     = len(g_df)
        days  = np.arange(0, 32)   # 0–30 days post-signup
        cumulative_rate = [
            (g_df["days_to_convert"] <= d).sum() / n * 100
            for d in days
        ]
        ax.plot(days, cumulative_rate, color=color, linewidth=2.5, label=label)
        ax.fill_between(days, cumulative_rate, alpha=0.1, color=color)

    ax.set_xlabel("Days Since Signup")
    ax.set_ylabel("Cumulative Conversion Rate (%)")
    ax.set_title("Cumulative Conversion Rate by Day Post-Signup")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend()
    ax.set_xlim(0, 30)
    fig.tight_layout()
    _save(fig, "03_cumulative_conversion.png")
    return fig


# ---------------------------------------------------------------------------
# 4. Subgroup Lift Chart (forest plot style)
# ---------------------------------------------------------------------------

def plot_subgroup_lift(subgroup_df: pd.DataFrame) -> plt.Figure:
    """
    Horizontal bar chart showing relative lift per industry segment.
    Bars are coloured green (significant) vs. grey (not significant) after
    Bonferroni correction.  This is the standard forest-plot-style view
    used in clinical trials and growth experiments.
    """
    df = subgroup_df.sort_values("relative_lift_pct")

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = [
        "#2ECC71" if sig else NEUTRAL_COLOR
        for sig in df["significant_bonferroni"]
    ]
    bars = ax.barh(df["industry"], df["relative_lift_pct"],
                   color=colors, height=0.5, zorder=3)

    # Annotate with lift values
    for bar, val in zip(bars, df["relative_lift_pct"]):
        x_pos = bar.get_width() + 0.3 if val >= 0 else bar.get_width() - 0.3
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%", va="center", fontsize=9)

    # Reference line at zero
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")

    # Legend
    sig_patch  = mpatches.Patch(color="#2ECC71", label="Significant (Bonferroni)")
    nsig_patch = mpatches.Patch(color=NEUTRAL_COLOR, label="Not Significant")
    ax.legend(handles=[sig_patch, nsig_patch], fontsize=9, loc="lower right")

    ax.set_xlabel("Relative Conversion Lift (%)")
    ax.set_title("Subgroup Analysis — Conversion Lift by Industry\n(Bonferroni-corrected α)")
    fig.tight_layout()
    _save(fig, "04_subgroup_lift.png")
    return fig


# ---------------------------------------------------------------------------
# 5. 60-Day GMV Comparison
# ---------------------------------------------------------------------------

def plot_60d_gmv(df: pd.DataFrame, gmv_result: dict) -> plt.Figure:
    """
    Violin plot of 60-day GMV (including zero-value non-converters).
    Illustrates the full distribution shape, not just the mean.
    """
    plot_df = df[["group", "gmv_60d"]].copy()
    plot_df["group_label"] = plot_df["group"].map(
        {"control": "Control", "treatment": "Treatment"}
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(
        data=plot_df, x="group_label", y="gmv_60d",
        palette={"Control": CONTROL_COLOR, "Treatment": TREATMENT_COLOR},
        inner="quartile", ax=ax, cut=0
    )

    # Overlay mean markers
    for group, color in [("control", CONTROL_COLOR), ("treatment", TREATMENT_COLOR)]:
        mean_val = df.loc[df["group"] == group, "gmv_60d"].mean()
        label = "Control" if group == "control" else "Treatment"
        ax.scatter([label], [mean_val], color="white", s=80,
                   zorder=5, edgecolors=color, linewidths=2)

    mean_lift = gmv_result["absolute_mean_lift"]
    p_val     = gmv_result["p_value"]
    sig_str   = "✓ p < 0.05" if gmv_result["significant"] else f"p = {p_val:.3f}"
    ax.set_title(
        f"60-Day GMV Distribution (All Accounts)\n"
        f"Mean lift: +${mean_lift:,.2f} per account  |  {sig_str}"
    )
    ax.set_xlabel("")
    ax.set_ylabel("60-Day GMV ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    _save(fig, "05_60d_gmv.png")
    return fig
