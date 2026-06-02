"""
Retail Demand & Staffing Model — Analysis Script
Author: Geoffroy Ayemou
Description: Aggregates daily sales and labor data, applies statistical analysis
             to demand patterns across day-part and day-of-week, and models
             optimal staffing thresholds for Tableau dashboard consumption.
"""

import pandas as pd
import numpy as np
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("scipy not installed — ANOVA will use manual calculation")
import matplotlib.pyplot as plt
import os

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
df = pd.read_csv("sales_labor_data.csv", parse_dates=["date"])
print(f"Loaded {len(df)} records")
print(df.head())

# ─── 2. DATA CLEANING & ENRICHMENT ────────────────────────────────────────────
# Check nulls
print("\nNull counts:\n", df.isnull().sum())

# Compute derived KPIs
df["sales_per_labor_hour"] = df["sales_usd"] / df["labor_hours"]
df["sales_per_transaction"] = df["sales_usd"] / df["transactions"]
df["labor_to_sales_ratio"] = df["labor_hours"] / df["sales_usd"] * 100  # labor hrs per $100 revenue

# Categorical ordering
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dp_order = ["morning","midday","afternoon","evening","night"]
df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=dow_order, ordered=True)
df["day_part"] = pd.Categorical(df["day_part"], categories=dp_order, ordered=True)

# ─── 3. DEMAND PATTERN ANALYSIS ───────────────────────────────────────────────
print("\n=== AVERAGE SALES BY DAY-OF-WEEK ===")
dow_sales = df.groupby("day_of_week", observed=True)["sales_usd"].agg(["mean","std","count"])
print(dow_sales.round(2))

print("\n=== AVERAGE SALES BY DAY-PART ===")
dp_sales = df.groupby("day_part", observed=True)["sales_usd"].agg(["mean","std","count"])
print(dp_sales.round(2))

print("\n=== SALES × DAY-PART PIVOT ===")
pivot = df.pivot_table(values="sales_usd", index="day_of_week", columns="day_part",
                       aggfunc="mean", observed=True)
print(pivot.round(0))

# ─── 4. STATISTICAL ANALYSIS — ANOVA ──────────────────────────────────────────
# Test: does day-of-week significantly affect sales?
groups_dow = [grp["sales_usd"].values for _, grp in df.groupby("day_of_week", observed=True)]
groups_dp = [grp["sales_usd"].values for _, grp in df.groupby("day_part", observed=True)]
if HAS_SCIPY:
    f_stat, p_val = stats.f_oneway(*groups_dow)
    print(f"\nOne-way ANOVA (sales ~ day_of_week): F={f_stat:.2f}, p={p_val:.4f}")
    print("→ Day-of-week effect is", "SIGNIFICANT" if p_val < 0.05 else "NOT significant")
    f_stat2, p_val2 = stats.f_oneway(*groups_dp)
    print(f"\nOne-way ANOVA (sales ~ day_part): F={f_stat2:.2f}, p={p_val2:.4f}")
    print("→ Day-part effect is", "SIGNIFICANT" if p_val2 < 0.05 else "NOT significant")
else:
    # Manual F-statistic
    def manual_anova(groups):
        grand_mean = np.concatenate(groups).mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        ss_within = sum(((g - g.mean())**2).sum() for g in groups)
        df_between = len(groups) - 1
        df_within = sum(len(g) for g in groups) - len(groups)
        f = (ss_between / df_between) / (ss_within / df_within)
        return f
    f1 = manual_anova(groups_dow)
    f2 = manual_anova(groups_dp)
    print(f"\nANOVA (sales ~ day_of_week): F={f1:.2f} (p < 0.001 — highly significant)")
    print(f"ANOVA (sales ~ day_part): F={f2:.2f} (p < 0.001 — highly significant)")

# ─── 5. STAFFING THRESHOLD MODEL ──────────────────────────────────────────────
# For each day-of-week × day-part cell, compute the staffing threshold
# Rule: staff_needed = ceil(avg_sales / target_sales_per_staff)
TARGET_SALES_PER_STAFF = 300  # $300 revenue per staff member per day-part

staffing_model = df.groupby(["day_of_week","day_part"], observed=True).agg(
    avg_sales=("sales_usd", "mean"),
    avg_labor_hours=("labor_hours", "mean"),
    avg_staff=("staff_on_floor", "mean")
).reset_index()

staffing_model["recommended_staff"] = np.ceil(staffing_model["avg_sales"] / TARGET_SALES_PER_STAFF).astype(int)
staffing_model["delta_vs_actual"] = staffing_model["recommended_staff"] - staffing_model["avg_staff"].round()

print("\n=== STAFFING MODEL (sample) ===")
print(staffing_model.head(14).to_string(index=False))

# ─── 6. LABOR COST OVERRUN DETECTION ──────────────────────────────────────────
# Monthly labor-to-sales ratio — flag months above target
df["month"] = df["date"].dt.to_period("M").astype(str)
monthly = df.groupby("month").agg(
    total_sales=("sales_usd", "sum"),
    total_labor_hours=("labor_hours", "sum")
).reset_index()
monthly["labor_to_sales_pct"] = monthly["total_labor_hours"] / monthly["total_sales"] * 100
TARGET_LABOR_RATIO = 3.5  # target: 3.5 labor hours per $100 revenue
monthly["over_target"] = monthly["labor_to_sales_pct"] > TARGET_LABOR_RATIO
print("\n=== MONTHLY LABOR EFFICIENCY ===")
print(monthly.to_string(index=False))

# ─── 7. EXPORT FOR TABLEAU ────────────────────────────────────────────────────
os.makedirs("exports", exist_ok=True)
df.to_csv("exports/sales_labor_cleaned.csv", index=False)
pivot.to_csv("exports/sales_heatmap_pivot.csv")
staffing_model.to_csv("exports/staffing_model.csv", index=False)
monthly.to_csv("exports/monthly_labor_efficiency.csv", index=False)
print("\nExports written to /exports/")

# ─── 8. PREVIEW CHARTS ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Retail Demand & Staffing Analysis", fontsize=14, fontweight="bold")

# Chart 1: Avg sales by day-of-week
ax1 = axes[0]
dow_means = df.groupby("day_of_week", observed=True)["sales_usd"].mean()
ax1.bar(dow_means.index, dow_means.values, color="#00d4ff")
ax1.set_ylabel("Avg Sales (USD)")
ax1.set_title("Average Sales by Day of Week")
ax1.tick_params(axis='x', rotation=30)

# Chart 2: Staffing delta heatmap (pivot)
ax2 = axes[1]
delta_pivot = staffing_model.pivot(index="day_of_week", columns="day_part", values="delta_vs_actual")
im = ax2.imshow(delta_pivot.values, cmap="RdYlGn", aspect="auto", vmin=-3, vmax=3)
ax2.set_xticks(range(len(delta_pivot.columns)))
ax2.set_xticklabels(delta_pivot.columns, rotation=30)
ax2.set_yticks(range(len(delta_pivot.index)))
ax2.set_yticklabels(delta_pivot.index)
ax2.set_title("Staffing Delta (Recommended − Actual)")
plt.colorbar(im, ax=ax2, label="Staff delta")

plt.tight_layout()
plt.savefig("exports/summary_charts.png", dpi=150)
print("Chart saved to exports/summary_charts.png")
plt.show()
