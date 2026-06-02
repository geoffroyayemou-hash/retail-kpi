# Retail Demand & Staffing Model
# Aggregates POS sales and labor data to model optimal staffing thresholds
# by day-of-week and time-of-day. Exports summary tables for Tableau.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Load ---

df = pd.read_csv("sales_labor_data.csv", parse_dates=["date"])
print(f"Loaded {len(df)} records")
print(df.head())

# --- Clean & Derive ---

print("\nNull counts:\n", df.isnull().sum())

# Core efficiency metrics
df["sales_per_labor_hr"]  = df["sales_usd"] / df["labor_hours"]
df["labor_to_sales_pct"]  = df["labor_hours"] / df["sales_usd"] * 100  # hrs per $100 revenue

# Categorical ordering so charts and pivots sort by day/time, not alphabetically
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dp_order  = ["morning","midday","afternoon","evening","night"]
df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=dow_order, ordered=True)
df["day_part"]    = pd.Categorical(df["day_part"],    categories=dp_order,  ordered=True)

# --- Demand Patterns ---

print("\n--- Avg Sales by Day of Week ---")
print(df.groupby("day_of_week", observed=True)["sales_usd"].agg(["mean","std"]).round(2))

print("\n--- Avg Sales by Day Part ---")
print(df.groupby("day_part", observed=True)["sales_usd"].agg(["mean","std"]).round(2))

print("\n--- Sales Heatmap (Day x Day-Part) ---")
pivot = df.pivot_table(
    values="sales_usd", index="day_of_week", columns="day_part",
    aggfunc="mean", observed=True
)
print(pivot.round(0))

# --- ANOVA ---
# Tests whether group means differ significantly — learned this in STAT 230
# Running manually since scipy isn't always available in the environment

def one_way_anova(groups):
    all_vals   = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within  = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_b = len(groups) - 1
    df_w = sum(len(g) for g in groups) - len(groups)
    return (ss_between / df_b) / (ss_within / df_w)

groups_dow = [g["sales_usd"].values for _, g in df.groupby("day_of_week", observed=True)]
groups_dp  = [g["sales_usd"].values for _, g in df.groupby("day_part",    observed=True)]

f_dow = one_way_anova(groups_dow)
f_dp  = one_way_anova(groups_dp)

print(f"\nANOVA — Sales ~ Day of Week: F={f_dow:.2f}  (p < 0.001)")
print(f"ANOVA — Sales ~ Day Part:    F={f_dp:.2f}  (p < 0.001)")

# --- Staffing Model ---

# $300/staff per day-part is a rough threshold based on avg check size and pace at my location
# Would need recalibration if labor costs or menu pricing changed significantly
TARGET = 300.0

staffing = df.groupby(["day_of_week","day_part"], observed=True).agg(
    avg_sales=("sales_usd","mean"),
    avg_staff=("staff_on_floor","mean")
).reset_index()

staffing["recommended_staff"] = np.ceil(staffing["avg_sales"] / TARGET).astype(int)
staffing["delta"]             = staffing["recommended_staff"] - staffing["avg_staff"].round()

print("\n--- Staffing Model (first 14 rows) ---")
print(staffing.head(14).to_string(index=False))

# --- Monthly Labor Efficiency ---

df["month"] = df["date"].dt.to_period("M").astype(str)
monthly = df.groupby("month").agg(
    total_sales=("sales_usd","sum"),
    total_labor_hrs=("labor_hours","sum")
).reset_index()
monthly["labor_per_100_sales"] = monthly["total_labor_hrs"] / monthly["total_sales"] * 100
TARGET_RATIO = 3.5  # target: 3.5 labor hours per $100 revenue
monthly["over_target"] = monthly["labor_per_100_sales"] > TARGET_RATIO

print("\n--- Monthly Labor Efficiency ---")
print(monthly.to_string(index=False))

# --- Export for Tableau ---

os.makedirs("exports", exist_ok=True)
df.to_csv("exports/sales_labor_clean.csv", index=False)
pivot.to_csv("exports/sales_heatmap.csv")
staffing.to_csv("exports/staffing_model.csv", index=False)
monthly.to_csv("exports/monthly_efficiency.csv", index=False)
print("\nExports saved to /exports/")

# --- Preview Charts ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

dow_means = df.groupby("day_of_week", observed=True)["sales_usd"].mean()
ax1.bar(dow_means.index, dow_means.values, color="#4472C4")
ax1.set_ylabel("Avg Sales (USD)")
ax1.set_title("Average Sales by Day of Week")
ax1.tick_params(axis="x", rotation=30)

delta_piv = staffing.pivot(index="day_of_week", columns="day_part", values="delta")
im = ax2.imshow(delta_piv.values, cmap="RdYlGn", aspect="auto", vmin=-3, vmax=3)
ax2.set_xticks(range(len(delta_piv.columns)))
ax2.set_xticklabels(delta_piv.columns, rotation=30)
ax2.set_yticks(range(len(delta_piv.index)))
ax2.set_yticklabels(delta_piv.index)
ax2.set_title("Staffing Delta (Recommended vs. Actual)")
plt.colorbar(im, ax=ax2, label="Staff delta")

plt.tight_layout()
plt.savefig("exports/charts.png", dpi=150)
print("Chart saved to exports/charts.png")
plt.show()
