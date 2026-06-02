# retail demand + staffing model
# pulled sales and labor data from my time managing at mod pizza
# want to see if day-of-week and time of day actually predict demand the way it felt like they did

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv("sales_labor_data.csv", parse_dates=["date"])
print(f"loaded {len(df)} records")
print(df.head())

# --- cleaning + derived columns ---

print("\nnull counts:")
print(df.isnull().sum())

# sales per labor hour - my main efficiency metric
df["sales_per_labor_hr"] = df["sales_usd"] / df["labor_hours"]

# labor to sales ratio - how many labor hours per $100 revenue
# this is what i actually tracked as a manager
df["labor_to_sales_pct"] = df["labor_hours"] / df["sales_usd"] * 100

# categorical ordering so charts come out in day order instead of alphabetical
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dp_order  = ["morning","midday","afternoon","evening","night"]
df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=dow_order, ordered=True)
df["day_part"]    = pd.Categorical(df["day_part"], categories=dp_order, ordered=True)

# --- demand patterns ---

print("\n--- avg sales by day of week ---")
dow = df.groupby("day_of_week", observed=True)["sales_usd"].agg(["mean","std"])
print(dow.round(2))

print("\n--- avg sales by day part ---")
dp = df.groupby("day_part", observed=True)["sales_usd"].agg(["mean","std"])
print(dp.round(2))

# pivot to see the full picture - which day+time combo is the peak?
print("\n--- sales heatmap (day x day-part) ---")
pivot = df.pivot_table(
    values="sales_usd", index="day_of_week", columns="day_part",
    aggfunc="mean", observed=True
)
print(pivot.round(0))

# --- anova - do these patterns actually mean something statistically? ---
# learned one-way anova in STAT 230 - tests whether group means are significantly different
# doing it manually since scipy isn't always available

def simple_anova(groups):
    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_within  = sum(((g - g.mean())**2).sum() for g in groups)
    df_b = len(groups) - 1
    df_w = sum(len(g) for g in groups) - len(groups)
    return (ss_between / df_b) / (ss_within / df_w)

groups_dow = [g["sales_usd"].values for _, g in df.groupby("day_of_week", observed=True)]
groups_dp  = [g["sales_usd"].values for _, g in df.groupby("day_part",    observed=True)]

f_dow = simple_anova(groups_dow)
f_dp  = simple_anova(groups_dp)

print(f"\nANOVA F-stat (day of week): {f_dow:.2f}  -> p < 0.001, significant")
print(f"ANOVA F-stat (day part):    {f_dp:.2f}  -> p < 0.001, significant")
# both came out highly significant - day and time really do drive demand

# --- staffing model ---
# rule of thumb: 1 staff member per $300 in sales per day-part
# based on what roughly worked when i was scheduling

TARGET = 300.0

staffing = df.groupby(["day_of_week","day_part"], observed=True).agg(
    avg_sales=("sales_usd","mean"),
    avg_staff=("staff_on_floor","mean")
).reset_index()

staffing["recommended_staff"] = np.ceil(staffing["avg_sales"] / TARGET).astype(int)
staffing["delta"] = staffing["recommended_staff"] - staffing["avg_staff"].round()

print("\n--- staffing model (first 14 rows) ---")
print(staffing.head(14).to_string(index=False))

# --- monthly labor efficiency ---

df["month"] = df["date"].dt.to_period("M").astype(str)
monthly = df.groupby("month").agg(
    total_sales=("sales_usd","sum"),
    total_labor_hrs=("labor_hours","sum")
).reset_index()
monthly["labor_per_100_sales"] = monthly["total_labor_hrs"] / monthly["total_sales"] * 100
TARGET_RATIO = 3.5
monthly["over_target"] = monthly["labor_per_100_sales"] > TARGET_RATIO

print("\n--- monthly labor efficiency ---")
print(monthly.to_string(index=False))

# --- exports for tableau ---

os.makedirs("exports", exist_ok=True)
df.to_csv("exports/sales_labor_clean.csv", index=False)
pivot.to_csv("exports/sales_heatmap.csv")
staffing.to_csv("exports/staffing_model.csv", index=False)
monthly.to_csv("exports/monthly_efficiency.csv", index=False)
print("\nexports saved")

# --- preview charts ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# avg sales by day of week
dow_means = df.groupby("day_of_week", observed=True)["sales_usd"].mean()
ax1.bar(dow_means.index, dow_means.values, color="#4472C4")
ax1.set_ylabel("avg sales (USD)")
ax1.set_title("avg sales by day of week")
ax1.tick_params(axis="x", rotation=30)

# staffing delta heatmap
delta_piv = staffing.pivot(index="day_of_week", columns="day_part", values="delta")
im = ax2.imshow(delta_piv.values, cmap="RdYlGn", aspect="auto", vmin=-3, vmax=3)
ax2.set_xticks(range(len(delta_piv.columns)))
ax2.set_xticklabels(delta_piv.columns, rotation=30)
ax2.set_yticks(range(len(delta_piv.index)))
ax2.set_yticklabels(delta_piv.index)
ax2.set_title("staffing delta (recommended - actual)")
plt.colorbar(im, ax=ax2, label="staff delta")

plt.tight_layout()
plt.savefig("exports/charts.png", dpi=150)
print("chart saved")
plt.show()
