import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# =========================
# 1. Load data
# =========================
# IMPORTANT:
# Rolling-window analysis needs chronological order.

filename = "Return.dat"
df = pd.read_csv(filename)

# Clean column names just in case
df.columns = df.columns.str.strip()

# Convert "-11.59%" into -0.1159
returns = (
    df["Return"]
    .astype(str)
    .str.strip()
    .str.replace("%", "", regex=False)
    .astype(float)
    / 100
)

returns = returns.dropna().reset_index(drop=True)

n = len(returns)

print("Dataset")
print("-------")
print(f"Observations: {n}")
print(f"Mean: {np.mean(returns):.6f}")
print(f"Std:  {np.std(returns, ddof=1):.6f}")

# =========================
# 2. Settings
# =========================
window = 750
confidence = 0.99
alpha = 1 - confidence

# =========================
# 3. Rolling calculations
# =========================
rolling_hist_var = []
rolling_hist_es = []

rolling_normal_var = []
rolling_normal_es = []

rolling_t_var = []
rolling_t_es = []
rolling_t_df = []
rolling_t_loc = []
rolling_t_scale = []

actual_next_return = []

for i in range(window, len(returns)):
    train = returns.iloc[i-window:i]
    next_ret = returns.iloc[i]

    # -------------------------
    # Historical VaR / ES
    # -------------------------
    hist_var = np.quantile(train, alpha)
    hist_es = train[train <= hist_var].mean()

    # -------------------------
    # Normal VaR / ES
    # -------------------------
    mu = train.mean()
    sigma = train.std(ddof=1)
    z = norm.ppf(alpha)

    normal_var = mu + sigma * z
    normal_es = mu - sigma * norm.pdf(z) / alpha

    # -------------------------
    # t-distribution VaR / ES
    # -------------------------
    try:
        df_t, loc_t, scale_t = t.fit(train)

        q_t = t.ppf(alpha, df_t)

        t_var = loc_t + scale_t * q_t

        # Left-tail Expected Shortfall for t-distribution
        pdf_q = t.pdf(q_t, df_t)
        t_es_standard = -((df_t + q_t**2) / (df_t - 1)) * pdf_q / alpha
        t_es = loc_t + scale_t * t_es_standard

    except Exception:
        df_t, loc_t, scale_t = np.nan, np.nan, np.nan
        t_var, t_es = np.nan, np.nan

    # Save results
    rolling_hist_var.append(hist_var)
    rolling_hist_es.append(hist_es)

    rolling_normal_var.append(normal_var)
    rolling_normal_es.append(normal_es)

    rolling_t_var.append(t_var)
    rolling_t_es.append(t_es)
    rolling_t_df.append(df_t)
    rolling_t_loc.append(loc_t)
    rolling_t_scale.append(scale_t)

    actual_next_return.append(next_ret)

# =========================
# 4. Results dataframe
# =========================
results = pd.DataFrame({
    "Actual_Return": actual_next_return,

    "Historical_VaR": rolling_hist_var,
    "Historical_ES": rolling_hist_es,

    "Normal_VaR": rolling_normal_var,
    "Normal_ES": rolling_normal_es,

    "t_VaR": rolling_t_var,
    "t_ES": rolling_t_es,
    "t_df": rolling_t_df,
    "t_loc": rolling_t_loc,
    "t_scale": rolling_t_scale
})

# Exceedance indicators
results["Historical_Exceedance"] = results["Actual_Return"] < results["Historical_VaR"]
results["Normal_Exceedance"] = results["Actual_Return"] < results["Normal_VaR"]
results["t_Exceedance"] = results["Actual_Return"] < results["t_VaR"]

# =========================
# 5. Backtesting summary
# =========================
expected_exceedances = len(results) * alpha

hist_actual = results["Historical_Exceedance"].sum()
normal_actual = results["Normal_Exceedance"].sum()
t_actual = results["t_Exceedance"].sum()

summary = pd.DataFrame({
    "Model": [
        "Rolling Historical",
        "Rolling Normal",
        "Rolling t-distribution"
    ],
    "Expected Exceedances": [
        expected_exceedances,
        expected_exceedances,
        expected_exceedances
    ],
    "Actual Exceedances": [
        hist_actual,
        normal_actual,
        t_actual
    ],
    "Actual / Expected": [
        hist_actual / expected_exceedances,
        normal_actual / expected_exceedances,
        t_actual / expected_exceedances
    ]
})

print("\nRolling VaR Backtesting Summary")
print("--------------------------------")
print(summary.to_string(index=False))

# =========================
# 6. Save results
# =========================
results.to_csv("rolling_var_results_with_t.csv", index=False)
summary.to_csv("rolling_var_backtesting_summary_with_t.csv", index=False)

print("\nSaved:")
print("rolling_var_results_with_t.csv")
print("rolling_var_backtesting_summary_with_t.csv")

# =========================
# 7. Plot rolling VaR
# =========================
plt.figure(figsize=(11, 6))

plt.plot(
    results.index,
    results["Actual_Return"],
    linewidth=0.7,
    alpha=0.5,
    label="Actual Return"
)

plt.plot(
    results.index,
    results["Historical_VaR"],
    linewidth=1.5,
    label="Rolling Historical VaR"
)

plt.plot(
    results.index,
    results["Normal_VaR"],
    linewidth=1.5,
    label="Rolling Normal VaR"
)

plt.plot(
    results.index,
    results["t_VaR"],
    linewidth=1.5,
    label="Rolling t-distribution VaR"
)

plt.axhline(0, linewidth=0.8)

plt.xlabel("Time Index")
plt.ylabel("Daily Return")
plt.title(f"Rolling {confidence:.0%} VaR Comparison, Window = {window} Days")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("rolling_var_plot_with_t.png", dpi=300)
plt.show()

print("\nSaved:")
print("rolling_var_plot_with_t.png")
