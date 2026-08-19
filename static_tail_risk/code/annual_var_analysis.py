import pandas as pd
import numpy as np
from scipy.stats import norm, t, skew, kurtosis

# ============================================================
# 1. Load data
# ============================================================

filename = "ReturnByYear.dat"   # change if needed

# File should have columns: Year, Return
df = pd.read_csv(filename)
#df = df.dropna(subset=["Year", "Return"])
df["Year"] = df["Year"].astype(int)

# ============================================================
# 2. Settings
# ============================================================

confidence_levels = [0.98]

# ============================================================
# 3. Functions
# ============================================================

def empirical_var_es(data, cl):
    alpha = 1 - cl
    var = np.quantile(data, alpha)
    es = data[data <= var].mean()
    return var, es

def normal_var_es(data, cl):
    alpha = 1 - cl
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    z = norm.ppf(alpha)

    var = mu + z * sigma
    es = mu - sigma * norm.pdf(z) / alpha
    return var, es

def t_var_es(data, cl):
    alpha = 1 - cl

    # Fit t-distribution
    df_t, loc_t, scale_t = t.fit(data)

    q = t.ppf(alpha, df_t)

    var = loc_t + scale_t * q

    # Left-tail Expected Shortfall for t-distribution
    pdf_q = t.pdf(q, df_t)
    es_standard = -((df_t + q**2) / (df_t - 1)) * pdf_q / alpha
    es = loc_t + scale_t * es_standard

    return var, es, df_t, loc_t, scale_t

# ============================================================
# 4. Year-by-year calculation
# ============================================================

rows = []

for year, group in df.groupby("Year"):
    data = group["Return"].dropna().values
    n = len(data)

    if n < 30:
        continue

    mean_return = np.mean(data)
    std_return = np.std(data, ddof=1)
    skewness = skew(data)
    excess_kurtosis = kurtosis(data, fisher=True)

    for cl in confidence_levels:

        emp_var, emp_es = empirical_var_es(data, cl)
        norm_var, norm_es = normal_var_es(data, cl)

        try:
            tvar, tes, df_t, loc_t, scale_t = t_var_es(data, cl)
        except Exception:
            tvar, tes, df_t, loc_t, scale_t = np.nan, np.nan, np.nan, np.nan, np.nan

        rows.append({
            "Year": year,
            "N": n,
            "Confidence": f"{cl:.0%}",
            "Mean": mean_return,
            "Std": std_return,
            "Skewness": skewness,
            "Excess_Kurtosis": excess_kurtosis,
            "Empirical_VaR": emp_var,
            "Empirical_ES": emp_es,
            "Normal_VaR": norm_var,
            "Normal_ES": norm_es,
            "t_VaR": tvar,
            "t_ES": tes,
            "t_df": df_t,
            "t_loc": loc_t,
            "t_scale": scale_t
        })

results = pd.DataFrame(rows)

# ============================================================
# 5. Display formatted table
# ============================================================

display_results = results.copy()

percent_cols = [
    "Mean", "Std",
    "Empirical_VaR", "Empirical_ES",
    "Normal_VaR", "Normal_ES",
    "t_VaR", "t_ES"
]

for col in percent_cols:
    display_results[col] = display_results[col].apply(lambda x: f"{x:.3%}" if pd.notnull(x) else "")

print(display_results.to_string(index=False))

# ============================================================
# 6. Save output
# ============================================================

results.to_csv("yearly_var_es_results.csv", index=False)

print("\nSaved:")
print("yearly_var_es_results.csv")
