import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import (
    norm,
    t,
    gaussian_kde,
    anderson
)

from sklearn.mixture import GaussianMixture

# ============================================================
# 1. LOAD DATA
# ============================================================

filename = "Return.dat"

df = pd.read_csv(filename)

# Clean column names
df.columns = df.columns.str.strip()

# Convert "-11.59%" -> -0.1159
returns = (
    df["Return"]
    .astype(str)
    .str.strip()
    .str.replace("%", "", regex=False)
    .astype(float)
    / 100
)

returns = returns.dropna().values.reshape(-1, 1)

x = returns.flatten()

n = len(x)

print("================================================")
print("DATASET")
print("================================================")
print(f"Observations: {n}")
print(f"Mean: {np.mean(x):.6f}")
print(f"Std : {np.std(x, ddof=1):.6f}")

# ============================================================
# 2. FIT DISTRIBUTIONS
# ============================================================

# ------------------------------------------------
# Normal Distribution
# ------------------------------------------------
mu_norm = np.mean(x)
sigma_norm = np.std(x, ddof=1)

# ------------------------------------------------
# t-Distribution
# ------------------------------------------------
df_t, loc_t, scale_t = t.fit(x)

# ------------------------------------------------
# GMM-2
# ------------------------------------------------
gmm = GaussianMixture(
    n_components=2,
    covariance_type='full',
    random_state=42
)

gmm.fit(returns)

weights = gmm.weights_
means = gmm.means_.flatten()
stds = np.sqrt(gmm.covariances_.flatten())

# ============================================================
# 3. OUTPUT PARAMETERS
# ============================================================

print("\n================================================")
print("t-DISTRIBUTION PARAMETERS")
print("================================================")
print(f"Degrees of Freedom = {df_t:.4f}")
print(f"Location (Mean)    = {loc_t:.6f}")
print(f"Scale              = {scale_t:.6f}")

print("\n================================================")
print("GMM-2 PARAMETERS")
print("================================================")

for i in range(2):
    print(f"\nComponent {i+1}")
    print(f"Weight = {weights[i]:.4f}")
    print(f"Mean   = {means[i]:.6f}")
    print(f"Std    = {stds[i]:.6f}")

# ============================================================
# 4. NORMALITY TESTS
# ============================================================

print("\n================================================")
print("NORMALITY TESTS")
print("================================================")

# ------------------------------------------------
# Anderson-Darling Test
# ------------------------------------------------
ad_result = anderson(x, dist='norm')

print("\nAnderson-Darling Test")
print("----------------------")
print(f"AD Statistic = {ad_result.statistic:.6f}")

for sig, crit in zip(ad_result.significance_level,
                     ad_result.critical_values):
    print(f"Significance Level = {sig:>5.1f}%"
          f" | Critical Value = {crit:.6f}")

# ------------------------------------------------
# Bera-Jarque Test
# ------------------------------------------------
# Manual calculation

mean_x = np.mean(x)
std_x = np.std(x, ddof=1)

skewness = np.mean(((x - mean_x) / std_x) ** 3)
kurtosis = np.mean(((x - mean_x) / std_x) ** 4)

jb_stat = (n / 6) * (
    skewness**2 +
    ((kurtosis - 3)**2) / 4
)

print("\nBera-Jarque Test")
print("----------------------")
print(f"Skewness = {skewness:.6f}")
print(f"Kurtosis = {kurtosis:.6f}")
print(f"JB Statistic = {jb_stat:.6f}")

# ============================================================
# 5. PDF CURVES
# ============================================================

xmin = np.min(x)
xmax = np.max(x)

grid = np.linspace(xmin, xmax, 3000)

# Empirical KDE
kde = gaussian_kde(x)
empirical_pdf = kde(grid)

# Normal PDF
normal_pdf = norm.pdf(grid, mu_norm, sigma_norm)

# t PDF
t_pdf = t.pdf(grid, df_t, loc_t, scale_t)

# GMM PDF
gmm_pdf = np.zeros_like(grid)

for w, m, s in zip(weights, means, stds):
    gmm_pdf += w * norm.pdf(grid, m, s)

# ============================================================
# 6. FULL DISTRIBUTION PLOT + HISTOGRAM
# ============================================================

plt.figure(figsize=(12, 6))

# Histogram
plt.hist(
    x,
    bins=50,
    density=True,
    alpha=0.4,
    label="Histogram"
)

# Curves
plt.plot(
    grid,
    empirical_pdf,
    linewidth=2.5,
    label="Empirical KDE"
)

plt.plot(
    grid,
    normal_pdf,
    linewidth=2,
    label="Normal"
)

plt.plot(
    grid,
    t_pdf,
    linewidth=2,
    label="t-Distribution"
)

plt.plot(
    grid,
    gmm_pdf,
    linewidth=2,
    label="GMM-2"
)

plt.xlabel("Daily Log Return")
plt.ylabel("Density")

plt.title("Distribution Comparison of SPY Daily Returns")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("distribution_comparison_histogram.png", dpi=300)

# ============================================================
# 7. LEFT TAIL ZOOM (BOTTOM 5%)
# ============================================================
plt.xlim(-0.12, -0.01)
plt.ylim(0, 8)
plt.title("Distribution Comparison of SPY Daily Returns: Left-Tail")
plt.savefig("tail_comparison_bottom5.png", dpi=300)

# ============================================================
# 8. Q-Q PLOTS
# ============================================================

sorted_returns = np.sort(x)

p = (np.arange(1, n + 1) - 0.5) / n

# Normal quantiles
qq_normal = norm.ppf(p, mu_norm, sigma_norm)

# t quantiles
qq_t = t.ppf(p, df_t, loc_t, scale_t)

# GMM quantiles via simulation
gmm_samples = gmm.sample(n_samples=300000)[0].flatten()
gmm_samples = np.sort(gmm_samples)

indices = (p * (len(gmm_samples)-1)).astype(int)
qq_gmm = gmm_samples[indices]

# ============================================================
# 9. COMBINED Q-Q PLOT
# ============================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    qq_normal,
    sorted_returns,
    s=8,
    alpha=0.5,
    label="Normal"
)

plt.scatter(
    qq_t,
    sorted_returns,
    s=8,
    alpha=0.5,
    label="t-Distribution"
)

plt.scatter(
    qq_gmm,
    sorted_returns,
    s=8,
    alpha=0.5,
    label="GMM-2"
)

# Reference line
minv = min(np.min(sorted_returns), np.min(qq_normal))
maxv = max(np.max(sorted_returns), np.max(qq_normal))

plt.plot(
    [minv, maxv],
    [minv, maxv],
    linestyle='--',
    linewidth=2,
    color='black',
)

plt.xlabel("Theoretical Quantiles")
plt.ylabel("Empirical Quantiles")

plt.title("Q-Q Plot Comparison")
plt.xlim(-0.2, 0.2)
plt.ylim(-0.2, 0.2)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("qq_comparison.png", dpi=300)

# ============================================================
# 10. SHOW PLOTS
# ============================================================

plt.show()

print("\n================================================")
print("SAVED FIGURES")
print("================================================")

print("distribution_comparison_histogram.png")
print("tail_comparison_bottom5.png")
print("qq_comparison.png")
