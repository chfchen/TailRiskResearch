import pandas as pd
import numpy as np

from scipy.stats import norm, t
from sklearn.mixture import GaussianMixture

# ============================================================
# 1. LOAD DATA
# ============================================================
filename = "Return.dat" #

df = pd.read_csv(filename)

# Convert "-11.59%" into -0.1159
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

#n = len(x)
n = len(returns)

#X = returns.reshape(-1, 1)

print("Dataset")
print("-------")
print(f"Observations: {n}")
print(f"Mean: {np.mean(returns):.6f}")
print(f"Std:  {np.std(returns, ddof=1):.6f}")

# ============================================================
# 2. NORMAL DISTRIBUTION
# ============================================================
mu_normal = np.mean(returns)
sigma_normal = np.std(returns, ddof=1)

loglik_normal = np.sum(norm.logpdf(returns, loc=mu_normal, scale=sigma_normal))

k_normal = 2  # mean and standard deviation

aic_normal = 2 * k_normal - 2 * loglik_normal
bic_normal = k_normal * np.log(n) - 2 * loglik_normal

# ============================================================
# 3. t-DISTRIBUTION
# ============================================================
df_t, loc_t, scale_t = t.fit(returns)

loglik_t = np.sum(t.logpdf(returns, df_t, loc=loc_t, scale=scale_t))

k_t = 3  # degrees of freedom, location, scale

aic_t = 2 * k_t - 2 * loglik_t
bic_t = k_t * np.log(n) - 2 * loglik_t

# ============================================================
# 4. GMM-2
# ============================================================
gmm2 = GaussianMixture(
    n_components=2,
    covariance_type="full",
    random_state=42
)

gmm2.fit(returns)

loglik_gmm2 = gmm2.score(returns) * n

# For 1D GMM-2:
# 2 means + 2 variances + 1 independent weight = 5 parameters
k_gmm2 = 5

aic_gmm2 = 2 * k_gmm2 - 2 * loglik_gmm2
bic_gmm2 = k_gmm2 * np.log(n) - 2 * loglik_gmm2

weights = gmm2.weights_
means = gmm2.means_.flatten()
stds = np.sqrt(gmm2.covariances_.flatten())

order = np.argsort(means)
weights = weights[order]
means = means[order]
stds = stds[order]

# ============================================================
# 5. OUTPUT RESULTS
# ============================================================
results = pd.DataFrame({
    "Model": ["Normal", "t-distribution", "GMM-2"],
    "Parameters": [k_normal, k_t, k_gmm2],
    "Log-Likelihood": [loglik_normal, loglik_t, loglik_gmm2],
    "AIC": [aic_normal, aic_t, aic_gmm2],
    "BIC": [bic_normal, bic_t, bic_gmm2]
})

print("\nModel Selection Results")
print("-----------------------")
print(results.to_string(index=False))

print("\nNormal Parameters")
print("-----------------")
print(f"Mean: {mu_normal:.6f}")
print(f"Std:  {sigma_normal:.6f}")

print("\nt-Distribution Parameters")
print("-------------------------")
print(f"Degrees of freedom: {df_t:.6f}")
print(f"Location:           {loc_t:.6f}")
print(f"Scale:              {scale_t:.6f}")

print("\nGMM-2 Parameters")
print("----------------")
for i in range(2):
    print(f"Component {i+1}")
    print(f"  Weight: {weights[i]:.6f}")
    print(f"  Mean:   {means[i]:.6f}")
    print(f"  Std:    {stds[i]:.6f}")

# ============================================================
# 6. SAVE TO CSV
# ============================================================
results.to_csv("aic_bic_model_comparison.csv", index=False)

print("\nSaved:")
print("aic_bic_model_comparison.csv")
