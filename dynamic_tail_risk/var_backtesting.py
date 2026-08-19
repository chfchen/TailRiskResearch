"""
VaR backtesting:
1. Kupiec unconditional-coverage test
2. Christoffersen independence test
3. Christoffersen conditional-coverage test
4. Quantile loss

VaR columns must contain positive loss values.
Example:
    ReturnPct = -3.00
    VaR = 2.50
This is a violation because -3.00 < -2.50.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_FILE = Path("VaR1250.csv")
DATE_COLUMN = "Date"
RETURN_COLUMN = "Return"

VAR_COLUMNS = {
    "Historical 1250d Rolling": "Rolling",
    "GARCH(1,1)-Student-t": "GARCH",
    "GJR-GARCH(1,1)-Student-t": "GJR",
}

TAIL_PROBABILITY = 0.01
OUTPUT_FILE = Path("VaR_Backtest_Results.xlsx")

# ============================================================
# 2. READ AND CLEAN DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find: {INPUT_FILE.resolve()}"
    )

# Automatically detects comma, tab, or whitespace delimiters.
df = pd.read_csv(
    INPUT_FILE,
    sep=None,
    engine="python"
)

df.columns = df.columns.str.strip()

required_columns = [
    DATE_COLUMN,
    RETURN_COLUMN,
    *VAR_COLUMNS.values()
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

df[DATE_COLUMN] = pd.to_datetime(
    df[DATE_COLUMN],
    errors="coerce"
)

numeric_columns = [
    RETURN_COLUMN,
    *VAR_COLUMNS.values()
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = (
    df.dropna(subset=[DATE_COLUMN, RETURN_COLUMN])
      .sort_values(DATE_COLUMN)
      .drop_duplicates(subset=DATE_COLUMN, keep="last")
      .reset_index(drop=True)
)

# Use one common evaluation period for all three models.
# The first 250 Rolling values are expected to be missing.
common_df = df.dropna(
    subset=[
        RETURN_COLUMN,
        *VAR_COLUMNS.values()
    ]
).copy()

print("Full observations:", len(df))
print("Common backtest observations:", len(common_df))
print("First common date:", common_df[DATE_COLUMN].min())
print("Last common date:", common_df[DATE_COLUMN].max())


# ============================================================
# 3. SUPPORTING FUNCTIONS
# ============================================================

def xlogy(count: int, probability: float) -> float:
    """
    Calculate count * log(probability), treating 0*log(0) as zero.
    """

    if count == 0:
        return 0.0

    if probability <= 0:
        return -np.inf

    return count * np.log(probability)


def kupiec_test(
    violations: np.ndarray,
    expected_probability: float
) -> tuple[float, float]:
    """
    Kupiec unconditional-coverage test.

    H0:
        Actual violation probability equals expected_probability.

    Returns:
        LR statistic and p-value.
    """

    violations = np.asarray(violations, dtype=int)

    n = len(violations)
    x = int(violations.sum())

    if n == 0:
        return np.nan, np.nan

    observed_probability = x / n

    log_likelihood_null = (
        xlogy(x, expected_probability)
        + xlogy(n - x, 1.0 - expected_probability)
    )

    log_likelihood_alternative = (
        xlogy(x, observed_probability)
        + xlogy(n - x, 1.0 - observed_probability)
    )

    lr_uc = -2.0 * (
        log_likelihood_null
        - log_likelihood_alternative
    )

    p_value = chi2.sf(lr_uc, df=1)

    return lr_uc, p_value


def transition_counts(
    violations: np.ndarray
) -> tuple[int, int, int, int]:
    """
    Count consecutive violation transitions:

    n00: 0 followed by 0
    n01: 0 followed by 1
    n10: 1 followed by 0
    n11: 1 followed by 1
    """

    previous = violations[:-1]
    current = violations[1:]

    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))

    return n00, n01, n10, n11


def christoffersen_independence_test(
    violations: np.ndarray
) -> tuple[float, float, int, int, int, int]:
    """
    Christoffersen independence test.

    H0:
        Violations are independent through time.

    Returns:
        LR statistic, p-value, and transition counts.
    """

    violations = np.asarray(violations, dtype=int)

    if len(violations) < 2:
        return np.nan, np.nan, 0, 0, 0, 0

    n00, n01, n10, n11 = transition_counts(violations)

    total_transitions = n00 + n01 + n10 + n11

    pi = (
        (n01 + n11) / total_transitions
        if total_transitions > 0
        else 0.0
    )

    pi_01 = (
        n01 / (n00 + n01)
        if (n00 + n01) > 0
        else 0.0
    )

    pi_11 = (
        n11 / (n10 + n11)
        if (n10 + n11) > 0
        else 0.0
    )

    # Restricted model: one common violation probability.
    log_likelihood_independent = (
        xlogy(n01 + n11, pi)
        + xlogy(n00 + n10, 1.0 - pi)
    )

    # Alternative: violation probability depends on prior state.
    log_likelihood_markov = (
        xlogy(n01, pi_01)
        + xlogy(n00, 1.0 - pi_01)
        + xlogy(n11, pi_11)
        + xlogy(n10, 1.0 - pi_11)
    )

    lr_ind = -2.0 * (
        log_likelihood_independent
        - log_likelihood_markov
    )

    p_value = chi2.sf(lr_ind, df=1)

    return (
        lr_ind,
        p_value,
        n00,
        n01,
        n10,
        n11
    )


def quantile_loss(
    returns: np.ndarray,
    positive_var: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Pinball/quantile loss for a lower-tail VaR forecast.

    positive_var:
        Positive loss VaR, such as 2.5 for 2.5%.

    return quantile:
        q_t = -VaR_t

    Lower average loss is better.
    """

    return_quantile = -positive_var

    error = returns - return_quantile

    losses = np.where(
        error < 0,
        (alpha - 1.0) * error,
        alpha * error
    )

    return losses


# ============================================================
# 4. RUN TESTS FOR EACH MODEL
# ============================================================

summary_rows = []
daily_output = df[[DATE_COLUMN, RETURN_COLUMN]].copy()

for model_name, var_column in VAR_COLUMNS.items():
    model_data = common_df[
        [DATE_COLUMN, RETURN_COLUMN, var_column]
    ].copy()

    model_data[var_column] = pd.to_numeric(
        model_data[var_column],
        errors="coerce"
    )

    model_data = model_data.dropna()

    returns = model_data[RETURN_COLUMN].to_numpy(dtype=float)
    var_values = model_data[var_column].to_numpy(dtype=float)

    # Violation:
    # actual return is below the negative VaR threshold.
    violations = (
        returns < -var_values
    ).astype(int)

    n = len(violations)
    x = int(violations.sum())

    violation_rate = x / n
    expected_violations = n * TAIL_PROBABILITY
    violation_ratio = violation_rate / TAIL_PROBABILITY

    # Step 1: Kupiec test
    lr_uc, kupiec_p = kupiec_test(
        violations,
        TAIL_PROBABILITY
    )

    # Step 2: Christoffersen independence
    (
        lr_ind,
        independence_p,
        n00,
        n01,
        n10,
        n11
    ) = christoffersen_independence_test(violations)

    # Step 3: Conditional coverage
    lr_cc = lr_uc + lr_ind
    conditional_coverage_p = chi2.sf(lr_cc, df=2)

    # Step 4: Quantile loss
    qloss = quantile_loss(
        returns,
        var_values,
        TAIL_PROBABILITY
    )

    mean_quantile_loss = float(np.mean(qloss))
    total_quantile_loss = float(np.sum(qloss))

    summary_rows.append(
        {
            "Model": model_name,
            "Observations": n,
            "Violations": x,
            "Expected violations": expected_violations,
            "Violation rate": violation_rate,
            "Violation ratio": violation_ratio,

            "Kupiec LR": lr_uc,
            "Kupiec p-value": kupiec_p,
            "Kupiec acceptable at 5%": kupiec_p >= 0.05,

            "n00": n00,
            "n01": n01,
            "n10": n10,
            "n11": n11,

            "Christoffersen Independence LR": lr_ind,
            "Independence p-value": independence_p,
            "Independent at 5%": independence_p >= 0.05,

            "Conditional Coverage LR": lr_cc,
            "Conditional Coverage p-value":
                conditional_coverage_p,
            "Conditional coverage acceptable at 5%":
                conditional_coverage_p >= 0.05,

            "Mean quantile loss": mean_quantile_loss,
            "Total quantile loss": total_quantile_loss,
        }
    )

    # Add model-specific values to daily output.
    aligned_index = model_data.index

    daily_output.loc[
        aligned_index,
        f"{model_name} VaR"
    ] = var_values

    daily_output.loc[
        aligned_index,
        f"{model_name} Violation"
    ] = violations

    daily_output.loc[
        aligned_index,
        f"{model_name} Quantile Loss"
    ] = qloss


# ============================================================
# 5. CREATE RESULTS TABLE
# ============================================================

results = pd.DataFrame(summary_rows)

results["Quantile-loss rank"] = (
    results["Mean quantile loss"]
    .rank(method="min", ascending=True)
    .astype(int)
)

results["Violation-ratio distance from 1"] = (
    results["Violation ratio"] - 1.0
).abs()

results["Coverage rank"] = (
    results["Violation-ratio distance from 1"]
    .rank(method="min", ascending=True)
    .astype(int)
)

results = results.sort_values(
    ["Quantile-loss rank", "Coverage rank"]
).reset_index(drop=True)

print("\nVaR BACKTEST RESULTS")
print("=" * 120)

display_columns = [
    "Model",
    "Observations",
    "Violations",
    "Violation ratio",
    "Kupiec p-value",
    "Independence p-value",
    "Conditional Coverage p-value",
    "Mean quantile loss",
    "Quantile-loss rank",
]

print(
    results[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}"
    )
)


# ============================================================
# 6. EXPORT TO EXCEL
# ============================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    results.to_excel(
        writer,
        sheet_name="Backtest Summary",
        index=False
    )

    daily_output.to_excel(
        writer,
        sheet_name="Daily Tests",
        index=False
    )

print(f"\nSaved results to: {OUTPUT_FILE.resolve()}")
