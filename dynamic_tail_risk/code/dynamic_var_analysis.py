"""
Dynamic VaR analysis for SPY
Data file: SPYLogReturn.dat

Models:
1. 250-day Historical VaR
2. GARCH(1,1)-Student-t VaR
3. GJR-GARCH(1,1)-Student-t VaR
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import t as student_t

# ============================================================
# 1. USER SETTINGS
# ============================================================
DATA_FILE = Path("SPYLogReturn.dat")
HISTORICAL_WINDOW = 250
# Main confidence level
CONFIDENCE_LEVEL = 0.99
TAIL_PROBABILITY = 1.0 - CONFIDENCE_LEVEL  # 0.01 for 99% VaR

OUTPUT_CSV = Path("SPY_Dynamic_VaR_Results.csv")
OUTPUT_EXCEL = Path("SPY_Dynamic_VaR_Results.xlsx")
OUTPUT_FIGURE = Path("SPY_Dynamic_99pct_VaR.png")

# ============================================================
# 2. READ AND CLEAN THE DATA
# ============================================================
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_FILE.resolve()}.\n"
        "Place SPYLogReturn.dat in the same folder as this script."
    )
# sep=None asks pandas to detect tab, comma, or whitespace separation.
df = pd.read_csv(
    DATA_FILE,
    sep=None,
    engine="python"
)
# Remove extra spaces from column names.
df.columns = df.columns.str.strip()
required_columns = {"Date", "Return"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}.\n"
        f"Columns found: {df.columns.tolist()}"
    )
# Convert columns to proper types.
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Return"] = pd.to_numeric(df["Return"], errors="coerce")
# Remove invalid or missing observations.
df = df.dropna(subset=["Date", "Return"]).copy()
# Arrange observations from oldest to newest.
df = (
    df.sort_values("Date")
      .drop_duplicates(subset="Date", keep="last")
      .reset_index(drop=True)
)

if len(df) < HISTORICAL_WINDOW + 1:
    raise ValueError(
        f"At least {HISTORICAL_WINDOW + 1} valid observations are required."
    )

# Check whether returns appear to be decimals or percentages.
# Decimal format example: 0.012 = 1.2%
# Percentage format example: 1.2 = 1.2%
median_absolute_return = df["Return"].abs().median()

if median_absolute_return < 0.10:
    print("Returns appear to be decimals. Multiplying by 100.")
    df["ReturnPct"] = df["Return"] * 100.0
else:
    print("Returns appear to already be percentages.")
    df["ReturnPct"] = df["Return"]

print("\nData summary")
print("-" * 60)
print(f"First date:          {df['Date'].min().date()}")
print(f"Last date:           {df['Date'].max().date()}")
print(f"Number of returns:   {len(df):,}")
print(f"Mean daily return:   {df['ReturnPct'].mean():.4f}%")
print(f"Daily volatility:    {df['ReturnPct'].std(ddof=1):.4f}%")

# ============================================================
# 3. 250-DAY HISTORICAL VaR
# ============================================================
historical_return_quantile = (
    df["ReturnPct"]
    .shift(1)
    .rolling(
        window=HISTORICAL_WINDOW,
        min_periods=HISTORICAL_WINDOW
    )
    .quantile(TAIL_PROBABILITY)
)
df["Historical_VaR_99"] = -historical_return_quantile

# ============================================================
# 4. FUNCTION TO ESTIMATE STUDENT-t GARCH VaR
# ============================================================
def estimate_student_t_garch_var(
    returns_pct: pd.Series,
    asymmetric: bool
):
    """
    Estimate either:

    Standard GARCH(1,1)-t:
        sigma_t^2 = omega
                    + alpha * epsilon_(t-1)^2
                    + beta * sigma_(t-1)^2

    GJR-GARCH(1,1)-t:
        sigma_t^2 = omega
                    + alpha * epsilon_(t-1)^2
                    + gamma * I_(epsilon<0) * epsilon_(t-1)^2
                    + beta * sigma_(t-1)^2

    Parameters
    ----------
    returns_pct:
        Daily log returns expressed in percentage points.

    asymmetric:
        False gives standard GARCH.
        True gives GJR-GARCH.

    Returns
    -------
    fitted_result:
        Fitted arch model result.

    conditional_volatility:
        Estimated daily conditional volatility.

    conditional_var:
        Estimated daily 99% VaR as a positive loss percentage.
    """

    asymmetry_order = 1 if asymmetric else 0

    model = arch_model(
        returns_pct,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=asymmetry_order,
        q=1,
        power=2.0,
        dist="t",
        rescale=False
    )

    fitted_result = model.fit(
        disp="off",
        show_warning=True
    )

    parameters = fitted_result.params

    # Constant conditional mean return.
    conditional_mean = parameters["mu"]

    # Estimated degrees of freedom for Student-t innovations.
    degrees_of_freedom = parameters["nu"]

    if degrees_of_freedom <= 2:
        raise ValueError(
            "Estimated Student-t degrees of freedom must exceed 2 "
            "for the variance to exist."
        )   
    standardized_t_quantile = (
        student_t.ppf(
            TAIL_PROBABILITY,
            df=degrees_of_freedom
        )
        * np.sqrt(
            (degrees_of_freedom - 2.0) /
            degrees_of_freedom
        )
    )

    conditional_volatility = pd.Series(
        fitted_result.conditional_volatility,
        index=returns_pct.index
    )

    # Lower return quantile:
    # q_return,t = mu + sigma_t * q_t
    #
    # Positive-loss VaR:
    # VaR_t = -q_return,t

    conditional_var = -(
        conditional_mean
        + conditional_volatility * standardized_t_quantile
    )

    return (
        fitted_result,
        conditional_volatility,
        conditional_var
    )
    
# ============================================================
# 5. STANDARD GARCH(1,1)-STUDENT-t
# ============================================================
garch_result, garch_volatility, garch_var = (
    estimate_student_t_garch_var(
        returns_pct=df["ReturnPct"],
        asymmetric=False
    )
)

df["GARCH_t_Volatility"] = garch_volatility
df["GARCH_t_VaR_99"] = garch_var

# ============================================================
# 6. GJR-GARCH(1,1)-STUDENT-t
# ============================================================
gjr_result, gjr_volatility, gjr_var = (
    estimate_student_t_garch_var(
        returns_pct=df["ReturnPct"],
        asymmetric=True
    )
)

df["GJR_GARCH_t_Volatility"] = gjr_volatility
df["GJR_GARCH_t_VaR_99"] = gjr_var

# ============================================================
# 7. IDENTIFY VaR VIOLATIONS
# ============================================================
df["ActualLossPct"] = -df["ReturnPct"]

df["Historical_Violation"] = (
    df["ActualLossPct"] > df["Historical_VaR_99"]
)

df["GARCH_t_Violation"] = (
    df["ActualLossPct"] > df["GARCH_t_VaR_99"]
)

df["GJR_GARCH_t_Violation"] = (
    df["ActualLossPct"] > df["GJR_GARCH_t_VaR_99"]
)

# Missing Historical VaR observations should not be classified as False.
df.loc[
    df["Historical_VaR_99"].isna(),
    "Historical_Violation"
] = np.nan

# ============================================================
# 8. PRINT MODEL RESULTS
# ============================================================
print("\n\nSTANDARD GARCH(1,1)-STUDENT-t")
print("=" * 70)
print(garch_result.summary())

print("\n\nGJR-GARCH(1,1)-STUDENT-t")
print("=" * 70)
print(gjr_result.summary())

# ============================================================
# 9. EXTRACT IMPORTANT MODEL PARAMETERS
# ============================================================
def safely_get_parameter(parameters, possible_names):
    """Return the first available parameter among possible names."""

    for name in possible_names:
        if name in parameters.index:
            return float(parameters[name])

    return np.nan

garch_alpha = safely_get_parameter(
    garch_result.params,
    ["alpha[1]"]
)

garch_beta = safely_get_parameter(
    garch_result.params,
    ["beta[1]"]
)

garch_persistence = garch_alpha + garch_beta

gjr_alpha = safely_get_parameter(
    gjr_result.params,
    ["alpha[1]"]
)

gjr_gamma = safely_get_parameter(
    gjr_result.params,
    ["gamma[1]"]
)

gjr_beta = safely_get_parameter(
    gjr_result.params,
    ["beta[1]"]
)

gjr_persistence = (
    gjr_alpha
    + gjr_beta
    + 0.5 * gjr_gamma
)

parameter_table = pd.DataFrame(
    {
        "Model": [
            "GARCH(1,1)-Student-t",
            "GJR-GARCH(1,1)-Student-t"
        ],
        "Omega": [
            safely_get_parameter(garch_result.params, ["omega"]),
            safely_get_parameter(gjr_result.params, ["omega"])
        ],
        "Alpha": [
            garch_alpha,
            gjr_alpha
        ],
        "Gamma_Asymmetry": [
            np.nan,
            gjr_gamma
        ],
        "Beta": [
            garch_beta,
            gjr_beta
        ],
        "Student_t_df": [
            safely_get_parameter(garch_result.params, ["nu"]),
            safely_get_parameter(gjr_result.params, ["nu"])
        ],
        "Persistence": [
            garch_persistence,
            gjr_persistence
        ],
        "AIC": [
            garch_result.aic,
            gjr_result.aic
        ],
        "BIC": [
            garch_result.bic,
            gjr_result.bic
        ]
    }
)
print("\n\nMODEL PARAMETER COMPARISON")
print("=" * 70)
print(parameter_table.to_string(index=False))

# ============================================================
# 10. SIMPLE VaR SUMMARY
# ============================================================
def create_var_summary(
    data: pd.DataFrame,
    var_column: str,
    violation_column: str,
    model_name: str
):
    valid = data[[var_column, violation_column]].dropna()

    number_of_observations = len(valid)
    number_of_violations = int(valid[violation_column].sum())

    violation_rate = (
        number_of_violations / number_of_observations
        if number_of_observations > 0
        else np.nan
    )

    return {
        "Model": model_name,
        "Valid observations": number_of_observations,
        "Mean VaR (%)": valid[var_column].mean(),
        "Median VaR (%)": valid[var_column].median(),
        "Minimum VaR (%)": valid[var_column].min(),
        "Maximum VaR (%)": valid[var_column].max(),
        "VaR violations": number_of_violations,
        "Violation rate": violation_rate,
        "Expected violation rate": TAIL_PROBABILITY
    }

var_summary = pd.DataFrame(
    [
        create_var_summary(
            df,
            "Historical_VaR_99",
            "Historical_Violation",
            "250-day Historical VaR"
        ),
        create_var_summary(
            df,
            "GARCH_t_VaR_99",
            "GARCH_t_Violation",
            "GARCH(1,1)-Student-t"
        ),
        create_var_summary(
            df,
            "GJR_GARCH_t_VaR_99",
            "GJR_GARCH_t_Violation",
            "GJR-GARCH(1,1)-Student-t"
        )
    ]
)

print("\n\nVaR SUMMARY")
print("=" * 70)
print(var_summary.to_string(index=False))

# ============================================================
# 11. SAVE RESULTS
# ============================================================
output_columns = [
    "Date",
    "Return",
    "ReturnPct",
    "ActualLossPct",
    "Historical_VaR_99",
    "GARCH_t_Volatility",
    "GARCH_t_VaR_99",
    "GJR_GARCH_t_Volatility",
    "GJR_GARCH_t_VaR_99",
    "Historical_Violation",
    "GARCH_t_Violation",
    "GJR_GARCH_t_Violation"
]

df[output_columns].to_csv(
    OUTPUT_CSV,
    index=False,
    date_format="%m/%d/%Y"
)

with pd.ExcelWriter(
    OUTPUT_EXCEL,
    engine="openpyxl"
) as writer:

    df[output_columns].to_excel(
        writer,
        sheet_name="Daily VaR",
        index=False
    )

    parameter_table.to_excel(
        writer,
        sheet_name="Model Parameters",
        index=False
    )

    var_summary.to_excel(
        writer,
        sheet_name="VaR Summary",
        index=False
    )

print(f"\nSaved daily results to: {OUTPUT_CSV.resolve()}")
print(f"Saved Excel workbook to: {OUTPUT_EXCEL.resolve()}")

# ============================================================
# 12. CREATE THE MAIN FIGURE
# ============================================================
plot_df = df.dropna(
    subset=[
        "Historical_VaR_99",
        "GARCH_t_VaR_99",
        "GJR_GARCH_t_VaR_99"
    ]
).copy()

plt.figure(figsize=(14, 7))

# Actual losses are plotted only when positive.
positive_losses = plot_df["ActualLossPct"].clip(lower=0)

plt.plot(
    plot_df["Date"],
    positive_losses,
    linewidth=0.6,
    alpha=0.45,
    label="Actual daily loss"
)

plt.plot(
    plot_df["Date"],
    plot_df["Historical_VaR_99"],
    linewidth=1.0,
    label="250-day Historical 99% VaR"
)

plt.plot(
    plot_df["Date"],
    plot_df["GARCH_t_VaR_99"],
    linewidth=1.0,
    label="GARCH(1,1)-t 99% VaR"
)

plt.plot(
    plot_df["Date"],
    plot_df["GJR_GARCH_t_VaR_99"],
    linewidth=1.0,
    label="GJR-GARCH(1,1)-t 99% VaR"
)

plt.title(
    "Dynamic One-Day 99% VaR for SPY, 1996–2025"
)
plt.xlabel("Date")
plt.ylabel("Loss / VaR (%)")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved figure to: {OUTPUT_FIGURE.resolve()}")
# Initial Variance for Excel Input
print(garch_result.conditional_volatility.iloc[0] ** 2)
print(gjr_result.conditional_volatility.iloc[0] ** 2)
