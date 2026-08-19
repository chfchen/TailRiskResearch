import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("VaR250.dat", sep=None, engine="python")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

model_columns = {
    "Historical": "Rolling",
    "GARCH": "GARCH",
    "GJR-GARCH": "GJR",
}

# Select the 2008 crisis window - 2008-0901, 2008-12-31; 2020 crisis 02-01, 04-30.
crisis = df[
    (df["Date"] >= "2020-02-01")
    & (df["Date"] <= "2020-04-30")
].copy()

for model, var_column in model_columns.items():
    crisis[f"{model}_Violation"] = (
        crisis["Return"] < -crisis[var_column]
    ).astype(int)

def violation_transition_counts(violations):
    violations = np.asarray(violations, dtype=int)

    previous = violations[:-1]
    current = violations[1:]

    n00 = np.sum((previous == 0) & (current == 0))
    n01 = np.sum((previous == 0) & (current == 1))
    n10 = np.sum((previous == 1) & (current == 0))
    n11 = np.sum((previous == 1) & (current == 1))

    return int(n00), int(n01), int(n10), int(n11)


for model in model_columns:
    violations = crisis[f"{model}_Violation"].to_numpy()

    n00, n01, n10, n11 = violation_transition_counts(violations)

    p_after_no_violation = (
        n01 / (n00 + n01)
        if n00 + n01 > 0
        else np.nan
    )

    p_after_violation = (
        n11 / (n10 + n11)
        if n10 + n11 > 0
        else np.nan
    )

    print(f"\n{model}")
    print("Violations:", violations.sum())
    print("n11:", n11)
    print(
        "P(violation | previous no violation):",
        p_after_no_violation
    )
    print(
        "P(violation | previous violation):",
        p_after_violation
    )

for model, var_column in model_columns.items():

    valid = crisis.dropna(
        subset=["Return", var_column]
    ).copy()

    violations = valid["Return"] < -valid[var_column]

    plt.figure(figsize=(12, 5))

    plt.plot(
        valid["Date"],
        valid["Return"],
        label="Daily return",
        linewidth=0.8
    )

    plt.plot(
        valid["Date"],
        -valid[var_column],
        label=f"{model} 99% VaR threshold",
        linewidth=1.2
    )

    plt.scatter(
        valid.loc[violations, "Date"],
        valid.loc[violations, "Return"],
        label="VaR violation",
        s=25
    )

    plt.axhline(0, linewidth=0.7)
    plt.title(f"{model}: SPY VaR Violations During 2008–2009")
    plt.xlabel("Date")
    plt.ylabel("Daily return")
    plt.legend()
    plt.tight_layout()
    plt.show()

    baseline = df[
    (df["Date"] >= "2008-08-01")
    & (df["Date"] <= "2008-12-31")
].copy()

decay_results = []

for model, var_column in model_columns.items():

    model_crisis = crisis.dropna(
        subset=[var_column]
    ).copy()

    baseline_median = baseline[var_column].median()

    peak_index = model_crisis[var_column].idxmax()
    peak_date = model_crisis.loc[peak_index, "Date"]
    peak_var = model_crisis.loc[peak_index, var_column]

    after_peak = model_crisis.loc[
        model_crisis["Date"] >= peak_date
    ].copy().reset_index(drop=True)

    def days_to_threshold(threshold):
        reached = after_peak[
            after_peak[var_column] <= threshold
        ]

        if reached.empty:
            return np.nan, pd.NaT

        row_number = reached.index[0]
        date_reached = reached.iloc[0]["Date"]

        return int(row_number), date_reached

    days_75, date_75 = days_to_threshold(
        0.75 * peak_var
    )

    days_50, date_50 = days_to_threshold(
        0.50 * peak_var
    )

    days_baseline, date_baseline = days_to_threshold(
        baseline_median
    )

    decay_results.append(
        {
            "Model": model,
            "Peak date": peak_date,
            "Peak VaR": peak_var,
            "Pre-crisis median VaR": baseline_median,
            "Days to 75% of peak": days_75,
            "Days to 50% of peak": days_50,
            "Days to pre-crisis median": days_baseline,
        }
    )

decay_table = pd.DataFrame(decay_results)

print(decay_table)

plt.figure(figsize=(12, 5))

for model, var_column in model_columns.items():

    model_crisis = crisis.dropna(
        subset=[var_column]
    ).copy()

    peak_index = model_crisis[var_column].idxmax()
    peak_date = model_crisis.loc[peak_index, "Date"]
    peak_var = model_crisis.loc[peak_index, var_column]

    after_peak = model_crisis[
        model_crisis["Date"] >= peak_date
    ].copy()

    # Compare the first 250 trading days after each peak.
    after_peak = after_peak.iloc[:250].reset_index(drop=True)

    normalized_var = after_peak[var_column] / peak_var

    plt.plot(
        after_peak.index,
        normalized_var,
        label=model
    )

plt.axhline(0.50, linestyle="--", linewidth=0.8)
plt.xlabel("Trading days after each model's VaR peak")
plt.ylabel("VaR relative to peak")
plt.title("VaR Decay Following the 2008 Crisis Peak")
plt.legend()
plt.tight_layout()
plt.show()
