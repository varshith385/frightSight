import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

df = pd.read_csv("data/freight_features.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

series = df.set_index("Date")["BDRY_Price_USD"]
series.index = pd.DatetimeIndex(series.index).to_period("M").to_timestamp()
series = series.asfreq("MS").interpolate()

# Walk-forward backtest: start after we have at least 24 months of history
# to fit a reasonable model, then step forward one month at a time,
# forecasting 3 months ahead each time and comparing to actual values.
MIN_TRAIN_SIZE = 24
FORECAST_HORIZON = 3

results = []

for cutoff in range(MIN_TRAIN_SIZE, len(series) - FORECAST_HORIZON):
    train = series.iloc[:cutoff]
    actual_future = series.iloc[cutoff:cutoff + FORECAST_HORIZON]

    try:
        model = ExponentialSmoothing(train, trend="add", seasonal=None, initialization_method="estimated")
        fitted = model.fit()
        forecast = fitted.forecast(FORECAST_HORIZON)

        for i, (pred, actual) in enumerate(zip(forecast.values, actual_future.values)):
            results.append({
                "cutoff_date": train.index[-1],
                "months_ahead": i + 1,
                "predicted": pred,
                "actual": actual,
                "error": pred - actual,
                "abs_error": abs(pred - actual),
                "pct_error": abs(pred - actual) / actual * 100 if actual != 0 else np.nan
            })
    except Exception as e:
        print(f"Skipped cutoff {train.index[-1]}: {e}")

results_df = pd.DataFrame(results)

print("=" * 60)
print("BACKTEST RESULTS — Walk-Forward Validation")
print("=" * 60)
print(f"\nTotal forecast instances tested: {len(results_df)}")
print(f"Backtest period: {results_df['cutoff_date'].min()} to {results_df['cutoff_date'].max()}")

print("\n--- Overall Accuracy ---")
print(f"Mean Absolute Error (MAE): {results_df['abs_error'].mean():.3f}")
print(f"Mean Absolute Percentage Error (MAPE): {results_df['pct_error'].mean():.2f}%")
print(f"Root Mean Squared Error (RMSE): {np.sqrt((results_df['error']**2).mean()):.3f}")

print("\n--- Accuracy by Forecast Horizon (does accuracy degrade further out?) ---")
by_horizon = results_df.groupby("months_ahead").agg(
    MAE=("abs_error", "mean"),
    MAPE=("pct_error", "mean")
).round(3)
print(by_horizon)

# Directional accuracy: did we at least get the trend direction right?
directional = []
for cutoff in results_df["cutoff_date"].unique():
    subset = results_df[results_df["cutoff_date"] == cutoff].sort_values("months_ahead")
    if len(subset) > 0:
        pred_direction = subset.iloc[-1]["predicted"] - series.loc[cutoff]
        actual_direction = subset.iloc[-1]["actual"] - series.loc[cutoff]
        correct = (pred_direction > 0) == (actual_direction > 0)
        directional.append(correct)

directional_accuracy = np.mean(directional) * 100
print(f"\n--- Directional Accuracy ---")
print(f"Correctly predicted UP vs DOWN trend: {directional_accuracy:.1f}% of the time")
print("(50% = random guessing, higher is better)")

# Save chart
plt.figure(figsize=(10, 5))
plt.scatter(results_df["cutoff_date"], results_df["error"], alpha=0.5, s=20)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.title("Backtest Errors Over Time (0 = perfect prediction)")
plt.xlabel("Forecast Made On")
plt.ylabel("Prediction Error (Predicted - Actual)")
plt.tight_layout()
plt.savefig("backtest_errors.png", dpi=150)
print("\nSaved chart to backtest_errors.png")

results_df.to_csv("backtest_results.csv", index=False)
print("Saved detailed results to backtest_results.csv")