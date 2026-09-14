import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("data/freight_features.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

series = df.set_index("Date")["BDRY_Price_USD"]
series = series[~series.index.duplicated(keep='last')]
series = series.asfreq("W-MON").interpolate()

MIN_TRAIN_SIZE = 52
FORECAST_HORIZON = 12


def method_holt_winters(train, horizon):
    model = ExponentialSmoothing(train, trend="add", seasonal=None, initialization_method="estimated")
    fitted = model.fit()
    return fitted.forecast(horizon).values


def method_naive_last_value(train, horizon):
    """Forecast = last known value, repeated. The simplest possible baseline."""
    return np.repeat(train.iloc[-1], horizon)


def method_moving_average(train, horizon):
    """Forecast = average of last 3 months, repeated."""
    avg = train.iloc[-3:].mean()
    return np.repeat(avg, horizon)


def method_damped_trend(train, horizon):
    """Like Holt-Winters but with a damped trend - trend effect weakens
    over the forecast horizon instead of extrapolating linearly forever.
    Often better for volatile, mean-reverting series."""
    model = ExponentialSmoothing(train, trend="add", damped_trend=True, seasonal=None, initialization_method="estimated")
    fitted = model.fit()
    return fitted.forecast(horizon).values


def method_arima(train, horizon):
    """ARIMA(1,1,1) - a standard, more flexible time-series model."""
    model = ARIMA(train, order=(1, 1, 1))
    fitted = model.fit()
    return fitted.forecast(horizon).values


methods = {
    "Holt-Winters (current)": method_holt_winters,
    "Naive (last value)": method_naive_last_value,
    "Moving Average (3mo)": method_moving_average,
    "Damped Trend": method_damped_trend,
    "ARIMA(1,1,1)": method_arima,
}

all_results = {name: [] for name in methods}

for cutoff in range(MIN_TRAIN_SIZE, len(series) - FORECAST_HORIZON):
    train = series.iloc[:cutoff]
    actual_future = series.iloc[cutoff:cutoff + FORECAST_HORIZON]
    current_value = train.iloc[-1]

    for name, method_fn in methods.items():
        try:
            forecast = method_fn(train, FORECAST_HORIZON)
            final_pred = forecast[-1]
            final_actual = actual_future.iloc[-1]

            pred_direction = final_pred - current_value
            actual_direction = final_actual - current_value
            correct_direction = (pred_direction > 0) == (actual_direction > 0)

            mae = np.mean(np.abs(forecast - actual_future.values))
            mape = np.mean(np.abs(forecast - actual_future.values) / actual_future.values) * 100

            all_results[name].append({
                "mae": mae, "mape": mape, "correct_direction": correct_direction
            })
        except Exception:
            continue

print("=" * 70)
print("METHOD COMPARISON — 3-Month Ahead Forecast Backtest")
print("=" * 70)
print(f"\n{'Method':<28}{'MAE':>8}{'MAPE':>10}{'Dir. Accuracy':>16}")
print("-" * 70)

summary = []
for name, results in all_results.items():
    if len(results) == 0:
        continue
    results_df = pd.DataFrame(results)
    mae = results_df["mae"].mean()
    mape = results_df["mape"].mean()
    dir_acc = results_df["correct_direction"].mean() * 100
    summary.append({"Method": name, "MAE": mae, "MAPE": mape, "Directional_Accuracy": dir_acc})
    print(f"{name:<28}{mae:>8.3f}{mape:>9.2f}%{dir_acc:>15.1f}%")

summary_df = pd.DataFrame(summary).sort_values("Directional_Accuracy", ascending=False)
print("\n" + "=" * 70)
print("Best method by directional accuracy:", summary_df.iloc[0]["Method"])
print("=" * 70)

summary_df.to_csv("forecast_method_comparison.csv", index=False)
print("\nSaved to forecast_method_comparison.csv")