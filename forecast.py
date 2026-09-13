import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("data/freight_features.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

series = df.set_index("Date")["BDRY_Price_USD"]

# --- Holt-Winters Exponential Smoothing ---
# A well-established, explainable time-series forecasting method
# (not a black box) - models level + trend in the data.
# We use additive trend since BDRY doesn't show strong seasonality
# at this data length (only ~7 years monthly).
model = ExponentialSmoothing(
    series,
    trend="add",
    seasonal=None,
    initialization_method="estimated"
)
fitted = model.fit()

# Forecast next 3 months
forecast = fitted.forecast(3)

print("Historical BDRY (last 6 months):")
print(series.tail(6))

print("\nForecast (next 3 months):")
print(forecast)

# --- Trend direction and recommendation ---
current_value = series.iloc[-1]
forecast_avg = forecast.mean()
pct_change = ((forecast_avg - current_value) / current_value) * 100

print(f"\nCurrent BDRY: {current_value:.2f}")
print(f"3-Month Forecast Average: {forecast_avg:.2f}")
print(f"Projected Change: {pct_change:+.1f}%")

if pct_change > 5:
    recommendation = "Freight market trending UP — recommend booking NOW before rates rise further."
elif pct_change < -5:
    recommendation = "Freight market trending DOWN — recommend WAITING for better rates."
else:
    recommendation = "Freight market relatively STABLE — booking timing is flexible."

print(f"\nRECOMMENDATION: {recommendation}")

# --- Save a chart for your PPT ---
plt.figure(figsize=(10, 5))
plt.plot(series.index, series.values, label="Historical BDRY", color="#0d2b52")
plt.plot(forecast.index, forecast.values, label="3-Month Forecast", color="#2e8b57", linestyle="--", marker="o")
plt.title("frightSight — Dry Bulk Freight Market Forecast")
plt.xlabel("Date")
plt.ylabel("BDRY Index Value")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("forecast_chart.png", dpi=150)
print("\nSaved chart to forecast_chart.png")

joblib.dump(fitted, "forecast_model.pkl")
print("Saved forecasting model to forecast_model.pkl")