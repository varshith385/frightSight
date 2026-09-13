import joblib
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = joblib.load("freight_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_list = joblib.load("model_features.pkl")

vessels = [
    {"name": "Handysize", "capacity_mt": 35000, "days_to_port": 20, "daily_rate_multiplier": 0.55},
    {"name": "Supramax",  "capacity_mt": 55000, "days_to_port": 16, "daily_rate_multiplier": 0.75},
    {"name": "Panamax",   "capacity_mt": 75000, "days_to_port": 14, "daily_rate_multiplier": 1.00},
]

BDRY_MIN, BDRY_MAX = 5, 30
RATE_MIN, RATE_MAX = 8000, 25000

def bdry_to_daily_rate(bdry_value):
    bdry_clamped = max(BDRY_MIN, min(BDRY_MAX, bdry_value))
    ratio = (bdry_clamped - BDRY_MIN) / (BDRY_MAX - BDRY_MIN)
    return RATE_MIN + ratio * (RATE_MAX - RATE_MIN)

def predict_bdry(input_dict):
    """input_dict must contain all keys in feature_list."""
    input_df = pd.DataFrame([input_dict])[feature_list]
    input_scaled = scaler.transform(input_df)
    return model.predict(input_scaled)[0]

def get_forecast_trend():
    hist = pd.read_csv("data/freight_features.csv", parse_dates=["Date"])
    hist = hist.sort_values("Date").reset_index(drop=True)
    series = hist.set_index("Date")["BDRY_Price_USD"]
    series.index = pd.DatetimeIndex(series.index).to_period("M").to_timestamp()
    series = series.asfreq("MS")
    series = series.interpolate()  # fill any small gaps instead of leaving NaN

    hw_model = ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated")
    fitted = hw_model.fit()
    forecast = fitted.forecast(3)

    current_value = series.iloc[-1]
    forecast_avg = forecast.mean()
    pct_change = ((forecast_avg - current_value) / current_value) * 100

    if pct_change > 5:
        trend, timing_advice = "RISING", "Book NOW — freight rates trending upward."
    elif pct_change < -5:
        trend, timing_advice = "FALLING", "Consider WAITING — freight rates trending downward."
    else:
        trend, timing_advice = "STABLE", "Market stable — booking timing is flexible."

    return {
        "current_bdry": round(current_value, 2),
        "forecast_3m_avg": round(forecast_avg, 2),
        "pct_change": round(pct_change, 1),
        "trend": trend,
        "timing_advice": timing_advice
    }


def recommend_vessel(cargo_quantity_mt, predicted_bdry):
    daily_rate_panamax_equiv = bdry_to_daily_rate(predicted_bdry)
    results = []
    for v in vessels:
        vessel_daily_rate = daily_rate_panamax_equiv * v["daily_rate_multiplier"]
        trips_needed = -(-cargo_quantity_mt // v["capacity_mt"])
        total_days = v["days_to_port"] * trips_needed
        total_cost = vessel_daily_rate * total_days
        cost_per_mt = total_cost / cargo_quantity_mt
        results.append({
            "Vessel": v["name"], "Trips_Needed": trips_needed,
            "Daily_Rate_USD": round(vessel_daily_rate, 2),
            "Total_Cost_USD": round(total_cost, 2),
            "Cost_per_MT_USD": round(cost_per_mt, 2),
            "Estimated_Days": total_days
        })
    df_results = pd.DataFrame(results)
    best = df_results.loc[df_results["Total_Cost_USD"].idxmin()]
    return df_results, best


if __name__ == "__main__":
    latest = pd.read_csv("data/freight_features.csv").iloc[-1]
    input_dict = {f: latest[f] for f in feature_list}

    predicted_bdry = predict_bdry(input_dict)
    forecast_info = get_forecast_trend()
    comparison, best = recommend_vessel(50000, predicted_bdry)

    print("=" * 55)
    print("FRIGHTSIGHT — FULL DECISION REPORT")
    print("=" * 55)
    print(f"\nPredicted BDRY: {predicted_bdry:.2f}")
    print(f"Current BDRY: {forecast_info['current_bdry']} | Forecast Avg: {forecast_info['forecast_3m_avg']} | Trend: {forecast_info['trend']}")
    print(f"\n📅 {forecast_info['timing_advice']}")
    print("\nVessel Comparison:")
    print(comparison.to_string(index=False))
    print(f"\n🚢 RECOMMENDATION: {best['Vessel']} — ${best['Total_Cost_USD']:,.2f} total, {best['Estimated_Days']} days")