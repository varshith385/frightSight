from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import requests
import json
from statsmodels.tsa.arima.model import ARIMA
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
model = joblib.load("freight_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_list = joblib.load("model_features.pkl")
model_rmse = joblib.load("model_rmse.pkl")

API_NINJAS_KEY = os.getenv("API_NINJAS_KEY")

vessels = [
    {"name": "Handysize", "capacity_mt": 35000, "days_to_port": 20, "daily_rate_multiplier": 0.55, "speed_knots": 13, "cii_band": "B"},
    {"name": "Supramax",  "capacity_mt": 55000, "days_to_port": 16, "daily_rate_multiplier": 0.75, "speed_knots": 14, "cii_band": "B"},
    {"name": "Panamax",   "capacity_mt": 75000, "days_to_port": 14, "daily_rate_multiplier": 1.00, "speed_knots": 14, "cii_band": "C"},
]

BDRY_MIN, BDRY_MAX = 5, 30
RATE_MIN, RATE_MAX = 8000, 25000

latest_data = pd.read_csv("data/freight_features.csv").iloc[-1]
route_data = pd.read_csv("data/route_data.csv")
tariff_data = pd.read_csv("data/tax_tariff_data.csv")

REFERENCE_DISTANCE_NM = 2800
VESSEL_SPEED_KNOTS = 14


def get_route_distance(origin_port, destination_port):
    match = route_data[(route_data["Origin_Port"] == origin_port) & (route_data["Destination_Port"] == destination_port)]
    if len(match) == 0:
        return REFERENCE_DISTANCE_NM
    return match.iloc[0]["Distance_NM"]


def get_distance_multiplier(distance_nm):
    return distance_nm / REFERENCE_DISTANCE_NM


def get_tariff_info(commodity):
    match = tariff_data[tariff_data["Commodity"] == commodity]
    if len(match) == 0:
        return 0, 5
    row = match.iloc[0]
    return row["Basic_Customs_Duty_Percent"], row["IGST_Percent"]


def calculate_landed_cost(freight_cost_total, cargo_value_usd, bcd_pct, igst_pct):
    bcd_amount = cargo_value_usd * (bcd_pct / 100)
    assessable_value = cargo_value_usd + freight_cost_total + bcd_amount
    igst_amount = assessable_value * (igst_pct / 100)
    total_landed_cost = cargo_value_usd + freight_cost_total + bcd_amount + igst_amount
    return {
        "bcd_amount": round(bcd_amount, 2),
        "igst_amount": round(igst_amount, 2),
        "total_landed_cost": round(total_landed_cost, 2)
    }


def bdry_to_daily_rate(bdry_value):
    bdry_clamped = max(BDRY_MIN, min(BDRY_MAX, bdry_value))
    ratio = (bdry_clamped - BDRY_MIN) / (BDRY_MAX - BDRY_MIN)
    return RATE_MIN + ratio * (RATE_MAX - RATE_MIN)


def bdry_to_availability(bdry_value):
    bdry_clamped = max(BDRY_MIN, min(BDRY_MAX, bdry_value))
    ratio = (bdry_clamped - BDRY_MIN) / (BDRY_MAX - BDRY_MIN)
    if ratio > 0.66:
        return "LOW", "Market tight — vessels in high demand, book early"
    elif ratio > 0.33:
        return "MODERATE", "Normal vessel supply conditions"
    else:
        return "HIGH", "Vessel oversupply — favorable negotiating position"


def calculate_risk_score(oil_vol, coal_vol, vix):
    oil_risk = min(oil_vol / 10, 1.0) * 100
    coal_risk = min(coal_vol / 15, 1.0) * 100
    vix_risk = min(vix / 35, 1.0) * 100
    composite = (oil_risk * 0.35) + (coal_risk * 0.25) + (vix_risk * 0.40)
    if composite < 33:
        level = "LOW"
    elif composite < 66:
        level = "MEDIUM"
    else:
        level = "HIGH"
    return round(composite, 1), level


def build_full_input(base_values):
    row = dict(base_values)
    row["Coal_MA3"] = base_values["Coal_Price_USD_per_MT"]
    row["Oil_MA3"] = base_values["Oil_Price_USD_per_Barrel"]
    row["IronOre_MA3"] = base_values["IronOre_Price_USD_per_MT"]
    row["NaturalGas_MA3"] = base_values["NaturalGas_Price_USD_per_MMBtu"]
    row["USD_INR_MA3"] = base_values["USD_INR"]
    row["VIX_MA3"] = base_values["VIX_Value"]
    row["WTI_MA3"] = base_values["WTI_Price_USD"]
    row["SBLK_MA3"] = base_values["SBLK_Price_USD"]
    row["Oil_Volatility_3M"] = latest_data["Oil_Volatility_3M"]
    row["Coal_Volatility_3M"] = latest_data["Coal_Volatility_3M"]
    return {f: row[f] for f in feature_list}


def predict_bdry(input_dict):
    input_df = pd.DataFrame([input_dict])[feature_list]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    lower = prediction - model_rmse
    upper = prediction + model_rmse
    return prediction, lower, upper


from statsmodels.tsa.arima.model import ARIMA

def get_forecast_trend():
    hist = pd.read_csv("data/freight_features.csv", parse_dates=["Date"])
    hist = hist.sort_values("Date").reset_index(drop=True)
    series = hist.set_index("Date")["BDRY_Price_USD"]
    series.index = pd.DatetimeIndex(series.index).to_period("M").to_timestamp()
    series = series.asfreq("MS").interpolate()

    # ARIMA(1,1,1) selected after backtesting against Holt-Winters,
    # naive, moving average, and damped trend methods. ARIMA achieved
    # the best directional accuracy (61.7% vs 50% baseline) and lowest
    # error in walk-forward validation - see backtest_compare_methods.py
    arima_model = ARIMA(series, order=(1, 1, 1))
    fitted = arima_model.fit()
    forecast = fitted.forecast(3)

    current_value = series.iloc[-1]
    forecast_avg = forecast.mean()
    pct_change = ((forecast_avg - current_value) / current_value) * 100

    if pct_change > 5:
        trend, timing_advice = "RISING", "Book NOW — freight rates are trending upward, waiting will likely cost more."
    elif pct_change < -5:
        trend, timing_advice = "FALLING", "Consider WAITING — freight rates are trending downward."
    else:
        trend, timing_advice = "STABLE", "Market is stable — booking timing is flexible."

    chart_history = series.tail(12)
    chart_labels = [d.strftime("%b %y") for d in chart_history.index] + [d.strftime("%b %y") for d in forecast.index]
    chart_actual = [round(v, 2) for v in chart_history.values] + [None, None, None]
    chart_forecast = [None] * (len(chart_history) - 1) + [round(chart_history.values[-1], 2)] + [round(v, 2) for v in forecast.values]

    return {
        "current_bdry": round(current_value, 2),
        "forecast_3m_avg": round(forecast_avg, 2),
        "pct_change": round(pct_change, 1),
        "trend": trend,
        "timing_advice": timing_advice,
        "chart_labels": chart_labels,
        "chart_actual": chart_actual,
        "chart_forecast": chart_forecast
    }

def recommend_vessel(cargo_quantity_mt, predicted_bdry, distance_nm):
    daily_rate_panamax_equiv = bdry_to_daily_rate(predicted_bdry)
    availability, availability_note = bdry_to_availability(predicted_bdry)
    distance_multiplier = get_distance_multiplier(distance_nm)
    results = []
    for v in vessels:
        vessel_daily_rate = daily_rate_panamax_equiv * v["daily_rate_multiplier"]
        trips_needed = -(-cargo_quantity_mt // v["capacity_mt"])
        days_per_trip = round((distance_nm / (VESSEL_SPEED_KNOTS * 24)) * 2)
        total_days = days_per_trip * trips_needed
        total_cost = vessel_daily_rate * total_days * distance_multiplier
        cost_per_mt = total_cost / cargo_quantity_mt
        results.append({
            "vessel": v["name"],
            "dwt": v["capacity_mt"],
            "speed": v["speed_knots"],
            "cii_band": v["cii_band"],
            "trips": trips_needed,
            "daily_rate": round(vessel_daily_rate, 2),
            "cost": round(total_cost, 2),
            "cost_per_mt": round(cost_per_mt, 2),
            "days": total_days
        })
    best = min(results, key=lambda r: r["cost"])
    return results, best, availability, availability_note


@app.route("/glossary")
def glossary():
    return render_template("glossary.html")


@app.route("/autofill", methods=["GET"])
def autofill():
    values = {
        "coal_price": round(float(latest_data["Coal_Price_USD_per_MT"]), 2),
        "ironore_price": round(float(latest_data["IronOre_Price_USD_per_MT"]), 2),
        "natgas_price": round(float(latest_data["NaturalGas_Price_USD_per_MMBtu"]), 2),
        "usd_inr": round(float(latest_data["USD_INR"]), 2),
        "vix": round(float(latest_data["VIX_Value"]), 2),
        "wti": round(float(latest_data["WTI_Price_USD"]), 2),
        "sblk": round(float(latest_data["SBLK_Price_USD"]), 2),
    }
    source_note = "Oil = live; others = latest known real data"

    try:
        response = requests.get(
            "https://api.api-ninjas.com/v1/oilprice?type=brent",
            headers={"X-Api-Key": API_NINJAS_KEY},
            timeout=5
        )
        oil_data = response.json()
        values["oil_price"] = round(float(oil_data.get("price", latest_data["Oil_Price_USD_per_Barrel"])), 2)
    except Exception:
        values["oil_price"] = round(float(latest_data["Oil_Price_USD_per_Barrel"]), 2)
        source_note = "Live fetch failed — used latest known values"

    values["source_note"] = source_note
    return jsonify(values)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    forecast_info = get_forecast_trend()

    if request.method == "POST":
        cargo_qty = float(request.form["cargo_qty"])
        origin_port = request.form["origin_port"]
        destination_port = request.form["destination_port"]
        commodity = request.form["commodity"]

        distance_nm = get_route_distance(origin_port, destination_port)
        bcd_pct, igst_pct = get_tariff_info(commodity)

        base_values = {
            "Coal_Price_USD_per_MT": float(request.form["coal_price"]),
            "Oil_Price_USD_per_Barrel": float(request.form["oil_price"]),
            "IronOre_Price_USD_per_MT": float(request.form["ironore_price"]),
            "NaturalGas_Price_USD_per_MMBtu": float(request.form["natgas_price"]),
            "USD_INR": float(request.form["usd_inr"]),
            "VIX_Value": float(request.form["vix"]),
            "WTI_Price_USD": float(request.form["wti"]),
            "SBLK_Price_USD": float(request.form["sblk"]),
        }

        commodity_price = base_values["Coal_Price_USD_per_MT"] if commodity == "Coal" else base_values["IronOre_Price_USD_per_MT"]
        cargo_value_usd = commodity_price * cargo_qty

        input_dict = build_full_input(base_values)
        predicted_bdry, lower_bound, upper_bound = predict_bdry(input_dict)
        comparison, best, availability, availability_note = recommend_vessel(cargo_qty, predicted_bdry, distance_nm)
        risk_score, risk_level = calculate_risk_score(
            latest_data["Oil_Volatility_3M"], latest_data["Coal_Volatility_3M"], base_values["VIX_Value"]
        )
        landed_cost = calculate_landed_cost(best["cost"], cargo_value_usd, bcd_pct, igst_pct)

        result = {
            "predicted_bdry": round(predicted_bdry, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "comparison": comparison,
            "best": best,
            "availability": availability,
            "availability_note": availability_note,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "distance_nm": round(distance_nm, 1),
            "origin_port": origin_port,
            "destination_port": destination_port,
            "commodity": commodity,
            "bcd_pct": bcd_pct,
            "igst_pct": igst_pct,
            "cargo_value": round(cargo_value_usd, 2),
            "landed_cost": landed_cost
        }

    return render_template(
        "index.html",
        result=result,
        forecast=forecast_info,
        origins=sorted(route_data["Origin_Port"].unique()),
        destinations=sorted(route_data["Destination_Port"].unique()),
        chart_labels_json=json.dumps(forecast_info["chart_labels"]),
        chart_actual_json=json.dumps(forecast_info["chart_actual"]),
        chart_forecast_json=json.dumps(forecast_info["chart_forecast"])
    )


if __name__ == "__main__":
    app.run(debug=True)