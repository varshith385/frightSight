
    bdry_clamped = max(BDRY_MIN, min(BDRY_MAX, bdry_value))
    ratio = (bdry_clamped - BDRY_MIN) / (BDRY_MAX - BDRY_MIN)
    return RATE_MIN + ratio * (RATE_MAX - RATE_MIN)


def predict_bdry(coal_price, oil_price, ironore_price, natgas_price, usd_inr,
                  coal_ma3, oil_ma3, ironore_ma3, natgas_ma3, usd_inr_ma3,
                  oil_vol, coal_vol, month, year):
    input_df = pd.DataFrame([{
        "Coal_Price_USD_per_MT": coal_price,
        "Oil_Price_USD_per_Barrel": oil_price,
        "IronOre_Price_USD_per_MT": ironore_price,
        "NaturalGas_Price_USD_per_MMBtu": natgas_price,
        "USD_INR": usd_inr,
        "Coal_MA3": coal_ma3,
        "Oil_MA3": oil_ma3,
        "IronOre_MA3": ironore_ma3,
        "NaturalGas_MA3": natgas_ma3,
        "USD_INR_MA3": usd_inr_ma3,
        "Oil_Volatility_3M": oil_vol,
        "Coal_Volatility_3M": coal_vol,
        "Month": month,
        "Year": year
    }])
    input_scaled = scaler.transform(input_df)
    return model.predict(input_scaled)[0]


def get_forecast_trend():
    hist = pd.read_csv("data/freight_features.csv", parse_dates=["Date"])
    hist = hist.sort_values("Date").reset_index(drop=True)
    series = hist.set_index("Date")["BDRY_Price_USD"]

    hw_model = ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated")
    fitted = hw_model.fit()
    forecast = fitted.forecast(3)

    current_value = series.iloc[-1]
    forecast_avg = forecast.mean()
    pct_change = ((forecast_avg - current_value) / current_value) * 100

    if pct_change > 5:
        trend = "RISING"
        timing_advice = "Book NOW — freight rates are trending upward, waiting will likely cost more."
    elif pct_change < -5:
        trend = "FALLING"
        timing_advice = "Consider WAITING — freight rates are trending downward."
    else:
        trend = "STABLE"
        timing_advice = "Market is stable — booking timing is flexible."

    return {
        "current_bdry": round(current_value, 2),