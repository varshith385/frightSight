import yfinance as yf
import pandas as pd

print("Fetching VIX (Volatility Index) weekly historical data...")

vix = yf.Ticker("^VIX")
hist = vix.history(start="2010-01-01", end="2025-12-31", interval="1wk")

hist = hist.reset_index()
hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
df = hist[["Date", "Close"]].rename(columns={"Close": "VIX_Value"})

print(df.head())
print("...")
print(df.tail())
print("\nTotal rows:", len(df))

df.to_csv("data/vix.csv", index=False)
print("Saved to data/vix.csv")