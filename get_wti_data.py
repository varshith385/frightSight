import yfinance as yf
import pandas as pd

print("Fetching WTI Crude Oil Futures weekly historical data...")

wti = yf.Ticker("CL=F")
hist = wti.history(start="2010-01-01", end="2025-12-31", interval="1wk")

hist = hist.reset_index()
hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
df = hist[["Date", "Close"]].rename(columns={"Close": "WTI_Price_USD"})

print(df.head())
print("...")
print(df.tail())
print("\nTotal rows:", len(df))

df.to_csv("data/wti.csv", index=False)
print("Saved to data/wti.csv")