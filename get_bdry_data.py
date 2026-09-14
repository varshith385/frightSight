import yfinance as yf
import pandas as pd

print("Fetching BDRY (Breakwave Dry Bulk Shipping ETF) weekly historical data...")

bdry = yf.Ticker("BDRY")
hist = bdry.history(start="2018-01-01", end="2025-12-31", interval="1wk")

hist = hist.reset_index()
hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
df = hist[["Date", "Close"]].rename(columns={"Close": "BDRY_Price_USD"})

print(df.head())
print("...")
print(df.tail())
print("\nTotal rows:", len(df))

df.to_csv("data/bdry.csv", index=False)
print("Saved to data/bdry.csv")