import yfinance as yf
import pandas as pd

print("Fetching SEA (Guggenheim Shipping ETF) historical data...")

sea = yf.Ticker("SEA")
hist = sea.history(start="2010-01-01", end="2025-12-31", interval="1mo")

hist = hist.reset_index()
hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
df = hist[["Date", "Close"]].rename(columns={"Close": "SEA_Price_USD"})

print(df.head())
print("...")
print(df.tail())
print("\nTotal rows:", len(df))

df.to_csv("data/sea.csv", index=False)
print("Saved to data/sea.csv")