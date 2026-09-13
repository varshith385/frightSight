import yfinance as yf
import pandas as pd

print("Fetching Star Bulk Carriers (SBLK) historical data...")

sblk = yf.Ticker("SBLK")
hist = sblk.history(start="2010-01-01", end="2025-12-31", interval="1mo")

hist = hist.reset_index()
hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
df = hist[["Date", "Close"]].rename(columns={"Close": "SBLK_Price_USD"})

print(df.head())
print("...")
print(df.tail())
print("\nTotal rows:", len(df))

df.to_csv("data/sblk.csv", index=False)
print("Saved to data/sblk.csv")