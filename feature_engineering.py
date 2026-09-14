import pandas as pd
import numpy as np

df = pd.read_csv("data/freight_base_data.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Keep only rows where we have real BDRY data (2018 onward)
df = df.dropna(subset=["BDRY_Price_USD"]).reset_index(drop=True)

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# --- Trend features: use a 4-week window (~1 month equivalent at weekly resolution) ---
df["Coal_MA3"] = df["Coal_Price_USD_per_MT"].rolling(window=4).mean()
df["Oil_MA3"] = df["Oil_Price_USD_per_Barrel"].rolling(window=4).mean()
df["IronOre_MA3"] = df["IronOre_Price_USD_per_MT"].rolling(window=4).mean()
df["NaturalGas_MA3"] = df["NaturalGas_Price_USD_per_MMBtu"].rolling(window=4).mean()
df["USD_INR_MA3"] = df["USD_INR"].rolling(window=4).mean()
df["VIX_MA3"] = df["VIX_Value"].rolling(window=4).mean()
df["WTI_MA3"] = df["WTI_Price_USD"].rolling(window=4).mean()
df["SBLK_MA3"] = df["SBLK_Price_USD"].rolling(window=4).mean()

df["Oil_Volatility_3M"] = df["Oil_Price_USD_per_Barrel"].rolling(window=4).std()
df["Coal_Volatility_3M"] = df["Coal_Price_USD_per_MT"].rolling(window=4).std()

df = df.dropna(subset=[
    "Coal_MA3", "Oil_MA3", "IronOre_MA3", "NaturalGas_MA3", "USD_INR_MA3",
    "VIX_MA3", "WTI_MA3", "SBLK_MA3", "Oil_Volatility_3M", "Coal_Volatility_3M",
    "BDRY_Price_USD"
]).reset_index(drop=True)

print(df[["Date", "Coal_Price_USD_per_MT", "Oil_Price_USD_per_Barrel",
          "VIX_Value", "SBLK_Price_USD", "BDRY_Price_USD"]].head(10))
print("\nTotal rows after feature engineering:", len(df))
print("Date range:", df["Date"].min(), "to", df["Date"].max())

df.to_csv("data/freight_features.csv", index=False)
print("Saved to data/freight_features.csv")