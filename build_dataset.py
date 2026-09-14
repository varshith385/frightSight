import pandas as pd

df = pd.read_excel(
    "data/commodity_prices.xlsx",
    sheet_name="Monthly Prices",
    header=4
)

df = df[["Unnamed: 0", "Coal, Australian", "Crude oil, Brent", "Iron ore, cfr spot", "Natural gas, US"]]
df.columns = ["Date", "Coal_Price_USD_per_MT", "Oil_Price_USD_per_Barrel", "IronOre_Price_USD_per_MT", "NaturalGas_Price_USD_per_MMBtu"]

df["Coal_Price_USD_per_MT"] = pd.to_numeric(df["Coal_Price_USD_per_MT"], errors="coerce")
df["Oil_Price_USD_per_Barrel"] = pd.to_numeric(df["Oil_Price_USD_per_Barrel"], errors="coerce")
df["IronOre_Price_USD_per_MT"] = pd.to_numeric(df["IronOre_Price_USD_per_MT"], errors="coerce")
df["NaturalGas_Price_USD_per_MMBtu"] = pd.to_numeric(df["NaturalGas_Price_USD_per_MMBtu"], errors="coerce")

df = df.dropna()
df["Date"] = pd.to_datetime(df["Date"], format="%YM%m")
df = df.set_index("Date").sort_index()

# --- Resample monthly commodity data to weekly using forward-fill ---
# This is a standard, legitimate technique for mixed-frequency data:
# we carry forward the last REAL published monthly value until the next
# real value is published - not inventing new numbers, just correctly
# representing "this was the latest known price during this week."
weekly_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="W-MON")
df_weekly = df.reindex(df.index.union(weekly_index)).sort_index().ffill()
df_weekly = df_weekly.reindex(weekly_index)
df_weekly = df_weekly.reset_index().rename(columns={"index": "Date"})

# --- Merge in USD/INR (resample real daily data to weekly average) ---
fx = pd.read_csv("data/usdinr.csv", parse_dates=["Date"])
fx = fx.set_index("Date").sort_index()
fx_weekly = fx["USD_INR"].resample("W-MON").mean().reset_index()

df_weekly["Date"] = pd.to_datetime(df_weekly["Date"])
fx_weekly["Date"] = pd.to_datetime(fx_weekly["Date"])
df_weekly = pd.merge_asof(df_weekly.sort_values("Date"), fx_weekly.sort_values("Date"), on="Date", direction="nearest")

# --- Merge in BDRY, SEA, VIX, WTI, SBLK (already real weekly data) ---
for fname, col in [("bdry.csv", "BDRY_Price_USD"), ("sea.csv", "SEA_Price_USD"),
                     ("vix.csv", "VIX_Value"), ("wti.csv", "WTI_Price_USD"),
                     ("sblk.csv", "SBLK_Price_USD")]:
    extra = pd.read_csv(f"data/{fname}", parse_dates=["Date"])
    extra = extra.sort_values("Date")
    df_weekly = pd.merge_asof(df_weekly.sort_values("Date"), extra, on="Date", direction="nearest", tolerance=pd.Timedelta("4D"))

print(df_weekly.head(5))
print("...")
print(df_weekly.tail(5))
print("\nTotal usable rows:", len(df_weekly))
print("Date range:", df_weekly["Date"].min(), "to", df_weekly["Date"].max())

df_weekly.to_csv("data/freight_base_data.csv", index=False)
print("\nSaved weekly dataset to data/freight_base_data.csv")