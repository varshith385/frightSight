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
df = df.reset_index(drop=True)

# --- Merge in USD/INR exchange rate (real data) ---
fx = pd.read_csv("data/usdinr.csv", parse_dates=["Date"])
fx["YearMonth"] = fx["Date"].dt.to_period("M")
fx_monthly = fx.groupby("YearMonth")["USD_INR"].mean().reset_index()
fx_monthly["Date"] = fx_monthly["YearMonth"].dt.to_timestamp()
fx_monthly = fx_monthly[["Date", "USD_INR"]]

df["YearMonth"] = df["Date"].dt.to_period("M")
fx_monthly["YearMonth"] = fx_monthly["Date"].dt.to_period("M")

df = df.merge(fx_monthly[["YearMonth", "USD_INR"]], on="YearMonth", how="left")
df = df.drop(columns=["YearMonth"])

# --- Merge in BDRY (real dry bulk freight market proxy) ---
bdry = pd.read_csv("data/bdry.csv", parse_dates=["Date"])
bdry["YearMonth"] = bdry["Date"].dt.to_period("M")
bdry = bdry[["YearMonth", "BDRY_Price_USD"]]

df["YearMonth"] = df["Date"].dt.to_period("M")
df = df.merge(bdry, on="YearMonth", how="left")
df = df.drop(columns=["YearMonth"])

# --- Merge in VIX, WTI, SBLK (real market indicators) ---
for fname, col in [("vix.csv", "VIX_Value"), ("wti.csv", "WTI_Price_USD"), ("sblk.csv", "SBLK_Price_USD")]:
    extra = pd.read_csv(f"data/{fname}", parse_dates=["Date"])
    extra["YearMonth"] = extra["Date"].dt.to_period("M")
    extra = extra[["YearMonth", col]]

    df["YearMonth"] = df["Date"].dt.to_period("M")
    df = df.merge(extra, on="YearMonth", how="left")
    df = df.drop(columns=["YearMonth"])

print(df.head(5))
print("...")
print(df.tail(5))
print("\nTotal usable rows:", len(df))
print("Date range:", df["Date"].min(), "to", df["Date"].max())

df.to_csv("data/freight_base_data.csv", index=False)
print("\nSaved cleaned data to data/freight_base_data.csv")