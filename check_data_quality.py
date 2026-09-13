import pandas as pd

df = pd.read_csv("data/sea.csv", parse_dates=["Date"])

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nData types:\n", df.dtypes)
print("\nMin/Max SEA price:", df["SEA_Price_USD"].min(), "/", df["SEA_Price_USD"].max())
print("\nDate range:", df["Date"].min(), "to", df["Date"].max())

date_diffs = df["Date"].diff().dt.days
print("\nUnusual gaps (not ~28-31 days):")
print(date_diffs[(date_diffs > 35) | (date_diffs < 25)])