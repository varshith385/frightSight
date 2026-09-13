import requests
import pandas as pd

print("Fetching USD/INR historical data...")

response = requests.get(
    "https://api.frankfurter.app/1999-01-01..2025-12-31",
    params={"from": "USD", "to": "INR"}
)

data = response.json()
rates = data["rates"]  # dict of {date: {"INR": value}}

df = pd.DataFrame([
    {"Date": date, "USD_INR": values["INR"]}
    for date, values in rates.items()
])

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print(df.head())
print("...")
print(df.tail())
print("\nTotal rows:", len(df))

df.to_csv("data/usdinr.csv", index=False)
print("Saved to data/usdinr.csv")