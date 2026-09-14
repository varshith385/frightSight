import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score
import joblib

# Rebuild a SEA-based dataset the same way BDRY was built
df = pd.read_csv("data/freight_base_data.csv", parse_dates=["Date"])

sea = pd.read_csv("data/sea.csv", parse_dates=["Date"])
sea["YearMonth"] = sea["Date"].dt.to_period("M")
sea = sea[["YearMonth", "SEA_Price_USD"]]

df["YearMonth"] = df["Date"].dt.to_period("M")
df = df.merge(sea, on="YearMonth", how="left")
df = df.drop(columns=["YearMonth"])

df = df.dropna(subset=["SEA_Price_USD"]).reset_index(drop=True)

df["Coal_MA3"] = df["Coal_Price_USD_per_MT"].rolling(3).mean()
df["Oil_MA3"] = df["Oil_Price_USD_per_Barrel"].rolling(3).mean()
df["IronOre_MA3"] = df["IronOre_Price_USD_per_MT"].rolling(3).mean()
df["NaturalGas_MA3"] = df["NaturalGas_Price_USD_per_MMBtu"].rolling(3).mean()
df["USD_INR_MA3"] = df["USD_INR"].rolling(3).mean()
df["VIX_MA3"] = df["VIX_Value"].rolling(3).mean()
df["WTI_MA3"] = df["WTI_Price_USD"].rolling(3).mean()
df["SBLK_MA3"] = df["SBLK_Price_USD"].rolling(3).mean()
df["Oil_Volatility_3M"] = df["Oil_Price_USD_per_Barrel"].rolling(3).std()
df["Coal_Volatility_3M"] = df["Coal_Price_USD_per_MT"].rolling(3).std()

df = df.dropna().reset_index(drop=True)
print(f"SEA dataset: {len(df)} rows")

features = [
    "Coal_Price_USD_per_MT", "Oil_Price_USD_per_Barrel", "IronOre_Price_USD_per_MT",
    "NaturalGas_Price_USD_per_MMBtu", "USD_INR", "VIX_Value", "WTI_Price_USD", "SBLK_Price_USD",
    "Coal_MA3", "Oil_MA3", "IronOre_MA3", "NaturalGas_MA3", "USD_INR_MA3",
    "VIX_MA3", "WTI_MA3", "SBLK_MA3", "Oil_Volatility_3M", "Coal_Volatility_3M"
]

X = df[features]
y_sea = df["SEA_Price_USD"]

scaler_sea = StandardScaler()
X_scaled = scaler_sea.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_sea, test_size=0.2, random_state=42)

sea_model = Ridge(alpha=1.0)
sea_model.fit(X_train, y_train)
pred = sea_model.predict(X_test)
r2 = r2_score(y_test, pred)
cv_scores = cross_val_score(sea_model, X_scaled, y_sea, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")

print(f"\nSEA Model Performance:")
print(f"  Test R2: {r2:.3f}")
print(f"  5-Fold CV R2: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# --- Directional agreement check ---
# For each test point, does the SEA model and BDRY model agree on
# whether conditions represent "high" or "low" market pressure?
bdry_model = joblib.load("freight_model.pkl")
bdry_scaler = joblib.load("scaler.pkl")
bdry_features = joblib.load("model_features.pkl")

X_bdry = df[bdry_features]
X_bdry_scaled = bdry_scaler.transform(X_bdry)
bdry_predictions = bdry_model.predict(X_bdry_scaled)
sea_predictions = sea_model.predict(X_scaled)

# Normalize both to 0-1 scale for comparison, then check correlation
bdry_norm = (bdry_predictions - bdry_predictions.min()) / (bdry_predictions.max() - bdry_predictions.min())
sea_norm = (sea_predictions - sea_predictions.min()) / (sea_predictions.max() - sea_predictions.min())

correlation = np.corrcoef(bdry_norm, sea_norm)[0, 1]

print(f"\n--- Triangulation: BDRY-model vs SEA-model predictions ---")
print(f"Correlation between the two independent models' predictions: {correlation:.3f}")
if correlation > 0.5:
    print("✓ Strong positive agreement - suggests genuine shared market signal, not coincidence.")
elif correlation > 0.2:
    print("~ Moderate agreement - some shared signal, but not fully consistent.")
else:
    print("⚠ Weak/no agreement - predictions may be capturing noise rather than real market structure.")

joblib.dump(sea_model, "sea_model.pkl")
joblib.dump(scaler_sea, "sea_scaler.pkl")
print("\nSaved SEA model for reference.")