import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("data/freight_features.csv")

features = [
    "Coal_Price_USD_per_MT", "Oil_Price_USD_per_Barrel", "IronOre_Price_USD_per_MT",
    "NaturalGas_Price_USD_per_MMBtu", "USD_INR", "VIX_Value", "WTI_Price_USD", "SBLK_Price_USD",
    "Coal_MA3", "Oil_MA3", "IronOre_MA3", "NaturalGas_MA3", "USD_INR_MA3",
    "VIX_MA3", "WTI_MA3", "SBLK_MA3",
    "Oil_Volatility_3M", "Coal_Volatility_3M",
    "Month", "Year"
]
target = "BDRY_Price_USD"

X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

def evaluate(name, model, X_test, y_test, X_full, y_full):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring="r2")

    print(f"\n{name}")
    print(f"  Test MAE:  {mae:.3f}")
    print(f"  Test RMSE: {rmse:.3f}")
    print(f"  Test R2:   {r2:.3f}")
    print(f"  5-Fold CV R2: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    return r2

lr = LinearRegression()
lr.fit(X_train, y_train)
r2_lr = evaluate("Linear Regression", lr, X_test, y_test, X_scaled, y)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
r2_ridge = evaluate("Ridge Regression", ridge, X_test, y_test, X_scaled, y)

rf = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=3, random_state=42)
rf.fit(X_train, y_train)
r2_rf = evaluate("Random Forest (regularized)", rf, X_test, y_test, X_scaled, y)

scores = {"Linear Regression": (r2_lr, lr), "Ridge Regression": (r2_ridge, ridge), "Random Forest": (r2_rf, rf)}
best_name = max(scores, key=lambda k: scores[k][0])
best_model = scores[best_name][1]

joblib.dump(best_model, "freight_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "model_features.pkl")
print(f"\nBest model: {best_name} — saved to freight_model.pkl")# Save the actual RMSE of the winning model for confidence interval use later
best_rmse = {
    "Linear Regression": np.sqrt(mean_squared_error(y_test, lr.predict(X_test))),
    "Ridge Regression": np.sqrt(mean_squared_error(y_test, ridge.predict(X_test))),
    "Random Forest": np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))
}[best_name]

joblib.dump(best_model, "freight_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "model_features.pkl")
joblib.dump(best_rmse, "model_rmse.pkl")
print(f"\nBest model: {best_name} — saved to freight_model.pkl")
print(f"Model RMSE (for confidence interval): {best_rmse:.3f}")