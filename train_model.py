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
    "Oil_Volatility_3M", "Coal_Volatility_3M"
]
# Note: Year and Month deliberately excluded. Including raw calendar time
# let the model partly fit a trend line rather than genuine market
# relationships (confirmed via SHAP - Year was the single most
# influential feature). Removing them forces the model to rely only on
# real market indicators, which is more defensible and better reflects
# the actual goal: predicting freight market movement from economic
# conditions, not from the passage of time itself.
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

from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning: search over a range of alpha values instead of
# guessing. Alpha controls regularization strength - higher alpha shrinks
# feature weights more aggressively (helps prevent overfitting on small data).
ridge_param_grid = {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]}
ridge_grid = GridSearchCV(
    Ridge(), ridge_param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="r2"
)
ridge_grid.fit(X_train, y_train)
ridge = ridge_grid.best_estimator_
print(f"\nBest Ridge alpha found: {ridge_grid.best_params_['alpha']}")
r2_ridge = evaluate("Ridge Regression (tuned)", ridge, X_test, y_test, X_scaled, y)


rf_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 4, 5, 6],
    "min_samples_leaf": [2, 3, 5]
}
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42), rf_param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="r2", n_jobs=-1
)
rf_grid.fit(X_train, y_train)
rf = rf_grid.best_estimator_
print(f"\nBest Random Forest params found: {rf_grid.best_params_}")
r2_rf = evaluate("Random Forest (tuned)", rf, X_test, y_test, X_scaled, y)


# Select based on cross-validation mean, not single test-split R2 -
# CV is a far more reliable signal of true generalization performance,
# especially with a small dataset where one split can be misleading.
cv_scores = {
    "Linear Regression": cross_val_score(lr, X_scaled, y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2").mean(),
    "Ridge Regression": cross_val_score(ridge, X_scaled, y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2").mean(),
    "Random Forest": cross_val_score(rf, X_scaled, y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2").mean(),
}
models = {"Linear Regression": lr, "Ridge Regression": ridge, "Random Forest": rf}

best_name = max(cv_scores, key=cv_scores.get)
best_model = models[best_name]

print(f"\nModel selection based on 5-Fold CV R2 (more reliable than single test split):")
for name, score in cv_scores.items():
    marker = " <- selected" if name == best_name else ""
    print(f"  {name}: {score:.3f}{marker}")

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