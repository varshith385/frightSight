import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold

model = joblib.load("freight_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_list = joblib.load("model_features.pkl")
model_rmse = joblib.load("model_rmse.pkl")

df = pd.read_csv("data/freight_features.csv")
X = df[feature_list]
y = df["BDRY_Price_USD"].values
X_scaled = scaler.transform(X)

# Use cross-validation predictions (out-of-sample) to fairly test calibration -
# testing on training data would be misleadingly optimistic.
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_predictions = np.zeros(len(y))

for train_idx, test_idx in kf.split(X_scaled):
    model.fit(X_scaled[train_idx], y[train_idx])
    all_predictions[test_idx] = model.predict(X_scaled[test_idx])

# Refit on full data to restore the saved model's actual state
model.fit(X_scaled, y)
joblib.dump(model, "freight_model.pkl")

lower_bound = all_predictions - model_rmse
upper_bound = all_predictions + model_rmse

within_interval = (y >= lower_bound) & (y <= upper_bound)
coverage = within_interval.mean() * 100

print("=" * 60)
print("CONFIDENCE INTERVAL CALIBRATION CHECK")
print("=" * 60)
print(f"\nRMSE used for interval: ±{model_rmse:.3f}")
print(f"Expected coverage (if well-calibrated, ±1 RMSE): ~68%")
print(f"Actual coverage observed: {coverage:.1f}%")

if coverage < 55:
    print("\n⚠ Interval is TOO NARROW - real outcomes fall outside it more than expected.")
    print("  Recommendation: widen the interval (e.g., use ±1.5x RMSE)")
elif coverage > 80:
    print("\n⚠ Interval is WIDER than necessary - could be tightened for more useful precision.")
else:
    print("\n✓ Interval is reasonably well-calibrated.")

# Test a wider interval too, for comparison
wider_lower = all_predictions - (1.5 * model_rmse)
wider_upper = all_predictions + (1.5 * model_rmse)
wider_coverage = ((y >= wider_lower) & (y <= wider_upper)).mean() * 100
print(f"\nFor reference, ±1.5x RMSE would give: {wider_coverage:.1f}% coverage")