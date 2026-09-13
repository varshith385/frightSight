import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

model = joblib.load("freight_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_list = joblib.load("model_features.pkl")

df = pd.read_csv("data/freight_features.csv")
X = df[feature_list]
X_scaled = scaler.transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_list)

# LinearExplainer is the correct, exact SHAP method for linear/ridge models
# (much faster and more precise than the general-purpose KernelExplainer)
explainer = shap.LinearExplainer(model, X_scaled_df)
shap_values = explainer(X_scaled_df)

# --- Overall feature importance (mean absolute SHAP value per feature) ---
importance = pd.DataFrame({
    "Feature": feature_list,
    "Mean_Abs_SHAP": np.abs(shap_values.values).mean(axis=0)
}).sort_values("Mean_Abs_SHAP", ascending=False)

print("Feature Importance (ranked by average impact on prediction):")
print(importance.to_string(index=False))

importance.to_csv("shap_feature_importance.csv", index=False)
print("\nSaved to shap_feature_importance.csv")

# --- Save a summary plot for your PPT ---
plt.figure(figsize=(9, 7))
shap.summary_plot(shap_values, X_scaled_df, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
print("Saved chart to shap_summary.png")

# --- Explain ONE specific prediction (the most recent real data point) ---
latest_idx = len(X_scaled_df) - 1
print(f"\n--- Explanation for most recent data point (row {latest_idx}) ---")
print(f"Base value (average prediction): {explainer.expected_value:.2f}")
print(f"Actual prediction for this point: {model.predict(X_scaled_df.iloc[[latest_idx]])[0]:.2f}")
print("\nTop 5 contributing features for this specific prediction:")
single_shap = pd.DataFrame({
    "Feature": feature_list,
    "SHAP_Value": shap_values.values[latest_idx]
}).sort_values("SHAP_Value", key=abs, ascending=False).head(5)
print(single_shap.to_string(index=False))

plt.figure(figsize=(10, 4))
shap.plots.waterfall(shap_values[latest_idx], show=False)
plt.tight_layout()
plt.savefig("shap_waterfall_example.png", dpi=150, bbox_inches="tight")
print("\nSaved chart to shap_waterfall_example.png")