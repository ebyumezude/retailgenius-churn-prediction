import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# ==================================================
# 1. CONFIGURATION
# ==================================================

DATA_PATH = "data/processed/ecommerce_churn_cleaned.csv"
MODEL_PATH = "models/random_forest_churn_model.pkl"
OUTPUT_DIR = "outputs/shap"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================================================
# 2. LOAD DATA AND TRAINED PIPELINE
# ==================================================

df = pd.read_csv(DATA_PATH)

model_pipeline = joblib.load(MODEL_PATH)

print("Dataset and trained model loaded successfully.")


# ==================================================
# 3. STANDARDIZE CATEGORICAL VALUES
# ==================================================

df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace(
    {
        "Phone": "Mobile Phone",
    }
)

df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace(
    {
        "CC": "Credit Card",
        "COD": "Cash on Delivery",
    }
)


# ==================================================
# 4. SEPARATE FEATURES AND TARGET
# ==================================================

X = df.drop(columns=["Churn"])
y = df["Churn"]

print(f"Dataset shape: {X.shape}")


# ==================================================
# 5. EXTRACT PIPELINE COMPONENTS
# ==================================================

preprocessor = model_pipeline.named_steps["preprocessor"]
random_forest = model_pipeline.named_steps["model"]

X_transformed = preprocessor.transform(X)

feature_names = preprocessor.get_feature_names_out()

# Convert to a dense array if necessary
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

# Clean feature names
clean_feature_names = [
    name.replace("categorical__", "").replace("remainder__", "")
    for name in feature_names
]

X_transformed_df = pd.DataFrame(
    X_transformed,
    columns=clean_feature_names,
)

print(f"Transformed dataset shape: {X_transformed_df.shape}")


# ==================================================
# 6. CREATE SHAP TREE EXPLAINER
# ==================================================

explainer = shap.TreeExplainer(random_forest)

shap_values = explainer(X_transformed_df)

print("SHAP values calculated successfully.")


# ==================================================
# 7. HANDLE BINARY CLASSIFICATION OUTPUT
# ==================================================

# Depending on the SHAP version, Random Forest explanations
# may contain a separate SHAP value for each class.
if shap_values.values.ndim == 3:
    churn_values = shap_values.values[:, :, 1]
    churn_base_values = shap_values.base_values[:, 1]
else:
    churn_values = shap_values.values
    churn_base_values = shap_values.base_values

churn_explanation = shap.Explanation(
    values=churn_values,
    base_values=churn_base_values,
    data=X_transformed_df.values,
    feature_names=clean_feature_names,
)


# ==================================================
# 8. BEESWARM PLOT - WHOLE DATASET
# ==================================================

plt.figure()
shap.plots.beeswarm(
    churn_explanation,
    max_display=15,
    show=False,
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/beeswarm_plot.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Beeswarm plot saved.")


# ==================================================
# 9. BAR / MEAN SHAP PLOT
# ==================================================

plt.figure()

shap.plots.bar(
    churn_explanation,
    max_display=15,
    show=False,
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/mean_shap_plot.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Mean SHAP plot saved.")


# ==================================================
# 10. WATERFALL PLOT - ONE CUSTOMER
# ==================================================

customer_index = 0

plt.figure()

shap.plots.waterfall(
    churn_explanation[customer_index],
    max_display=15,
    show=False,
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/waterfall_plot.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Waterfall plot saved.")


# ==================================================
# 11. FORCE PLOT - ONE CUSTOMER
# ==================================================

force_plot = shap.force_plot(
    churn_explanation[customer_index].base_values,
    churn_explanation[customer_index].values,
    X_transformed_df.iloc[customer_index],
    feature_names=clean_feature_names,
)

shap.save_html(
    f"{OUTPUT_DIR}/force_plot.html",
    force_plot,
)

print("Force plot saved.")


# ==================================================
# 12. DEPENDENCE PLOT
# ==================================================

# Use the most important Random Forest feature automatically.
most_important_index = np.argmax(random_forest.feature_importances_)

most_important_feature = clean_feature_names[most_important_index]

plt.figure()

shap.dependence_plot(
    most_important_feature,
    churn_values,
    X_transformed_df,
    show=False,
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/dependence_plot.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Dependence plot saved for: " f"{most_important_feature}")


# ==================================================
# 13. CLASS-SPECIFIC SUMMARY PLOTS
# ==================================================

for class_value, class_name in [
    (0, "non_churn"),
    (1, "churn"),
]:
    class_mask = y.to_numpy() == class_value

    class_explanation = shap.Explanation(
        values=churn_values[class_mask],
        base_values=churn_base_values[class_mask],
        data=X_transformed_df.loc[class_mask].values,
        feature_names=clean_feature_names,
    )

    plt.figure()

    shap.plots.beeswarm(
        class_explanation,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/summary_{class_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Summary plot saved for class: " f"{class_name}")


# ==================================================
# 14. COMPLETE
# ==================================================

print("\nSHAP explainability analysis completed.")
print(f"Outputs saved in: {OUTPUT_DIR}")
