import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==================================================
# 1. LOAD FEATURE-ENGINEERED DATA
# ==================================================

file_path = "data/processed/ecommerce_churn_cleaned.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully.")

# ==================================================
# STANDARDIZE CATEGORICAL VALUES
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

print("Categorical values standardized successfully.")

# ==================================================
# DEFINE CATEGORICAL FEATURES
# ==================================================

categorical_columns = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]

# ==================================================
# CREATE PREPROCESSOR
# ==================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
            ),
            categorical_columns,
        )
    ],
    remainder="passthrough",
)

# ==================================================
# 2. SEPARATE FEATURES AND TARGET
# ==================================================

X = df.drop(columns=["Churn"])
y = df["Churn"]

print("\nFeature matrix shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# ==================================================
# 3. TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining set:")
print(X_train.shape)

print("\nTest set:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True) * 100)


# ==================================================
# 4. LOGISTIC REGRESSION BASELINE
# ==================================================

# Logistic Regression benefits from feature scaling.
# The pipeline ensures scaling is learned only from
# the training data.

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "scaler",
            StandardScaler(
                with_mean=False
            ),
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)

logistic_model.fit(X_train, y_train)

print("\nLogistic Regression training completed.")


# ==================================================
# 5. EVALUATE LOGISTIC REGRESSION
# ==================================================

logistic_pred = logistic_model.predict(X_test)
logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

logistic_accuracy = accuracy_score(y_test, logistic_pred)
logistic_precision = precision_score(y_test, logistic_pred)
logistic_recall = recall_score(y_test, logistic_pred)
logistic_f1 = f1_score(y_test, logistic_pred)
logistic_roc_auc = roc_auc_score(y_test, logistic_prob)

print("\nLogistic Regression Results:")
print(f"Accuracy:  {logistic_accuracy:.4f}")
print(f"Precision: {logistic_precision:.4f}")
print(f"Recall:    {logistic_recall:.4f}")
print(f"F1 Score:  {logistic_f1:.4f}")
print(f"ROC-AUC:   {logistic_roc_auc:.4f}")

print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, logistic_pred))


# ==================================================
# 6. RANDOM FOREST
# ==================================================

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
        ),
    ]
)

random_forest_model.fit(X_train, y_train)

rf_pred = random_forest_model.predict(X_test)
rf_prob = random_forest_model.predict_proba(X_test)[:, 1]


# ==================================================
# 7. EVALUATE RANDOM FOREST
# ==================================================

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_roc_auc = roc_auc_score(y_test, rf_prob)

print("\nRandom Forest Results:")
print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print(f"F1 Score:  {rf_f1:.4f}")
print(f"ROC-AUC:   {rf_roc_auc:.4f}")

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))


# ==================================================
# 8. OVERFITTING CHECK
# ==================================================

rf_train_pred = random_forest_model.predict(X_train)

rf_train_accuracy = accuracy_score(y_train, rf_train_pred)
rf_test_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Overfitting Check:")
print(f"Training Accuracy: {rf_train_accuracy:.4f}")
print(f"Test Accuracy:     {rf_test_accuracy:.4f}")
print(
    f"Difference:        "
    f"{rf_train_accuracy - rf_test_accuracy:.4f}"
)

# ==================================================
# 9. RANDOM FOREST FEATURE IMPORTANCE
# ==================================================

# Get the fitted preprocessor from the pipeline
fitted_preprocessor = random_forest_model.named_steps["preprocessor"]

# Get the feature names after preprocessing
feature_names = fitted_preprocessor.get_feature_names_out()

# Get the fitted Random Forest model from the pipeline
fitted_rf_model = random_forest_model.named_steps["model"]

# Match each transformed feature with its importance
feature_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": fitted_rf_model.feature_importances_,
    }
)

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False,
)

# Clean pipeline prefixes from feature names
feature_importance["Feature"] = (
    feature_importance["Feature"]
    .str.replace("remainder__", "", regex=False)
    .str.replace("categorical__", "", regex=False)
)

print("\nTop 15 Random Forest Feature Importances:")
print(feature_importance.head(15).to_string(index=False))

# ==================================================
# SAVE FEATURE IMPORTANCE VISUALIZATION
# ==================================================

# Select the 15 most important features
top_features = feature_importance.head(15).sort_values(
    by="Importance",
    ascending=True,
)

# Create the outputs directory if it does not exist
os.makedirs("outputs", exist_ok=True)

# Create the chart
plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"],
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features Influencing Customer Churn")

plt.tight_layout()

# Save the chart
feature_importance_path = "outputs/feature_importance.png"

plt.savefig(
    feature_importance_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"\nFeature importance chart saved to: "
    f"{feature_importance_path}"
)

# ==================================================
# 10. RANDOM FOREST CROSS-VALIDATION
# ==================================================

# Cross-validation uses only the training data so that
# the test set remains untouched during model development.

cv_scores = cross_val_score(
    random_forest_model,
    X_train,
    y_train,
    cv=5,
    scoring="roc_auc",
)

print("\nRandom Forest 5-Fold Cross-Validation:")
print("ROC-AUC scores:", cv_scores)
print(f"Mean ROC-AUC: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")


# ==================================================
# 11. RANDOM FOREST HYPERPARAMETER TUNING
# ==================================================

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
}

print("\nRandom Forest hyperparameter grid:")
print(param_grid)

grid_search = GridSearchCV(
    estimator=Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                ),
            ),
        ]
    ),
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)

print("\nRunning GridSearchCV...")

# Only the training data is used for hyperparameter selection.
grid_search.fit(X_train, y_train)

print("\nGridSearchCV completed.")

print("\nBest hyperparameters:")
print(grid_search.best_params_)

print("\nBest cross-validation ROC-AUC:")
print(f"{grid_search.best_score_:.4f}")


# ==================================================
# 12. EVALUATE THE BEST RANDOM FOREST
# ==================================================

best_rf_model = grid_search.best_estimator_

best_rf_pred = best_rf_model.predict(X_test)
best_rf_prob = best_rf_model.predict_proba(X_test)[:, 1]

best_rf_accuracy = accuracy_score(y_test, best_rf_pred)
best_rf_precision = precision_score(y_test, best_rf_pred)
best_rf_recall = recall_score(y_test, best_rf_pred)
best_rf_f1 = f1_score(y_test, best_rf_pred)
best_rf_roc_auc = roc_auc_score(y_test, best_rf_prob)

print("\nTuned Random Forest Test Results:")
print(f"Accuracy:  {best_rf_accuracy:.4f}")
print(f"Precision: {best_rf_precision:.4f}")
print(f"Recall:    {best_rf_recall:.4f}")
print(f"F1 Score:  {best_rf_f1:.4f}")
print(f"ROC-AUC:   {best_rf_roc_auc:.4f}")

print("\nTuned Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, best_rf_pred))


# ==================================================
# 13. SAVE FINAL TRAINED MODEL
# ==================================================

os.makedirs("models", exist_ok=True)

model_path = "models/random_forest_churn_model.pkl"

joblib.dump(best_rf_model, model_path)

print(
    f"\nFinal Random Forest model saved to: "
    f"{model_path}"
)