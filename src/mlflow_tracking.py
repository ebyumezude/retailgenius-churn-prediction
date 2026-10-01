import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==================================================
# 1. CONFIGURATION
# ==================================================

DATA_PATH = "data/processed/ecommerce_churn_cleaned.csv"
EXPERIMENT_NAME = "RetailGenius Customer Churn"

mlflow.set_experiment(EXPERIMENT_NAME)


# ==================================================
# 2. LOAD DATA
# ==================================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")


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
# 4. FEATURES AND TARGET
# ==================================================

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


categorical_columns = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]


def create_preprocessor():
    """Create preprocessing for categorical features."""

    return ColumnTransformer(
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
# 5. EVALUATION FUNCTION
# ==================================================


def evaluate_model(model):
    """Calculate classification metrics."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
        ),
        "recall": recall_score(
            y_test,
            predictions,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }


# ==================================================
# 6. LOGISTIC REGRESSION RUN
# ==================================================

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor(),
        ),
        (
            "scaler",
            StandardScaler(
                with_mean=False,
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


with mlflow.start_run(run_name="Logistic Regression Baseline"):

    logistic_model.fit(
        X_train,
        y_train,
    )

    logistic_metrics = evaluate_model(logistic_model)

    mlflow.log_param(
        "model_type",
        "LogisticRegression",
    )

    mlflow.log_param(
        "max_iter",
        1000,
    )

    mlflow.log_param(
        "test_size",
        0.20,
    )

    mlflow.log_param(
        "random_state",
        42,
    )

    mlflow.log_metrics(logistic_metrics)

    mlflow.sklearn.log_model(
        logistic_model,
        name="model",
    )

    print("\nLogistic Regression " "MLflow run completed.")

    print(logistic_metrics)


# ==================================================
# 7. RANDOM FOREST RUN
# ==================================================

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor(),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
            ),
        ),
    ]
)


with mlflow.start_run(run_name="Tuned Random Forest"):

    random_forest_model.fit(
        X_train,
        y_train,
    )

    rf_metrics = evaluate_model(random_forest_model)

    mlflow.log_param(
        "model_type",
        "RandomForestClassifier",
    )

    mlflow.log_param(
        "n_estimators",
        200,
    )

    mlflow.log_param(
        "max_depth",
        20,
    )

    mlflow.log_param(
        "min_samples_split",
        2,
    )

    mlflow.log_param(
        "min_samples_leaf",
        1,
    )

    mlflow.log_param(
        "test_size",
        0.20,
    )

    mlflow.log_param(
        "random_state",
        42,
    )

    mlflow.log_metrics(rf_metrics)

    mlflow.sklearn.log_model(
        random_forest_model,
        name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    print("\nRandom Forest " "MLflow run completed.")

    print(rf_metrics)


# ==================================================
# 8. COMPLETE
# ==================================================

print("\nMLflow experiment tracking completed.")

print(f"Experiment: {EXPERIMENT_NAME}")

print("Run 'mlflow ui' to view the experiments.")
