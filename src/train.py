import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load the feature-engineered dataset
file_path = "data/processed/ecommerce_churn_features.csv"

df = pd.read_csv(file_path)


# Separate features (X) and target (y)
X = df.drop(columns=["Churn"])
y = df["Churn"]


print("Feature matrix shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining set:")
print(X_train.shape)

print("\nTest set:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True) * 100)

# -----------------------------------
# LOGISTIC REGRESSION
# -----------------------------------

# Create the model
# -----------------------------------
# LOGISTIC REGRESSION
# -----------------------------------

# Create a pipeline that scales the features
# before training Logistic Regression
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Train the model
logistic_model.fit(X_train, y_train)

print("\nLogistic Regression training completed.")

# -----------------------------------
# EVALUATE LOGISTIC REGRESSION
# -----------------------------------

# Predict churn for the test customers
y_pred = logistic_model.predict(X_test)

# Predict probability of churn
y_prob = logistic_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nLogistic Regression Results:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------------
# RANDOM FOREST
# -----------------------------------

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
random_forest_model.fit(X_train, y_train)

# Make predictions
rf_pred = random_forest_model.predict(X_test)

# Predict churn probabilities
rf_prob = random_forest_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
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

# -----------------------------------
# CHECK FOR OVERFITTING
# -----------------------------------

rf_train_pred = random_forest_model.predict(X_train)

train_accuracy = accuracy_score(y_train, rf_train_pred)
test_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Overfitting Check:")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy:     {test_accuracy:.4f}")
print(f"Difference:        {train_accuracy - test_accuracy:.4f}")

# -----------------------------------
# FEATURE IMPORTANCE CHECK
# -----------------------------------

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": random_forest_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Random Forest Feature Importances:")
print(feature_importance.head(15).to_string(index=False))

# -----------------------------------
# CROSS-VALIDATION
# -----------------------------------

cv_scores = cross_val_score(
    random_forest_model,
    X_train,
    y_train,
    cv=5,
    scoring="roc_auc"
)

print("\nRandom Forest 5-Fold Cross-Validation:")
print("ROC-AUC scores:", cv_scores)
print(f"Mean ROC-AUC: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")
