import joblib

# --------------------------------------------------
# Load the saved Random Forest model
# --------------------------------------------------

model_path = "models/random_forest_churn_model.pkl"

model = joblib.load(model_path)

print("Model loaded successfully.")

print("\nModel type:")
print(type(model))

print("\nNumber of trees:")


print("\nModel parameters:")
print(model.get_params())

# --------------------------------------------------
# Inspect the features expected by the model
# --------------------------------------------------

print("\nNumber of features expected by model:")
print(model.n_features_in_)

print("\nFeatures expected by model:")
print(model.feature_names_in_)

# --------------------------------------------------
# Make a prediction using one customer
# --------------------------------------------------

import pandas as pd

# Load the feature-engineered dataset
data_path = "data/processed/ecommerce_churn_cleaned.csv"
df = pd.read_csv(data_path)

# Separate features from the actual churn result
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Select one customer
customer = X.iloc[[0]]

# Actual churn value for comparison
actual_churn = y.iloc[0]

# Make prediction
prediction = model.predict(customer)

# Get prediction probabilities
probability = model.predict_proba(customer)

print("\n--- First Customer Prediction ---")

print("\nPredicted class:")
print(prediction[0])

print("\nActual class:")
print(actual_churn)

print("\nPrediction probabilities:")
print(probability[0])
