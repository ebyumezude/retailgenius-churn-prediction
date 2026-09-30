import joblib
import pandas as pd

# --------------------------------------------------
# Load trained churn prediction model
# --------------------------------------------------

model_path = "models/random_forest_churn_model.pkl"

model = joblib.load(model_path)

print("Churn prediction model loaded successfully.")

print(f"Model expects {model.n_features_in_} features.")

# --------------------------------------------------
# Example new customer
# --------------------------------------------------

new_customer = {
    "Tenure": 3,
    "CityTier": 2,
    "WarehouseToHome": 15,
    "HourSpendOnApp": 3,
    "NumberOfDeviceRegistered": 4,
    "SatisfactionScore": 2,
    "NumberOfAddress": 3,
    "Complain": 1,
    "OrderAmountHikeFromlastYear": 12,
    "CouponUsed": 2,
    "OrderCount": 3,
    "DaySinceLastOrder": 8,
    "CashbackAmount": 150,
    "PreferredLoginDevice": "Mobile Phone",
    "PreferredPaymentMode": "Credit Card",
    "Gender": "Female",
    "PreferedOrderCat": "Fashion",
    "MaritalStatus": "Single"
}

customer_df = pd.DataFrame([new_customer])

print("\nNew customer:")
print(customer_df)

# --------------------------------------------------
# Standardize categorical values
# --------------------------------------------------

customer_df["PreferredLoginDevice"] = customer_df[
    "PreferredLoginDevice"
].replace({
    "Phone": "Mobile Phone"
})

customer_df["PreferredPaymentMode"] = customer_df[
    "PreferredPaymentMode"
].replace({
    "CC": "Credit Card",
    "COD": "Cash on Delivery"
})


# --------------------------------------------------
# One-hot encode categorical features
# --------------------------------------------------

categorical_columns = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus"
]

# --------------------------------------------------
# Encode new customer using training data structure
# --------------------------------------------------

# Load the original prepared dataset
reference_df = pd.read_csv("data/processed/ecommerce_churn_cleaned.csv")

# Remove target because the new customer does not have Churn yet
reference_features = reference_df.drop(columns=["Churn"])

# Temporarily add the new customer to the reference data
combined = pd.concat(
    [reference_features, customer_df],
    ignore_index=True
)

# Apply exactly the same dummy encoding structure
combined_encoded = pd.get_dummies(
    combined,
    drop_first=True
)

# Extract only our new customer again
customer_encoded = combined_encoded.tail(1).copy()

print("\nCustomer after reference-based encoding:")
print(customer_encoded)

print("\nShape before model alignment:")
print(customer_encoded.shape)

print("\nCustomer after initial encoding:")
print(customer_encoded)

print("\nShape after initial encoding:")
print(customer_encoded.shape)

# --------------------------------------------------
# Align customer features with trained model
# --------------------------------------------------
print("\nColumns BEFORE alignment:")
for column in customer_encoded.columns:
    print(column)

customer_encoded = customer_encoded.reindex(
    columns=model.feature_names_in_,
    fill_value=0
)

print("\nCustomer aligned with model features:")
print(customer_encoded)

print("\nFinal customer shape:")
print(customer_encoded.shape)

print("\nColumns match model:")
print(list(customer_encoded.columns) == list(model.feature_names_in_))

# --------------------------------------------------
# Step 9E: Predict churn for the new customer
# --------------------------------------------------

prediction = model.predict(customer_encoded)

prediction_probability = model.predict_proba(customer_encoded)

print("\n--- Churn Prediction ---")
print("Predicted class:", prediction[0])

print("\nPrediction probabilities:")
print(prediction_probability[0])

print("\nProbability of staying:", prediction_probability[0][0])
print("Probability of churning:", prediction_probability[0][1])