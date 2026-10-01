import joblib
import pandas as pd

# ==================================================
# 1. LOAD TRAINED CHURN PREDICTION PIPELINE
# ==================================================

model_path = "models/random_forest_churn_model.pkl"

model = joblib.load(model_path)

print("Churn prediction pipeline loaded successfully.")


# ==================================================
# 2. CREATE EXAMPLE NEW CUSTOMER
# ==================================================

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
    "MaritalStatus": "Single",
}

customer_df = pd.DataFrame([new_customer])

print("\nNew customer:")
print(customer_df)


# ==================================================
# 3. STANDARDIZE CATEGORICAL VALUES
# ==================================================

customer_df["PreferredLoginDevice"] = customer_df["PreferredLoginDevice"].replace(
    {
        "Phone": "Mobile Phone",
    }
)

customer_df["PreferredPaymentMode"] = customer_df["PreferredPaymentMode"].replace(
    {
        "CC": "Credit Card",
        "COD": "Cash on Delivery",
    }
)


# ==================================================
# 4. PREDICT CUSTOMER CHURN
# ==================================================

prediction = model.predict(customer_df)

prediction_probability = model.predict_proba(customer_df)

probability_staying = prediction_probability[0][0]
probability_churning = prediction_probability[0][1]


# ==================================================
# 5. DISPLAY RESULTS
# ==================================================

print("\n--- Churn Prediction ---")

print("Predicted class:", prediction[0])

print(f"Probability of staying: " f"{probability_staying:.2%}")

print(f"Probability of churning: " f"{probability_churning:.2%}")

if prediction[0] == 1:
    print("Prediction: Customer is likely to churn.")
else:
    print("Prediction: Customer is likely to stay.")
