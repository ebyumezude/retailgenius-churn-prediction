import pandas as pd


# Load the cleaned dataset
file_path = "data/processed/ecommerce_churn_cleaned.csv"

df = pd.read_csv(file_path)


# Identify categorical columns
categorical_columns = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus"
]


# Display unique values in each categorical column
print("Categorical feature inspection:")

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts())

    # -----------------------------------
# STANDARDIZE CATEGORICAL VALUES
# -----------------------------------

# Standardize login device names
df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace({
    "Phone": "Mobile Phone"
})

# Standardize payment mode names
df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace({
    "CC": "Credit Card",
    "COD": "Cash on Delivery"
})

print("\nCategories after standardization:")

print("\nPreferredLoginDevice:")
print(df["PreferredLoginDevice"].value_counts())

print("\nPreferredPaymentMode:")
print(df["PreferredPaymentMode"].value_counts())

# -----------------------------------
# ONE-HOT ENCODING
# -----------------------------------

# Convert categorical variables into numerical variables
df_encoded = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)


# Display the new dataset information
print("\nDataset shape after encoding:")
print(df_encoded.shape)

print("\nColumns after encoding:")
print(df_encoded.columns.tolist())

print("\nFirst 5 rows of encoded dataset:")
print(df_encoded.head())

# Save the feature-engineered dataset
output_path = "data/processed/ecommerce_churn_features.csv"

df_encoded.to_csv(output_path, index=False)

print(f"\nFeature-engineered dataset saved to: {output_path}")