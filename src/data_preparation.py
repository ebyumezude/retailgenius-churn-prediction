import pandas as pd


# Load the E-Commerce dataset
file_path = "data/raw/E Commerce Dataset.xlsx"

df = pd.read_excel(file_path, sheet_name="E Comm")


# Display basic information about the dataset
print("Dataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nChurn percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)
# -----------------------------------
# DATA CLEANING
# -----------------------------------

# Check for duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())


# Remove duplicate rows if any exist
df = df.drop_duplicates()


# Remove CustomerID because it is only an identifier
df = df.drop(columns=["CustomerID"])


# Columns containing missing numerical values
missing_value_columns = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder"
]


# Replace missing values with the median of each column
for column in missing_value_columns:
    df[column] = df[column].fillna(df[column].median())


# Check that missing values have been handled
print("\nMissing values after cleaning:")
print(df.isnull().sum())


# Display shape after cleaning
print("\nDataset shape after cleaning:")
print(df.shape)

# Save the cleaned dataset
output_path = "data/processed/ecommerce_churn_cleaned.csv"

df.to_csv(output_path, index=False)

print(f"\nCleaned dataset saved to: {output_path}")