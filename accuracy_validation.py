import pandas as pd

df = pd.read_excel(
    "merged_cleaned_retail_data.xlsx",
    engine="openpyxl",
    nrows=10000
)

print("Dataset Loaded Successfully")

print("Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nValidation Completed Successfully")