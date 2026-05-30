import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("merged_cleaned_retail_data.xlsx", engine="openpyxl")

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nSummary Statistics:")
print(df.describe())

if "Total_Amount" in df.columns:
    plt.figure(figsize=(8, 5))
    df["Total_Amount"].hist()
    plt.title("Total Amount Distribution")
    plt.xlabel("Total Amount")
    plt.ylabel("Frequency")
    plt.show()

print("Week 1 EDA Completed")