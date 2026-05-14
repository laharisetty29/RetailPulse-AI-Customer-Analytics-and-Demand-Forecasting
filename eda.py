import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

st.set_page_config(
    page_title="RetailPulse Dashboard",
    layout="wide"
)

st.title("RetailPulse – Week 1 Dashboard")

st.write(
    "EDA, Sales Analysis, Customer Segmentation and Churn Overview"
)

file_path = "merged_cleaned_retail_data.xlsx"
st.write("Checking file...")

if not os.path.exists(file_path):

    st.error("Excel file not found")
    st.stop()

st.success("File found successfully")

with st.spinner("Loading dataset... Please wait"):

    df = pd.read_excel(
        file_path,
        engine="openpyxl",
        nrows=5000
    )

st.success("Dataset Loaded Successfully")

st.subheader("Dataset Preview")

st.dataframe(df.head())

st.subheader("Dataset Shape")

st.write(df.shape)

st.subheader("Column Names")

st.write(df.columns.tolist())

st.subheader("Missing Values")

missing = df.isnull().sum().reset_index()

missing.columns = [
    "Column",
    "Missing Values"
]

st.dataframe(missing)

st.subheader("Summary Statistics")

st.dataframe(df.describe())

if "Invoice Date" in df.columns:

    df["Invoice Date"] = pd.to_datetime(
        df["Invoice Date"],
        errors="coerce"
    )

if "Total_Amount" not in df.columns:

    if "Quantity" in df.columns and "Price" in df.columns:

        df["Total_Amount"] = (
            df["Quantity"] * df["Price"]
        )

st.subheader("Sales Distribution")

fig, ax = plt.subplots(figsize=(8, 4))

sns.histplot(
    df["Total_Amount"],
    bins=30,
    ax=ax
)

st.pyplot(fig)

if "Profit" in df.columns:

    st.subheader("Profit Distribution")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        df["Profit"],
        bins=30,
        ax=ax
    )

    st.pyplot(fig)

if "Product_Category" in df.columns:

    st.subheader("Product Category Distribution")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.countplot(
        x="Product_Category",
        data=df,
        ax=ax
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

if "Customer_Type" in df.columns:

    st.subheader("Customer Type Distribution")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.countplot(
        x="Customer_Type",
        data=df,
        ax=ax
    )

    st.pyplot(fig)

if "Churn" in df.columns:

    st.subheader("Churn Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.countplot(
        x="Churn",
        data=df,
        ax=ax
    )

    st.pyplot(fig)

if "Invoice Date" in df.columns:

    st.subheader("Daily Sales Trend")

    daily_sales = df.groupby(
        df["Invoice Date"].dt.date
    )["Total_Amount"].sum()

    st.line_chart(daily_sales)

st.subheader("Customer Segmentation")

features = []

for col in [
    "Total_Amount",
    "Profit",
    "Current_Stock"
]:

    if col in df.columns:

        features.append(col)

if len(features) >= 2:

    data = df[features].dropna()

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42
    )

    data["Cluster"] = kmeans.fit_predict(
        scaled_data
    )

    st.dataframe(data.head())

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        x=data[features[0]],
        y=data[features[1]],
        hue=data["Cluster"],
        ax=ax
    )

    plt.title("Customer Segmentation")

    st.pyplot(fig)

else:

    st.warning(
        "Not enough columns for clustering"
    )

st.success(
    "Week 1 Dashboard Executed Successfully"
)