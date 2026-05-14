import os
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans2
from statsmodels.tsa.stattools import adfuller

st.set_page_config(page_title="RetailPulse Dashboard", layout="wide")

st.title("RetailPulse Dashboard")
st.write("Week 1: EDA, Data Cleaning, RFM, Segmentation and Time-Series Analysis")

file_path = "merged_cleaned_retail_data.xlsx"

if not os.path.exists(file_path):
    st.error("Dataset file not found. Please upload merged_cleaned_retail_data.xlsx to GitHub root folder.")
    st.stop()

df = pd.read_excel(file_path, engine="openpyxl", nrows=5000)

st.success("Dataset loaded successfully")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Shape")
st.write(df.shape)

st.subheader("Missing Values")
st.dataframe(df.isnull().sum().reset_index().rename(columns={"index": "Column", 0: "Missing Values"}))

st.subheader("Data Cleaning")
df = df.drop_duplicates()
df = df.dropna()
st.write("Cleaned Shape:", df.shape)

if "Quantity" in df.columns and "Price" in df.columns:
    df["TotalAmount"] = df["Quantity"] * df["Price"]

if "Invoice Date" in df.columns:
    df["Invoice Date"] = pd.to_datetime(df["Invoice Date"], errors="coerce")

st.subheader("Summary Statistics")
st.dataframe(df.describe())

numeric_cols = df.select_dtypes(include="number").columns

if len(numeric_cols) > 0:
    selected_col = st.selectbox("Select Numeric Column", numeric_cols)
    fig = px.histogram(df, x=selected_col, title=f"{selected_col} Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

if "Product_Category" in df.columns:
    st.subheader("Product Category Distribution")
    fig = px.histogram(df, x="Product_Category")
    st.plotly_chart(fig, use_container_width=True)

if "Customer_Type" in df.columns:
    st.subheader("Customer Type Distribution")
    fig = px.histogram(df, x="Customer_Type")
    st.plotly_chart(fig, use_container_width=True)

if "Churn" in df.columns:
    st.subheader("Churn Distribution")
    fig = px.histogram(df, x="Churn")
    st.plotly_chart(fig, use_container_width=True)

if "Invoice Date" in df.columns and "TotalAmount" in df.columns:
    st.subheader("Time Series Sales Analysis")

    daily_sales = df.groupby(df["Invoice Date"].dt.date)["TotalAmount"].sum()
    daily_sales.index = pd.to_datetime(daily_sales.index)

    daily_sales_df = daily_sales.reset_index()
    daily_sales_df.columns = ["Date", "Sales"]

    fig = px.line(daily_sales_df, x="Date", y="Sales", title="Daily Sales Trend")
    st.plotly_chart(fig, use_container_width=True)

    if len(daily_sales_df) > 20:
        result = adfuller(daily_sales_df["Sales"])
        st.write("ADF Statistic:", result[0])
        st.write("P-value:", result[1])

if all(col in df.columns for col in ["Customer ID", "Invoice Date", "Invoice", "TotalAmount"]):
    st.subheader("RFM Customer Segmentation")

    snapshot_date = df["Invoice Date"].max()

    rfm = df.groupby("Customer ID").agg({
        "Invoice Date": lambda x: (snapshot_date - x.max()).days,
        "Invoice": "nunique",
        "TotalAmount": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    centroids, labels = kmeans2(scaled_data, 4, minit="points")
    rfm["Cluster"] = labels

    st.dataframe(rfm.head())

    fig = px.scatter(
        rfm,
        x="Frequency",
        y="Monetary",
        color=rfm["Cluster"].astype(str),
        title="Customer Segmentation"
    )
    st.plotly_chart(fig, use_container_width=True)

st.success("Week 1 Dashboard Executed Successfully")