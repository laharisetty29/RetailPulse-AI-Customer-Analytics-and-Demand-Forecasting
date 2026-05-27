import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Sales Analysis")

df = pd.read_excel("merged_cleaned_retail_data.xlsx", engine="openpyxl", nrows=15000)

df["Invoice Date"] = pd.to_datetime(df["Invoice Date"], errors="coerce")

if "TotalAmount" not in df.columns:
    df["TotalAmount"] = df["Quantity"] * df["Price"]

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", round(df["TotalAmount"].sum(), 2))
col2.metric("Total Orders", df["Invoice"].nunique())
col3.metric("Total Customers", df["Customer ID"].nunique())

daily_sales = df.groupby(df["Invoice Date"].dt.date)["TotalAmount"].sum().reset_index()
daily_sales.columns = ["Date", "Sales"]

fig = px.line(daily_sales, x="Date", y="Sales", title="Daily Sales Trend")
st.plotly_chart(fig, width="stretch")

if "Product_Category" in df.columns:
    category_sales = df.groupby("Product_Category")["TotalAmount"].sum().reset_index()

    fig = px.bar(
        category_sales,
        x="Product_Category",
        y="TotalAmount",
        title="Sales by Product Category"
    )

    st.plotly_chart(fig, width="stretch")