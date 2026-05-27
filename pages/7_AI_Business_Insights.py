import streamlit as st
import pandas as pd

st.title("AI Business Insights")

df = pd.read_excel(
    "merged_cleaned_retail_data.xlsx",
    engine="openpyxl",
    nrows=15000
)

if "TotalAmount" not in df.columns:
    df["TotalAmount"] = (
        df["Quantity"] * df["Price"]
    )

st.subheader("Business KPIs")

total_sales = df["TotalAmount"].sum()

total_profit = df["Profit"].sum()

total_customers = df["Customer ID"].nunique()

total_orders = df["Invoice"].nunique()

col1, col2 = st.columns(2)

col1.metric(
    "Total Sales",
    round(total_sales, 2)
)

col2.metric(
    "Total Profit",
    round(total_profit, 2)
)

col3, col4 = st.columns(2)

col3.metric(
    "Total Customers",
    total_customers
)

col4.metric(
    "Total Orders",
    total_orders
)

st.subheader("AI Generated Business Recommendations")

top_category = df.groupby(
    "Product_Category"
)["TotalAmount"].sum().idxmax()

top_country = df.groupby(
    "Country"
)["TotalAmount"].sum().idxmax()

high_churn = df["Churn"].mean() * 100

st.success(
    f"Top selling category is {top_category}"
)

st.info(
    f"Highest sales country is {top_country}"
)

st.warning(
    f"Estimated churn rate is {round(high_churn, 2)}%"
)

st.subheader("Inventory Recommendations")

low_stock = df[
    df["Current_Stock"] < 50
]

st.dataframe(
    low_stock[[
        "Stock Code",
        "Description",
        "Current_Stock"
    ]].head(20)
)

csv = low_stock.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Inventory Report",
    data=csv,
    file_name="inventory_report.csv",
    mime="text/csv"
)

st.success(
    "AI Business Insights Generated Successfully"
)