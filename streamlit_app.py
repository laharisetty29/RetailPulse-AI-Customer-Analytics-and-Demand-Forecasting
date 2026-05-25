import os
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="RetailPulse Advanced Analytics Dashboard",
    layout="wide"
)

st.title("RetailPulse Advanced Analytics Dashboard")

st.write(
    "Interactive Retail Analytics, Churn Insights, Inventory Insights and Customer Segmentation"
)

file_path = "merged_cleaned_retail_data.xlsx"

if not os.path.exists(file_path):

    st.error(
        "Dataset file not found. Keep merged_cleaned_retail_data.xlsx in the same folder."
    )

    st.stop()

df = pd.read_excel(
    file_path,
    engine="openpyxl",
    nrows=15000
)

df["Stock Code"] = df["Stock Code"].astype(str)

df = df.drop_duplicates()

df = df.dropna()

df["Invoice Date"] = pd.to_datetime(
    df["Invoice Date"],
    errors="coerce"
)

if "TotalAmount" not in df.columns:

    df["TotalAmount"] = (
        df["Quantity"] * df["Price"]
    )

original_df = df.copy()

st.sidebar.header("Dashboard Filters")

filtered_df = original_df.copy()

if "Product_Category" in original_df.columns:

    selected_category = st.sidebar.multiselect(
        "Select Product Category",
        sorted(original_df["Product_Category"].unique()),
        default=sorted(original_df["Product_Category"].unique())
    )

    if selected_category:

        filtered_df = filtered_df[
            filtered_df["Product_Category"].isin(
                selected_category
            )
        ]

if "Customer_Type" in original_df.columns:

    selected_customer_type = st.sidebar.multiselect(
        "Select Customer Type",
        sorted(original_df["Customer_Type"].unique()),
        default=sorted(original_df["Customer_Type"].unique())
    )

    if selected_customer_type:

        filtered_df = filtered_df[
            filtered_df["Customer_Type"].isin(
                selected_customer_type
            )
        ]

if "Country" in original_df.columns:

    selected_country = st.sidebar.multiselect(
        "Select Country",
        sorted(original_df["Country"].unique()),
        default=sorted(original_df["Country"].unique())
    )

    if selected_country:

        filtered_df = filtered_df[
            filtered_df["Country"].isin(
                selected_country
            )
        ]

df = filtered_df.copy()

if df.empty:

    st.warning(
        "No data available for selected filters. Please select different filter values."
    )

    st.stop()

st.subheader("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Sales",
    round(df["TotalAmount"].sum(), 2)
)

if "Profit" in df.columns:

    col2.metric(
        "Total Profit",
        round(df["Profit"].sum(), 2)
    )

else:

    col2.metric(
        "Total Profit",
        "NA"
    )

col3.metric(
    "Total Customers",
    df["Customer ID"].nunique()
)

col4.metric(
    "Total Orders",
    df["Invoice"].nunique()
)

st.subheader("Dataset Preview")

st.dataframe(df.head())

st.subheader("Sales Trend Analysis")

daily_sales = df.groupby(
    df["Invoice Date"].dt.date
)["TotalAmount"].sum().reset_index()

daily_sales.columns = [
    "Date",
    "Sales"
]

fig = px.line(
    daily_sales,
    x="Date",
    y="Sales",
    title="Daily Sales Trend"
)

st.plotly_chart(
    fig,
    width="stretch"
)

st.subheader("Product Category Sales")

if "Product_Category" in df.columns:

    category_sales = df.groupby(
        "Product_Category"
    )["TotalAmount"].sum().reset_index()

    fig = px.bar(
        category_sales,
        x="Product_Category",
        y="TotalAmount",
        title="Sales by Product Category"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

st.subheader("Customer Type Analysis")

if "Customer_Type" in df.columns:

    customer_sales = df.groupby(
        "Customer_Type"
    )["TotalAmount"].sum().reset_index()

    fig = px.pie(
        customer_sales,
        names="Customer_Type",
        values="TotalAmount",
        title="Sales Share by Customer Type"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

st.subheader("Churn Analysis")

if "Churn" in df.columns:

    churn_count = df["Churn"].value_counts().reset_index()

    churn_count.columns = [
        "Churn",
        "Count"
    ]

    fig = px.bar(
        churn_count,
        x="Churn",
        y="Count",
        title="Churn Distribution"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

st.subheader("Inventory Insights")

if "Current_Stock" in df.columns:

    df["Reorder_Level"] = df["Current_Stock"].apply(
        lambda x: 100 - x if x < 100 else 0
    )

    df["Inventory_Status"] = df["Current_Stock"].apply(
        lambda x:
            "Reorder Required"
            if x < 100
            else "Stock Available"
    )

    inventory_status = df[
        "Inventory_Status"
    ].value_counts().reset_index()

    inventory_status.columns = [
        "Inventory Status",
        "Count"
    ]

    fig = px.bar(
        inventory_status,
        x="Inventory Status",
        y="Count",
        title="Inventory Status"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    show_cols = [
        "Stock Code",
        "Description",
        "Current_Stock",
        "Reorder_Level",
        "Inventory_Status"
    ]

    available_cols = [
        col for col in show_cols
        if col in df.columns
    ]

    st.dataframe(
        df[available_cols].head(20)
    )

st.subheader("Customer Segmentation")

snapshot_date = df["Invoice Date"].max()

rfm = df.groupby(
    "Customer ID"
).agg({

    "Invoice Date": lambda x:
        (snapshot_date - x.max()).days,

    "Invoice": "nunique",

    "TotalAmount": "sum"

})

rfm.columns = [
    "Recency",
    "Frequency",
    "Monetary"
]

rfm = rfm.reset_index()

rfm_numeric = rfm[[
    "Recency",
    "Frequency",
    "Monetary"
]]

scaler = StandardScaler()

scaled_rfm = scaler.fit_transform(
    rfm_numeric
)

kmeans = KMeans(
    n_clusters=4,
    random_state=42
)

rfm["Cluster"] = kmeans.fit_predict(
    scaled_rfm
)

st.dataframe(
    rfm.head()
)

fig = px.scatter(
    rfm,
    x="Frequency",
    y="Monetary",
    color=rfm["Cluster"].astype(str),
    title="RFM Customer Segmentation"
)

st.plotly_chart(
    fig,
    width="stretch"
)

st.subheader("Download Processed Data")

csv_data = df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Filtered Dataset",
    data=csv_data,
    file_name="week3_filtered_retail_data.csv",
    mime="text/csv"
)

st.success(
    "Week 3 Advanced Dashboard Completed Successfully"
)