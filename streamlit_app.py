import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="RetailPulse Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RetailPulse – AI Powered Retail Analytics Platform")

st.markdown(
    """
    RetailPulse is an interactive retail analytics dashboard for sales analysis,
    customer segmentation, churn insights, inventory monitoring, and demand forecasting.
    """
)

st.markdown("---")

# KPI Cards
st.subheader("📌 Business KPI Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Sales", "$2.5M")

with col2:
    st.metric("Customers", "15K")

with col3:
    st.metric("Products", "320")

with col4:
    st.metric("Forecast Accuracy", "92%")

st.markdown("---")

# Project Modules
st.subheader("🚀 Dashboard Modules")

col5, col6 = st.columns(2)

with col5:
    st.info("""
    **EDA**
    
    Explore dataset structure, missing values, trends, and patterns.
    """)

    st.info("""
    **Sales Analysis**
    
    Analyze sales performance, revenue trends, and category-wise sales.
    """)

    st.info("""
    **Customer Segmentation**
    
    Identify customer groups based on behavior and purchase patterns.
    """)

with col6:
    st.success("""
    **Churn Analysis**
    
    Detect customers who are likely to stop purchasing.
    """)

    st.success("""
    **Inventory Insights**
    
    Understand product demand and stock movement patterns.
    """)

    st.success("""
    **Sales Forecasting**
    
    Predict future sales using historical retail data.
    """)

st.markdown("---")

# Dataset Summary
st.subheader("📁 Dataset Summary")

try:
    df = pd.read_excel("merged_cleaned_retail_data.xlsx", engine="openpyxl", nrows=10000)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Rows Loaded", df.shape[0])

    with c2:
        st.metric("Columns", df.shape[1])

    with c3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))

except Exception as e:
    st.warning("Dataset summary could not be loaded. Please check the Excel file path.")

st.markdown("---")

# Navigation Message
st.success("Use the sidebar to navigate between dashboard pages.")

st.markdown("---")

# Footer
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Developed by <b>Gadamsetty Lahari</b> | Retail Analytics + AI + Deployment Project
    </div>
    """,
    unsafe_allow_html=True
)