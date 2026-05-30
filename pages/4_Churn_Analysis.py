import streamlit as st
import pandas as pd
import plotly.express as px

st.title("⚠️ Churn Analysis")

df = pd.read_excel("merged_cleaned_retail_data.xlsx", engine="openpyxl", nrows=15000)

if "Churn" in df.columns:
    churn_count = df["Churn"].value_counts().reset_index()
    churn_count.columns = ["Churn", "Count"]

    fig = px.bar(
        churn_count,
        x="Churn",
        y="Count",
        title="Churn Distribution"
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Churn Data Preview")
    st.dataframe(df[["Customer ID", "Customer_Type", "Churn"]].head())
else:
    st.warning("Churn column not found.")