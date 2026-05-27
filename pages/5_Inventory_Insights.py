import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Inventory Insights")

df = pd.read_excel("merged_cleaned_retail_data.xlsx", engine="openpyxl", nrows=15000)

df["Stock Code"] = df["Stock Code"].astype(str)

if "Current_Stock" in df.columns:
    df["Reorder_Level"] = df["Current_Stock"].apply(
        lambda x: 100 - x if x < 100 else 0
    )

    df["Inventory_Status"] = df["Current_Stock"].apply(
        lambda x: "Reorder Required" if x < 100 else "Stock Available"
    )

    inventory_status = df["Inventory_Status"].value_counts().reset_index()
    inventory_status.columns = ["Inventory Status", "Count"]

    fig = px.bar(
        inventory_status,
        x="Inventory Status",
        y="Count",
        title="Inventory Status"
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Inventory Recommendation Table")

    st.dataframe(
        df[[
            "Stock Code",
            "Description",
            "Current_Stock",
            "Reorder_Level",
            "Inventory_Status"
        ]].head(30)
    )
else:
    st.warning("Current_Stock column not found.")