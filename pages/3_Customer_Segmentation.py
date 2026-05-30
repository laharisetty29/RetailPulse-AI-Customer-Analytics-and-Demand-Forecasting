import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("👥 Customer Segmentation")

df = pd.read_excel("merged_cleaned_retail_data.xlsx", engine="openpyxl", nrows=15000)

df["Invoice Date"] = pd.to_datetime(df["Invoice Date"], errors="coerce")

if "TotalAmount" not in df.columns:
    df["TotalAmount"] = df["Quantity"] * df["Price"]

snapshot_date = df["Invoice Date"].max()

rfm = df.groupby("Customer ID").agg({
    "Invoice Date": lambda x: (snapshot_date - x.max()).days,
    "Invoice": "nunique",
    "TotalAmount": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

rfm = rfm.reset_index()

st.subheader("RFM Table")
st.dataframe(rfm.head())

if len(rfm) < 2:
    st.warning("Not enough customers for clustering.")
else:
    rfm_numeric = rfm[["Recency", "Frequency", "Monetary"]]

    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm_numeric)

    n_clusters = min(4, len(rfm))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm["Cluster"] = kmeans.fit_predict(scaled_rfm)

    fig = px.scatter(
        rfm,
        x="Frequency",
        y="Monetary",
        color=rfm["Cluster"].astype(str),
        title="RFM Customer Segmentation"
    )

    st.plotly_chart(fig, width="stretch")