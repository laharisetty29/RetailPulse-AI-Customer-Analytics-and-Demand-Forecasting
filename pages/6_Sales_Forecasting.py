import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

st.title("AI Sales Forecasting")

df = pd.read_excel(
    "merged_cleaned_retail_data.xlsx",
    engine="openpyxl",
    nrows=15000
)

df["Invoice Date"] = pd.to_datetime(
    df["Invoice Date"],
    errors="coerce"
)

if "TotalAmount" not in df.columns:
    df["TotalAmount"] = (
        df["Quantity"] * df["Price"]
    )

daily_sales = df.groupby(
    df["Invoice Date"].dt.date
)["TotalAmount"].sum().reset_index()

daily_sales.columns = [
    "ds",
    "y"
]

daily_sales["ds"] = pd.to_datetime(
    daily_sales["ds"]
)

st.subheader("Daily Sales Data")

st.dataframe(daily_sales.head())

model = Prophet()

model.fit(daily_sales)

future = model.make_future_dataframe(
    periods=30
)

forecast = model.predict(future)

st.subheader("Sales Forecast")

forecast_fig = px.line(
    forecast,
    x="ds",
    y="yhat",
    title="30 Days Sales Forecast"
)

st.plotly_chart(
    forecast_fig,
    width="stretch"
)

st.subheader("Forecast Dataset")

st.dataframe(
    forecast[[
        "ds",
        "yhat",
        "yhat_lower",
        "yhat_upper"
    ]].tail(30)
)

csv = forecast.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Forecast Report",
    data=csv,
    file_name="sales_forecast.csv",
    mime="text/csv"
)

st.success(
    "Sales Forecasting Completed"
)