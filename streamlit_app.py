import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.title("RetailPulse Dashboard")

df = pd.read_excel(
    "merged_cleaned_retail_data.xlsx"
)

st.subheader("Dataset Preview")

st.dataframe(df.head())

st.subheader("Dataset Shape")

st.write(df.shape)

st.subheader("Missing Values")

st.write(df.isnull().sum())

numeric_cols = df.select_dtypes(
    include='number'
).columns

selected_col = st.selectbox(
    "Select Column",
    numeric_cols
)

fig = px.histogram(
    df,
    x=selected_col
)

st.plotly_chart(fig)

st.subheader("Correlation Heatmap")

corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10,8))

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    ax=ax
)

st.pyplot(fig)
st.subheader("Data Cleaning")
df = df.drop_duplicates()
df = df.dropna()

st.write("Cleaned Shape:", df.shape)

if 'Quantity' in df.columns and 'Price' in df.columns:

    df['TotalAmount'] = (
        df['Quantity'] * df['Price']
    )

    st.subheader("Total Amount")

    st.write(df['TotalAmount'].head())

if 'Invoice Date' in df.columns:

    df['Invoice Date'] = pd.to_datetime(
        df['Invoice Date']
    )

    snapshot_date = df['Invoice Date'].max()

    rfm = df.groupby('Customer ID').agg({

        'Invoice Date': lambda x:
            (snapshot_date - x.max()).days,

        'Invoice': 'nunique',

        'TotalAmount': 'sum'

    })

    rfm.columns = [
        'Recency',
        'Frequency',
        'Monetary'
    ]

    st.subheader("RFM Table")

    st.dataframe(rfm.head())
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans2

st.subheader("Customer Segmentation")

scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    rfm[['Recency', 'Frequency', 'Monetary']]
)

centroids, labels = kmeans2(
    scaled_data,
    4,
    minit='points'
)

rfm['Cluster'] = labels

st.write(rfm.head())
fig = px.scatter(
    rfm,
    x='Frequency',
    y='Monetary',
    color=rfm['Cluster'].astype(str)
)

st.plotly_chart(fig)
st.subheader("Time Series Analysis")

daily_sales = df.groupby(
    df['Invoice Date'].dt.date
)['TotalAmount'].sum()

daily_sales.index = pd.to_datetime(
    daily_sales.index
)

daily_sales = daily_sales.reset_index()

daily_sales.columns = ['Date', 'Sales']

fig = px.line(
    daily_sales,
    x='Date',
    y='Sales'
)

st.plotly_chart(fig)
from statsmodels.tsa.stattools import adfuller

result = adfuller(
    daily_sales['Sales']
)

st.write("ADF Statistic:", result[0])

st.write("P-value:", result[1])
# !pip install prophet
from prophet import Prophet

st.subheader("Sales Forecasting")

prophet_df = daily_sales.rename(
    columns={
        'Date':'ds',
        'Sales':'y'
    }
)

model = Prophet()

model.fit(prophet_df)

future = model.make_future_dataframe(
    periods=30
)

forecast = model.predict(future)

forecast_fig = px.line(
    forecast,
    x='ds',
    y='yhat'
)

st.plotly_chart(forecast_fig)
#pip install torch
import torch
import torch.nn as nn

st.subheader("LSTM Model")

class LSTMModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):

        out, _ = self.lstm(x)

        return self.fc(out[:, -1, :])

model = LSTMModel()

st.write(model)

rfm.to_csv(
    "outputs/customer_segmentation.csv"
)

df.to_csv(
    "outputs/cleaned_dataset.csv",
    index=False
)

st.success("Files Saved Successfully")
