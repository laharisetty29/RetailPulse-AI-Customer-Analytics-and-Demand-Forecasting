import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 EDA – Exploratory Data Analysis")

df = pd.read_excel("merged_cleaned_retail_data.xlsx", engine="openpyxl", nrows=15000)

df["Stock Code"] = df["Stock Code"].astype(str)

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Shape")
st.write(df.shape)

st.subheader("Missing Values")
st.dataframe(df.isnull().sum())

st.subheader("Summary Statistics")
st.dataframe(df.describe())

st.subheader("Correlation Heatmap")

numeric_cols = df.select_dtypes(include="number").columns

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)