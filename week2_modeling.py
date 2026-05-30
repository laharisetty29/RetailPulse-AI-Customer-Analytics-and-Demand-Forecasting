import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_excel("merged_cleaned_retail_data.xlsx", engine="openpyxl")

print("Dataset Loaded:", df.shape)

# Customer Segmentation
seg_cols = []

for col in ["Total_Amount", "Profit", "Discount_Percentage"]:
    if col in df.columns:
        seg_cols.append(col)

if len(seg_cols) >= 2:
    X_seg = df[seg_cols].fillna(0)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Customer_Segment_Label"] = kmeans.fit_predict(X_seg)

    df.to_csv("outputs/customer_segmentation.csv", index=False)
    print("Customer Segmentation Completed")

# Churn Prediction
if "Churn" in df.columns:
    data = df.copy()

    for col in data.select_dtypes(include="object").columns:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Churn Model Report:")
    print(classification_report(y_test, y_pred))

print("Week 2 Modeling Completed")