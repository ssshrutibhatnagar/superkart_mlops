import streamlit as st
import requests

st.title("SuperKart Store Sales Prediction")

BACKEND_ROOT = "https://ssshruti-superkart-model-deployment-p.hf.space"
ONLINE_URL = f"{BACKEND_ROOT}/v1/storesales"
BATCH_URL  = f"{BACKEND_ROOT}/v1/storesalesbatch"

st.subheader("Online Prediction")

Product_Sugar_Content = st.selectbox(
    "Product Sugar Content",
    ["Low Sugar", "Regular", "No Sugar", "reg"]
)

Product_Weight = st.number_input(
    "Product Weight",
    min_value=0.1,
    max_value=50.0,
    step=0.1,
    value=10.0
)

Product_Allocated_Area = st.number_input(
    "Product Allocated Area",
    min_value=0.0,
    max_value=5.0,
    step=0.001,
    value=0.01
)

Product_MRP = st.number_input(
    "Product MRP",
    min_value=5.0,
    max_value=1000.0,
    step=1.0,
    value=90.0
)

Product_Type = st.selectbox(
    "Product Type",
    [
        "Frozen Foods", "Dairy", "Canned", "Baking Goods", "Health and Hygiene",
        "Snack Foods", "Meat", "Household", "Hard Drinks",
        "Fruits and Vegetables", "Breads", "Soft Drinks", "Breakfast",
        "Others", "Starchy Foods", "Seafood"
    ]
)

Store_Establishment_Year = st.selectbox(
    "Store Establishment Year",
    [1987, 1998, 1999, 2009]
)

Store_Size = st.selectbox(
    "Store Size",
    ["High", "Medium", "Small"]
)

Store_Location_City_Type = st.selectbox(
    "City Type",
    ["Tier 1", "Tier 2", "Tier 3"]
)

Store_Type = st.selectbox(
    "Store Type",
    [
        "Departmental Store",
        "Food Mart",
        "Supermarket Type1",
        "Supermarket Type2"
    ]
)

payload = {
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Weight": float(Product_Weight),
    "Product_Allocated_Area": float(Product_Allocated_Area),
    "Product_MRP": float(Product_MRP),
    "Product_Type": Product_Type,
    "Store_Establishment_Year": int(Store_Establishment_Year),
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type
}

if st.button("Predict"):
    try:
        resp = requests.post(ONLINE_URL, json=payload, timeout=60)

        if resp.status_code == 200:
            result = resp.json()

            prediction = result.get("predicted_product_store_sales_total")

            if prediction is not None:
                st.success(f"Predicted Product Store Sales Total: {prediction}")
            else:
                st.error("Prediction key not found in API response.")
                st.write(result)

        else:
            st.error(f"API Error: {resp.status_code}")
            st.write(resp.text)

    except Exception as e:
        st.error("Unable to connect to backend API.")
        st.write(str(e))

st.subheader("Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

if uploaded_file is not None and st.button("Predict Batch"):
    try:
        resp = requests.post(
            BATCH_URL,
            files={"file": uploaded_file},
            timeout=120
        )

        if resp.status_code == 200:
            st.success("Batch predictions completed!")
            st.write(resp.json())
        else:
            st.error(f"API Error: {resp.status_code}")
            st.write(resp.text)

    except Exception as e:
        st.error("Unable to connect to backend API.")
        st.write(str(e))
