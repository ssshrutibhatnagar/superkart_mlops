import os
import tempfile
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from huggingface_hub import hf_hub_download

MODEL_REPO_ID = os.getenv('MODEL_REPO_ID', 'ssshruti/superkart-sales-forecasting-model')
MODEL_FILENAME = os.getenv('MODEL_FILENAME', 'sales_prediction_model_v1_0.joblib')

app = Flask('Superkart Sales Predictor')

model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME, repo_type='model')
model = joblib.load(model_path)

FEATURE_COLUMNS = [
    'Product_Weight',
    'Product_Allocated_Area',
    'Product_MRP',
    'Product_Sugar_Content',
    'Product_Type',
    'Store_Establishment_Year',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type'
]

@app.get('/')
def home():
    return jsonify({'message': 'Welcome to the SuperKart Total Sales Prediction API', 'model_repo': MODEL_REPO_ID})

@app.post('/v1/storesales')
def predict_store_sales():
    try:
        payload = request.get_json(force=True)
        input_df = pd.DataFrame([payload])[FEATURE_COLUMNS]
        prediction = float(model.predict(input_df)[0])
        return jsonify({'predicted_product_store_sales_total': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.post('/v1/storesalesbatch')
def predict_store_sales_batch():
    try:
        uploaded_file = request.files.get('file')
        if uploaded_file is None:
            return jsonify({'error': 'Please upload a CSV file using form field name file.'}), 400
        input_df = pd.read_csv(uploaded_file)
        preds = model.predict(input_df[FEATURE_COLUMNS])
        output_df = input_df.copy()
        output_df['predicted_product_store_sales_total'] = preds.round(2)
        return output_df.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)}), 400
