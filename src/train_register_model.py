
import os
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download, upload_file, create_repo

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HF_DATASET_REPO = os.environ["HF_DATASET_REPO"]
HF_MODEL_REPO = os.environ["HF_MODEL_REPO"]

TARGET = "Product_Store_Sales_Total"

def main():
    os.makedirs("model", exist_ok=True)

    print("Downloading train and test data from Hugging Face Dataset Hub...")

    train_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename="data/train.csv",
        repo_type="dataset"
    )

    test_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename="data/test.csv",
        repo_type="dataset"
    )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    print("Training final Random Forest model...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("Model Evaluation Results")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

    model_path = "model/sales_prediction_model_v1_0.joblib"
    joblib.dump(pipeline, model_path)

    print("Model saved locally:", model_path)

    create_repo(
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        exist_ok=True
    )

    upload_file(
        path_or_fileobj=model_path,
        path_in_repo="sales_prediction_model_v1_0.joblib",
        repo_id=HF_MODEL_REPO,
        repo_type="model"
    )

    print("Model uploaded successfully to Hugging Face Model Hub.")

if __name__ == "__main__":
    main()
