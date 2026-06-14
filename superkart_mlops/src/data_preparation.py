
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, upload_file

# Hard-coded repo details to avoid blank GitHub secret issue
HF_DATASET_REPO = "ssshruti/superkart-sales-data"
RAW_FILE = "data/SuperKart_MLOps.csv"
TARGET = "Product_Store_Sales_Total"

def main():
    print("Downloading raw dataset from Hugging Face Dataset Hub...")
    print("Dataset repo:", HF_DATASET_REPO)
    print("Raw file:", RAW_FILE)

    raw_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=RAW_FILE,
        repo_type="dataset"
    )

    df = pd.read_csv(raw_path)
    print("Raw dataset loaded successfully.")
    print("Raw shape:", df.shape)

    # Drop unnecessary ID columns if present
    columns_to_drop = ["Product_Id", "Store_Id"]
    df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    print("Cleaned shape:", df_cleaned.shape)

    X = df_cleaned.drop(columns=[TARGET])
    y = df_cleaned[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    train_df = X_train.copy()
    train_df[TARGET] = y_train.values

    test_df = X_test.copy()
    test_df[TARGET] = y_test.values

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Train and test datasets saved locally.")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    upload_file(
        path_or_fileobj=str(train_path),
        path_in_repo="data/train.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset"
    )

    upload_file(
        path_or_fileobj=str(test_path),
        path_in_repo="data/test.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset"
    )

    print("Prepared train and test datasets uploaded to Hugging Face Dataset Hub.")

if __name__ == "__main__":
    main()
