
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, upload_file


HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "ssshruti/superkart-sales-data")

if not HF_DATASET_REPO:
    raise ValueError("HF_DATASET_REPO is empty. Add it as a GitHub Actions secret.")

RAW_FILE = "data/SuperKart_MLOps.csv"
TARGET = "Product_Store_Sales_Total"

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Product_Store_Sales_Total"

def main():
    os.makedirs("data", exist_ok=True)

    print("Downloading raw dataset from Hugging Face Dataset Hub...")

    raw_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename="data/SuperKart_MLOps.csv",
        repo_type="dataset"
    )

    df = pd.read_csv(raw_path)

    print("Raw dataset shape:", df.shape)

    # Drop unnecessary columns
    columns_to_drop = ["Product_Id", "Store_Id"]

    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=col)

    # Basic cleaning
    df = df.drop_duplicates()

    # Split data
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    train_path = "data/train.csv"
    test_path = "data/test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    # Upload train and test files back to HF Dataset Hub
    upload_file(
        path_or_fileobj=train_path,
        path_in_repo="data/train.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset"
    )

    upload_file(
        path_or_fileobj=test_path,
        path_in_repo="data/test.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset"
    )

    print("Train and test datasets uploaded successfully.")

if __name__ == "__main__":
    main()
