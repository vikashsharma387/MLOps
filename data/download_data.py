import os
import urllib.request
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(DATA_DIR, "heart.csv")
DATASET_URL = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"


def download_dataset():
    """Download the Heart Disease dataset from UCI repository mirror."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading dataset from {DATASET_URL}...")
    urllib.request.urlretrieve(DATASET_URL, OUTPUT_FILE)

    # Verify and clean BOM if present
    df = pd.read_csv(OUTPUT_FILE)
    df.columns = [col.lstrip("\ufeff") for col in df.columns]
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Dataset successfully saved to {OUTPUT_FILE}")
    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Class distribution:\n{df['target'].value_counts()}")


if __name__ == "__main__":
    download_dataset()
