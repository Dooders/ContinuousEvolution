import json
import os

import numpy as np
import pandas as pd
from data import Dataset
from sklearn.discriminant_analysis import StandardScaler


def process_json_to_csv(json_file: str) -> None:
    """
    Process a JSON file to a CSV file.

    Parameters
    ----------
    json_file : str
        The path to the JSON file.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    df_scaled = df.copy()
    df_scaled = df_scaled.drop(columns=["Timestamp"])
    scaler = StandardScaler()
    df_scaled[df_scaled.columns] = scaler.fit_transform(df_scaled[df_scaled.columns])
    df_scaled.to_csv("evolution/data/price_df.csv", index=False)


def create_dataset() -> np.ndarray:
    """
    Create a dataset from a CSV file.

    Returns
    -------
    np.ndarray
        The dataset.
    """
    df = pd.read_csv("evolution/data/price_df.csv")

    return df.to_numpy()


def load_dataset() -> Dataset:
    """
    Load a dataset from a CSV file.

    Returns
    -------
    Dataset
        The dataset.
    """
    if not os.path.exists("evolution/data/price_df.csv"):
        process_json_to_csv("evolution/data/price_dict.json")

    metadata = {
        "features": df.columns.tolist(),
        "target_index": 0,
    }
    df = create_dataset()

    return Dataset(df, np.arange(len(df)), metadata, 0)


bitcoin_dataset = load_dataset()
