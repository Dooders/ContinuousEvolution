import os

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from evolution.data.data import Dataset


def process_bitcoin_csv() -> None:
    """ """

    df = pd.read_csv("evolution/data/sample_bitcoin.csv")

    # Select only numeric columns for scaling
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Scale only the numeric columns
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])

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
    return df


def load_dataset() -> Dataset:
    """
    Load a dataset from a CSV file.

    Returns
    -------
    Dataset
        The dataset.
    """
    if not os.path.exists("evolution/data/price_df.csv"):
        process_bitcoin_csv()

    df = create_dataset()

    indices = df["Timestamp"].to_numpy()
    final_df = df.drop(columns=["Timestamp"])
    metadata = {
        "features": final_df.columns.tolist(),
    }
    return Dataset(final_df.to_numpy(), indices, metadata, target_index=0)


bitcoin_dataset = load_dataset()
