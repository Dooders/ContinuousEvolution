import json

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

with open("price_dict.json", "r") as file:
    price_dict = json.load(file)

# sort the dictionary by date then make list of prices
orig_prices = [price_dict[key] for key in sorted(price_dict.keys())]


scaler = StandardScaler()
prices = scaler.fit_transform(np.array(orig_prices).reshape(-1, 1)).flatten()


def create_sequences(data: np.ndarray, N: int, lag: int) -> tuple:
    """
    Create input and target sequences for training a model, with a lag for the target.

    Parameters
    ----------
    data : np.ndarray
        1D array of data.
    N : int
        Number of previous prices to use as input features.
    lag : int
        Number of intervals ahead for the target value.

    Returns
    -------
    tuple
        X : np.ndarray
            3D array of input sequences.
        y : np.ndarray
            1D array of target values.
    """
    X, y = [], []
    # Adjust the range to stop early enough so that there are `lag` future elements available after the last input sequence
    for i in range(len(data) - N - lag + 1):
        X.append(data[i : i + N])  # Input sequence from current index i to i+N
        y.append(
            data[i + N + lag - 1]
        )  # Target value lag steps after the end of the input sequence
    return np.array(X), np.array(y)


# Create input and target sequences
N = 3  # Using the last 3 prices to predict the next one
X, y = create_sequences(prices, N, 1)


# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension
y = torch.tensor(y, dtype=torch.float32)
