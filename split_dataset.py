#!/usr/bin/python3

import sys
import os
import pandas as pd
import numpy as np
from load_csv import load


def split_dataset(data: np.ndarray | pd.DataFrame | pd.Series, split_ratio: float):
    """
    split dataset with split ratio.
    input data should be an instance of np.ndarray, pd.DataFrame, or pd.Series.
    if data length is 100 and split_ratio is 0.2, return data length will be 80 and 20.
    if one of return data length is 0, it will throw exception.
    """

    assert isinstance(data, np.ndarray) or isinstance(pd.DataFrame) or isinstance(pd.Series), "not supported."
    m = len(data)
    split_num = int(m * split_ratio)
    if split_num == 0 or split_num == m:
        raise AssertionError("one of return dataset length is 0.")
    return data[: m - split_num], data[m - split_num:]


if __name__ == "__main__":

    try:

        path = sys.stdin.readline("Data file path or name: ")

        data: pd.DataFrame = load(path)

        assert data is not None, "data load failure."

        while True:
            try:
                split_ratio = float(sys.stdin.readline("Data ratio: "))
                break
            except:
                None

        train, valid = split_dataset(data, split_ratio)

        train.to_csv("train.csv")
        valid.to_csv("valid.csv")


    except Exception as e:
        print(f"{e.__class__.__name__}: {e}", file=sys.stderr)