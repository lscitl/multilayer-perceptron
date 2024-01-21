#!/usr/bin/python3

import sys
import shutil
import pandas as pd
import numpy as np
from load_csv import load


def split_dataset(data: np.ndarray | pd.DataFrame | pd.Series, split_ratio: float):
    """
    split dataset with split ratio.
    input data should be an instance of np.ndarray, pd.DataFrame, or pd.Series.
    if data length is 100 and split_ratio is 0.2, return data length will be 80 and 20.
    if one of return data length is 0, it will raise assertion error.
    """

    assert (
        isinstance(data, np.ndarray)
        or isinstance(data, pd.DataFrame)
        or isinstance(data, pd.Series)
    ), "not supported."

    m = len(data)
    split_num = int(m * split_ratio)

    assert split_num not in (0, m), "one of return dataset length is 0."

    return data.iloc[: m - split_num, :], data.iloc[m - split_num :, :]


if __name__ == "__main__":
    try:
        print("Data file path or name: ", end="", flush=True)
        path = sys.stdin.readline().strip("\n")

        data: pd.DataFrame = load(path, None)

        term_len = shutil.get_terminal_size()

        assert data is not None, "data load failure."

        while True:
            try:
                print("Data ratio: ", end="", flush=True)
                split_ratio = float(sys.stdin.readline().strip("\n"))
                break
            except ValueError:
                print("Invalid input. Try again")
            except:
                msg = "Interrupted."
                print(f"\r{msg:{term_len.columns}}")
                quit()

        while True:
            try:
                print("File name for saving splitted data: ", end="", flush=True)
                file_name = sys.stdin.readline().strip("\n")
                if len(file_name) == 0:
                    raise ValueError
                break
            except ValueError:
                print("Invalid input. Try again")
            except:
                msg = "Interrupted."
                print(f"\r{msg:{term_len.columns}}")
                quit()

        train, split = split_dataset(data, split_ratio)
        print(
            f"Data set will split to train set: {len(train)}, splitted data set: {len(split)}"
        )

        train.to_csv("train.csv", header=None, index=None)
        split.to_csv(f"{file_name}.csv", header=None, index=None)

    except Exception as e:
        print(f"{e.__class__.__name__}: {e}", file=sys.stderr)
