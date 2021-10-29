import pandas as pd  # type: ignore


def read_data(file):
    return pd.read_csv(file)