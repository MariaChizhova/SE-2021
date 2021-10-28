import pandas as pd
import numpy as np


def read_data(file):
    return pd.read_csv(file)


def one_hot(dataframe: pd.DataFrame):
    columns = dataframe.select_dtypes(include="object")
    for c in columns:
        values = dataframe[c].values
        uniques = np.unique(values)
        for u in uniques:
            new_column = values == u
            dataframe[f'{c}_{u}'] = new_column
        dataframe.drop([c], axis=1, inplace=True)
    return dataframe


