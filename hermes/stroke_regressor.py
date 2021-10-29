import pandas as pd
import numpy as np
from typing import AnyStr

import sklearn.preprocessing


def read_data(file: AnyStr):
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

def normalize(dataframe: pd.DataFrame, type='minmax'):
    if type == 'minmax':
        scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(dataframe)
    return scaler.transform(dataframe)