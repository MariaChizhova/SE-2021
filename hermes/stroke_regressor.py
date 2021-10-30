"""Hermes stroke regressor tools"""

from typing import AnyStr
import pandas as pd  # type: ignore
import numpy as np

import sklearn.preprocessing


def read_data(file: AnyStr) -> pd.DataFrame:
    """Reads data from csv """
    return pd.read_csv(file)


def one_hot(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = dataframe.select_dtypes(include="object")
    for c in columns:
        values = dataframe[c].values
        uniques = np.unique(values)
        for u in uniques:
            new_column = values == u
            dataframe[f'{c}_{u}'] = new_column
        dataframe.drop([c], axis=1, inplace=True)
    return dataframe


def normalize(dataframe: pd.DataFrame, scale_type='minmax') -> pd.DataFrame:
    if scale_type == 'minmax':
        scaler = sklearn.preprocessing.MinMaxScaler()
    elif scale_type == 'standard':
        scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(dataframe)
    return scaler.transform(dataframe)


def statistics(dataframe: pd.DataFrame, stats_type='mean_age', col='ever_married', target='Yes', opposite_target='No'):
    if stats_type == 'mean_age':
        return dataframe[dataframe[col] == target]['age'].mean()
    if stats_type == 'stroke':
        sum_target = dataframe[dataframe[col] == target][stats_type].sum()
        cnt_target = (dataframe[col] == target).sum()
        sum_opposite_target = dataframe[dataframe[col] == opposite_target][stats_type].sum()
        cnt_opposite_target = (dataframe[col] == opposite_target).sum()
        return target if (sum_target / cnt_target) > (sum_opposite_target / cnt_opposite_target) else opposite_target
