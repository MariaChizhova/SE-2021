"""Hermes stroke regressor tools"""

from typing import AnyStr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Perceptron

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


def normalize(dataframe: pd.DataFrame, scale_type='minmax') -> np.ndarray:
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


def add_data(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame) -> pd.DataFrame:
    dataframe1 = pd.concat([dataframe1, dataframe2]).drop_duplicates().reset_index(drop=True)
    return dataframe1


def remove_col(dataframe: pd.DataFrame, col) -> pd.DataFrame:
    dataframe.drop([col], axis=1, inplace=True)
    return dataframe


def create_plot(name: str, data: list, labels: list, title: str, plot_type='bar') -> plt:
    if plot_type == 'bar':
        fig = plt.bar(data[0], data[1])
        plt.title(title)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.savefig('data/' + name + '.png')
        return fig


def linear_regression(X: np.ndarray, y: np.ndarray, return_value: str = None):
    reg = LinearRegression().fit(X, y)
    if return_value == 'coef':
        return reg.coef_
    elif return_value == 'intercept':
        return reg.intercept_
    else:
        return reg


def perceptron(X: np.ndarray, y: np.ndarray, return_value: str = None):
    reg = Perceptron().fit(X, y)
    if return_value == 'coef':
        return reg.coef_
    elif return_value == 'intercept':
        return reg.intercept_
    else:
        return reg


def top_coef(coefs: np.ndarray, parameters: np.ndarray) -> str:
    return parameters[abs(coefs).argmax()]


def bottom_coef(coefs: np.ndarray, parameters: np.ndarray) -> str:
    print(parameters)
    print(coefs)
    return parameters[abs(coefs).argmin()]
