import pytest
import hermes.stroke_regressor
import pandas as pd
import numpy as np


def test_read_data():
    file_loc = '/tmp/simple.csv'
    with open(file_loc, 'w') as f:
        f.write('Name, Number, Double\nhalp me, 7, 22.8\nplease, 69, 13.37\n')
    simple = hermes.stroke_regressor.read_data(file_loc)
    assert simple.shape == (2, 3)
    assert simple.equals(pd.DataFrame({'Name': ['halp me', 'please'], 'Number': [7, 69], 'Double': [22.8, 13.37]}))


def test_read_data_2():
    name = np.random.uniform(0, 1000, 100)
    for i in range(100):
        file_loc = f'/tmp/test{name[i]}.csv'
        data = np.random.uniform(0, 1000, (100, 3))
        df = pd.DataFrame(data=data, columns=['Name', 'Number', 'Double'])
        df.to_csv(file_loc)
        simple = hermes.stroke_regressor.read_data(file_loc)
        assert simple.shape == (100, 3)
        assert simple.equals(df)
        