"""Test of hermes"""

from pathlib import Path
import string
import random
import numpy as np
import pandas as pd

import hermes.stroke_regressor

random.seed(0)


def test_read_data():
    """Simple test for reading data csv -> pandas Dataframe"""
    file_loc = '/tmp/simple.csv'
    with open(file_loc, 'w', encoding='utf-8') as f:
        f.write('Name,Number,Double\nhalp me,7,22.8\nplease,69,13.37\n')
    simple = hermes.stroke_regressor.read_data(file_loc)
    assert simple.shape == (2, 3)
    assert simple.equals(pd.DataFrame({'Name': ['halp me', 'please'],
                                       'Number': [7, 69],
                                       'Double': [22.8, 13.37]}))


def test_read_data_2():
    name = np.random.uniform(0, 1000, 100)
    for i in range(100):
        file_loc = f'/tmp/test{name[i]}.csv'
        data = np.random.uniform(0, 1000, (100, 3))
        df = pd.DataFrame(data=data, columns=['Name', 'Number', 'Double'])
        df.to_csv(file_loc, index=False)
        simple = hermes.stroke_regressor.read_data(file_loc)
        assert simple.shape == (100, 3)
        assert np.allclose(df.to_numpy(), simple.to_numpy())


def get_random_string(n):
    """Return random string of upper case
    latin letters and numbers, of length n"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))


def test_one_hot_encoded():
    """Test one hot encoding: if a feature
    contains a set of values, converts it
    into len(set) with 0 and 1, pointing
    if this entry has certain value from set or not"""
    leng = np.reshape(list(map(get_random_string, range(1, 3 * 4 + 1))), (4, 3))
    mx = np.array(np.random.rand(4, 3) * 3, dtype=np.int64)
    data = []
    ans = np.zeros((4, 9), dtype=np.int64)
    for (i, row) in enumerate(mx):
        data.append([])
        for (j, val) in enumerate(row):
            data[-1].append(leng[j][val])
            ans[i][3 * j + val] = 1
    cols = []
    for (i, name) in enumerate(leng[3]):
        cols.append(f"{name}_{leng[i][0]}")
        cols.append(f"{name}_{leng[i][1]}")
        cols.append(f"{name}_{leng[i][2]}")
    df_q = pd.DataFrame(data=data, columns=leng[3])
    df_ans = pd.DataFrame(data=ans, columns=cols)
    df_ans = df_ans.loc[:, (df_ans != 0).any(axis=0)]
    df_check = hermes.stroke_regressor.one_hot(df_q)
    df_ans = df_ans.reindex(sorted(df_ans.columns), axis=1)
    df_check = df_check.reindex(sorted(df_check.columns), axis=1)
    assert np.allclose(df_ans.to_numpy(), df_check.to_numpy())


def test_normalize_minmax():
    data = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
    ans = (data - np.nanmin(data, axis=0)) / (np.nanmax(data, axis=0) - np.nanmin(data, axis=0))
    check = hermes.stroke_regressor.normalize(pd.DataFrame(data), scale_type='minmax')
    assert np.allclose(ans, check)


def test_normalize_standard():
    data = np.random.rand(7, 12)
    ans = (data - data.mean(axis=0)) / data.std(axis=0)
    check = hermes.stroke_regressor.normalize(data, scale_type='standard')
    assert np.allclose(ans, check)


def test_statistics_mean_age():
    file_loc = 'data/healthcare-dataset-stroke-data.csv'
    data = hermes.stroke_regressor.read_data(file_loc)
    total = sum(data['age'])
    hypertension_age = hermes.stroke_regressor.statistics(data, stats_type='mean_age', col='hypertension', target=1)
    not_hypertension_age = hermes.stroke_regressor.statistics(data, stats_type='mean_age', col='hypertension', target=0)
    result = len(data[data['hypertension'] == 1]) * hypertension_age + len(
        data[data['hypertension'] == 0]) * not_hypertension_age
    assert abs(result - total) < 0.0001
    male_age = hermes.stroke_regressor.statistics(data, stats_type='mean_age', col='gender', target='Male')
    female_age = hermes.stroke_regressor.statistics(data, stats_type='mean_age', col='gender', target='Female')
    other_age = hermes.stroke_regressor.statistics(data, stats_type='mean_age', col='gender', target='Other')
    result = len(data[data['gender'] == 'Male']) * male_age + len(data[data['gender'] == 'Female']) * female_age + len(
        data[data['gender'] == 'Other']) * other_age
    assert abs(result - total) < 0.0001
    ever_married_age = hermes.stroke_regressor.statistics(data, stats_type='mean_age', col='ever_married', target='Yes')
    never_married_age = hermes.stroke_regressor.statistics(data, stats_type='mean_age', col='ever_married', target='No')
    result = len(data[data['ever_married'] == 'Yes']) * ever_married_age + len(
        data[data['ever_married'] == 'No']) * never_married_age
    assert abs(result - total) < 0.0001


def test_statistics_stroke():
    file_loc = 'data/healthcare-dataset-stroke-data.csv'
    data = hermes.stroke_regressor.read_data(file_loc)
    gender = hermes.stroke_regressor.statistics(data, stats_type='stroke', col='gender', target='Male',
                                                opposite_target='Female')
    assert gender == 'Male'
    residence_type = hermes.stroke_regressor.statistics(data, stats_type='stroke', col='Residence_type', target='Urban',
                                                        opposite_target='Rural')
    assert residence_type == 'Urban'
    ever_married = hermes.stroke_regressor.statistics(data, stats_type='stroke', col='ever_married', target='Yes',
                                                      opposite_target='No')
    assert ever_married == 'Yes'
    hypertension = hermes.stroke_regressor.statistics(data, stats_type='stroke', col='hypertension', target=1,
                                                      opposite_target=0)
    assert hypertension == 1
    heart_disease = hermes.stroke_regressor.statistics(data, stats_type='stroke', col='heart_disease', target=1,
                                                       opposite_target=0)
    assert heart_disease == 1


def test_add_data():
    file_loc = 'data/healthcare-dataset-stroke-data.csv'
    data1 = hermes.stroke_regressor.read_data(file_loc)
    cols = ['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
            'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']
    person1 = [9999, 'Male', 38, 0, 0, 'No', 'Self-employed', 'Urba', 82.28, 24, 'formerly smoked', 1]
    person2 = [12095, 'Female', 61, 0, 1, 'Yes', 'Govt_job', 'Rural', 120.46, 36.8, 'smokes', 1]
    data2 = pd.DataFrame([person1, person2], columns=cols)
    result = hermes.stroke_regressor.add_data(data1, data2)
    assert result.shape == (data1.shape[0] + 1, data1.shape[1])
    assert len(result[result['id'] == 9999]) == 1
    assert len(result[result['id'] == 12095]) == 1


def test_remove_col():
    file_loc = 'data/healthcare-dataset-stroke-data.csv'
    data = hermes.stroke_regressor.read_data(file_loc)
    columns = data.columns
    for col in columns:
        new_data = data.drop([col], axis=1)
        check_data = hermes.stroke_regressor.remove_col(dataframe=data, col=col)
        assert check_data.equals(new_data)


def test_create_bar_plot():
    gender = ['Male', 'Female', 'Other']
    total_per_gender = [100, 120, 50]
    name = "bar_chart"
    labels = ["gender", "count"]
    title = "total per gender"
    plot = hermes.stroke_regressor.create_plot(name=name, data=[gender, total_per_gender], labels=labels, title=title,
                                               plot_type='bar')
    result = []
    for rect in plot:
        height = rect.get_height()
        result.append(height)
    assert Path('data/' + name + '.png').exists()
    assert result == [100, 120, 50]


def test_linear_regression():
    file_loc = 'data/healthcare-dataset-stroke-data.csv'
    data = hermes.stroke_regressor.read_data(file_loc)
    data = hermes.stroke_regressor.one_hot(data)
    data.fillna(data['bmi'].mean(), inplace=True)  # pylint: disable=unsubscriptable-object
    a = data.astype(np.float64).to_numpy()
    coef = hermes.stroke_regressor.linear_regression(a[1:, 1:-1], a[1:, -1], 'coef')
    assert np.allclose(coef, np.array([-1.1574462149573293e-18, 3.419088876507203e-17, -3.9205165534584025e-16,
                                       -1.4155062715314148e-17, 3.927994092015258e-17, -9.523883523664444e-16,
                                       -5.683557029752764e-16, -7.00727650591634e-16, 9.977235206953956e-16,
                                       -2.080232712684715e-16, -2.7369826322273065e-16, 1.5374108564703447e-16,
                                       1.4253092326172553e-17, -2.815903077890237e-17, 3.7470388052664107e-16,
                                       -3.821081183930575e-16, 4.1102650480927346e-17, -7.306152541490587e-18,
                                       -0.9999999999999997, -1.0000000000000002, -1.0000000000000004]))


def test_top_coef():
    file_loc = 'data/healthcare-dataset-stroke-data.csv'
    data = hermes.stroke_regressor.read_data(file_loc)
    data = hermes.stroke_regressor.one_hot(data)
    parameters = data.columns
    data.fillna(data['bmi'].mean(), inplace=True)
    a = data.astype(np.float64).to_numpy()
    coef = hermes.stroke_regressor.linear_regression(a[1:, 1:-1], a[1:, -1], 'coef')
    ans = hermes.stroke_regressor.top_coef(coef, parameters[1:])
    assert ans == 'smoking_status_never smoked'


def test_bottom_coef():
    file_loc = 'data/healthcare-dataset-stroke-data.csv'
    data = hermes.stroke_regressor.read_data(file_loc)
    data = hermes.stroke_regressor.one_hot(data)
    parameters = data.columns
    data.fillna(data['bmi'].mean(), inplace=True)
    a = data.astype(np.float64).to_numpy()
    coef = hermes.stroke_regressor.linear_regression(a[1:, 1:-1], a[1:, -1], 'coef')
    ans = hermes.stroke_regressor.bottom_coef(coef, parameters[1:])
    assert ans == 'avg_glucose_level'


def test_intercept():
    file_loc = 'data/healthcare-dataset-stroke-data.csv'
    data = hermes.stroke_regressor.read_data(file_loc)
    data = hermes.stroke_regressor.one_hot(data)
    data.fillna(data['bmi'].mean(), inplace=True)  # pylint: disable=unsubscriptable-object
    a = data.astype(np.float64).to_numpy()
    intercept = hermes.stroke_regressor.linear_regression(a[1:, 1:-1], a[1:, -1], 'intercept')
    assert abs(intercept - 1) < 0.0001


def test_score():
    file_loc = 'data/healthcare-dataset-stroke-data.csv'
    data = hermes.stroke_regressor.read_data(file_loc)
    data = hermes.stroke_regressor.one_hot(data)
    data.fillna(data['bmi'].mean(), inplace=True)  # pylint: disable=unsubscriptable-object
    a = data.astype(np.float64).to_numpy()
    X = a[1:, 1:-1]
    y = a[1:, -1]
    score = hermes.stroke_regressor.linear_regression(X, y, 'score')
    coef = hermes.stroke_regressor.linear_regression(X, y, 'coef')
    intercept = hermes.stroke_regressor.linear_regression(X, y, 'intercept')
    total = np.sum(y - np.mean(y) ** 2)
    residual = np.sum((y - (X @ coef + intercept)) ** 2)
    assert score == 1 - residual / total
