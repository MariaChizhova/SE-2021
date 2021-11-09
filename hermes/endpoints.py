from fastapi import FastAPI
import hermes.stroke_regressor

app = FastAPI()
file_loc = 'data/healthcare-dataset-stroke-data.csv'
data = hermes.stroke_regressor.read_data(file_loc)


@app.get("/statistics/mean_age/{col}/{target}")
def get_mean_age_statistics(col: str, target: str):
    return hermes.stroke_regressor.statistics(data, stats_type='mean_age', col=col, target=target)


@app.get("/statistics/mean_age_num/hypertension/{target}")
def get_mean_age_hypertension(target: int):
    return hermes.stroke_regressor.statistics(data, stats_type='mean_age', col='hypertension', target=target)


@app.get("/statistics/stroke/{col}/{target}/{opposite_target}")
def get_stroke_stats(col: str, target: str, opposite_target: str):
    return hermes.stroke_regressor.statistics(data, stats_type='stroke', col=col, target=target,
                                              opposite_target=opposite_target)
