import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import pandas as pd
import numpy as np
import holidays
from sklearn.metrics import mean_squared_error, mean_absolute_error
from autots import AutoTS

CSV_PATH = "/home/meow_d/nerd-stuff/2-shared/BDA-assignment/dataset/komuter_combined.csv"
TEMPLATE_PATH = "model_template.csv"
TEST_SPLIT = "2026-06-01"
TRAIN_START = "2025-06-01"

TOP_ROUTES = [
    ("KL Sentral", "Batu Caves"),
]

MODEL_LIST = {
    "SeasonalNaive": 1,
    "ETS": 1,
    "DatepartRegression": 1,
    "GLM": 1,
    "GLS": 1,
    "FFT": 1,
    "BasicLinearModel": 1,
    "AverageValueNaive": 1,
}


def load_routes() -> dict[tuple[str, str], pd.DataFrame]:
    chunks = pd.read_csv(CSV_PATH, parse_dates=["datetime"], chunksize=500_000)
    parts = {r: [] for r in TOP_ROUTES}
    for chunk in chunks:
        for (origin, dest) in TOP_ROUTES:
            sub = chunk[(chunk["origin"] == origin) & (chunk["destination"] == dest)]
            if len(sub) > 0:
                parts[(origin, dest)].append(sub[["datetime", "ridership"]])
    result = {}
    for route, dfs in parts.items():
        if dfs:
            df = pd.concat(dfs).sort_values("datetime").reset_index(drop=True)
            result[route] = df
    return result


def make_regressor(idx: pd.DatetimeIndex) -> pd.DataFrame:
    years = range(idx[0].year, idx[-1].year + 1)
    my_holidays = holidays.Malaysia(years=years)
    reg = pd.DataFrame(index=idx)
    reg["is_holiday"] = reg.index.isin(my_holidays).astype(int)
    reg["dayofweek"] = reg.index.dayofweek
    reg["hour_sin"] = np.sin(2 * np.pi * reg.index.hour / 24)
    reg["hour_cos"] = np.cos(2 * np.pi * reg.index.hour / 24)
    return reg


def prepare_data(df: pd.DataFrame) -> tuple:
    df = df[df["datetime"] >= TRAIN_START].copy()
    df = df.set_index("datetime").asfreq("h")
    df["ridership"] = df["ridership"].interpolate()
    train = df[df.index < TEST_SPLIT].copy()
    test = df[df.index >= TEST_SPLIT].copy()
    reg_train = make_regressor(train.index)
    reg_test = make_regressor(test.index)
    return train, test, reg_train, reg_test


def train():
    route_data = load_routes()
    print(f"Loaded {len(route_data)} routes")

    for (origin, dest) in TOP_ROUTES:
        if (origin, dest) not in route_data:
            continue
        print(f"\n{'='*60}")
        print(f"Route: {origin} -> {dest}")
        print(f"{'='*60}")

        train_df, test_df, reg_train, reg_test = prepare_data(route_data[(origin, dest)])
        print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

        model = AutoTS(
            forecast_length=len(test_df),
            frequency="h",
            ensemble="simple",
            max_generations=1,
            num_validations=0,
            model_list=MODEL_LIST,
            n_jobs=1,
        )

        model.fit(train_df, future_regressor=reg_train)

        model.export_template(TEMPLATE_PATH, models="best", max_per_model_class=1, include_results=True)
        print(f"Template saved to {TEMPLATE_PATH}")

        prediction = model.predict(future_regressor=reg_test)
        forecast = prediction.forecast

        y_true = test_df["ridership"].values
        y_pred = forecast["ridership"].values
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        print(f"Best model: {model.best_model}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")


def predict():
    route_data = load_routes()
    print(f"Loaded {len(route_data)} routes")

    for (origin, dest) in TOP_ROUTES:
        if (origin, dest) not in route_data:
            continue
        print(f"\n{'='*60}")
        print(f"Route: {origin} -> {dest}")
        print(f"{'='*60}")

        train_df, test_df, reg_train, reg_test = prepare_data(route_data[(origin, dest)])
        print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

        model = AutoTS(
            forecast_length=len(test_df),
            frequency="h",
            ensemble="simple",
            max_generations=0,
            num_validations=0,
            model_list=MODEL_LIST,
            n_jobs=1,
        )
        model = model.import_best_model(TEMPLATE_PATH)
        model.fit_data(train_df, future_regressor=reg_train)

        prediction = model.predict(future_regressor=reg_test)
        forecast = prediction.forecast

        y_true = test_df["ridership"].values
        y_pred = forecast["ridership"].values
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")


if __name__ == "__main__":
    if "--predict" in sys.argv:
        predict()
    else:
        train()
