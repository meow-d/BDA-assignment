import pandas as pd
import numpy as np
import holidays
import pickle
import sys
import os
import matplotlib.pyplot as plt
import optuna
from typing import Any
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
from lightgbm import Booster

MODE = sys.argv[1] if len(sys.argv) > 1 else "train"

CAT_FEATURES = ["origin", "destination"]
NUM_FEATURES = [
    "hour", "day", "is_holiday",
    "hour_sin", "hour_cos",
    "lag_1h", "lag_2h", "lag_3h",
    "rolling_mean_3", "rolling_mean_6", "rolling_mean_24", "rolling_mean_168", "rolling_mean_336", "rolling_mean_672",
    "rolling_std_168", "rolling_std_336",
    "origin_mean", "destination_mean", "destination_std", "route_mean", "route_std",
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["origin", "destination", "datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    years: list[int] = list(range(int(df["datetime"].dt.year.to_numpy().min()), int(df["datetime"].dt.year.to_numpy().max()) + 1))
    my_holidays = holidays.Malaysia(years=years)
    df["is_holiday"] = df["datetime"].dt.date.isin(my_holidays).astype(int)
    for lag, name in [(1, "lag_1h"), (2, "lag_2h"), (3, "lag_3h")]:
        df[name] = df.groupby(["origin", "destination"])["ridership"].shift(lag)
    for window in [3, 6, 24, 168, 336, 672]:
        df[f"rolling_mean_{window}"] = df.groupby(["origin", "destination"])["ridership"].transform(
            lambda x: x.shift(1).rolling(window).mean()
        )
    for window in [168, 336]:
        df[f"rolling_std_{window}"] = df.groupby(["origin", "destination"])["ridership"].transform(
            lambda x: x.shift(1).rolling(window).std()
        )
    df = df.dropna()
    return df


def target_encode(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for key, cols in [("origin", ["origin"]), ("destination", ["destination"]), ("route", ["origin", "destination"])]:
        stats = pd.DataFrame(train_df.groupby(cols)["ridership"].agg(["mean", "std"]))
        stats.columns = [f"{key}_mean", f"{key}_std"]
        train_df = train_df.merge(stats, on=cols, how="left")
        test_df = test_df.merge(stats, on=cols, how="left")
    return train_df, test_df


def make_X(chunk: pd.DataFrame, encoder: OrdinalEncoder) -> pd.DataFrame:
    X_cat = encoder.transform(chunk[CAT_FEATURES])
    return pd.DataFrame(X_cat, columns=pd.Index(CAT_FEATURES), index=chunk.index).join(chunk[NUM_FEATURES])


def train_model(train_df: pd.DataFrame, encoder: OrdinalEncoder) -> Booster:
    # optimum hyperparameters found using optuna
    params: dict[str, Any] = {
        "random_state": 42,
        "verbose": 1,
        "force_col_wise": True,
        "n_estimators": 1,
        "boosting_type": "gbdt",
        "num_leaves": 94,
        "learning_rate": 0.12338927224763702,
        "min_child_samples": 29,
        "subsample": 0.7526667257286987,
        "subsample_freq": 1,
        "colsample_bytree": 0.9413895414693264,
        "objective": "quantile",
        "alpha": 0.657140860312094,
    }
    X = make_X(train_df, encoder)
    y = np.log1p(train_df["ridership"])
    train_set = lgb.Dataset(X, label=y, categorical_feature=CAT_FEATURES)
    model = lgb.train(params, train_set)

    model.save_model("model.txt")
    with open("encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    importance = pd.Series(model.feature_importance(), index=CAT_FEATURES + NUM_FEATURES).sort_values(ascending=False)
    print("\nFeature importance:")
    print(importance)

    return model


def evaluate(model: Booster, test_df: pd.DataFrame, encoder: OrdinalEncoder) -> None:
    X = make_X(test_df, encoder)
    pred = np.expm1(np.asarray(model.predict(X)))
    y_true = test_df["ridership"]
    print("RMSE:", np.sqrt(mean_squared_error(y_true, pred)))
    print("MAE:", mean_absolute_error(y_true, pred))
    print("R²:", r2_score(y_true, pred))


def visualize(model: Booster, test_df: pd.DataFrame, encoder: OrdinalEncoder) -> None:
    X = make_X(test_df, encoder)
    pred = np.expm1(np.asarray(model.predict(X)))
    results = pd.DataFrame({"datetime": test_df["datetime"], "actual": test_df["ridership"], "predicted": pred})
    agg = results.groupby("datetime").agg({"actual": "sum", "predicted": "sum"}).reset_index()
    plt.figure(figsize=(14, 5))
    plt.plot(agg["datetime"], agg["actual"], label="Actual", alpha=0.7)
    plt.plot(agg["datetime"], agg["predicted"], label="Predicted", alpha=0.7)
    plt.xlabel("Datetime")
    plt.ylabel("Total Ridership")
    plt.title("Ridership: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.show()


def tune(train_df: pd.DataFrame, encoder: OrdinalEncoder) -> Booster:
    # sample = train_df.sample(n=100000, random_state=42)
    sample = train_df
    split_idx = int(len(sample) * 0.8)
    train_sample = sample.iloc[:split_idx]
    val_sample = sample.iloc[split_idx:]
    X_train = make_X(train_sample, encoder)
    y_train = np.log1p(train_sample["ridership"])
    X_val = make_X(val_sample, encoder)
    y_val = val_sample["ridership"]

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_FEATURES, free_raw_data=False)

    def objective(trial: optuna.Trial) -> float:
        boosting = trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"])
        obj = trial.suggest_categorical("objective", ["regression", "regression_l1", "quantile", "huber"])
        params: dict[str, Any] = {
            "random_state": 42,
            "verbose": -1,
            "force_col_wise": True,
            "n_estimators": 1,
            # "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "boosting_type": boosting,
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": 1.0 if boosting == "goss" else trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "objective": obj,
        }
        if obj == "quantile":
            params["alpha"] = trial.suggest_float("alpha", 0.1, 0.9)
        elif obj == "huber":
            params["alpha"] = trial.suggest_float("alpha", 0.1, 10.0)
        model = lgb.train(params, train_set)
        pred = np.expm1(np.asarray(model.predict(X_val)))
        return float(np.sqrt(mean_squared_error(y_val, pred)))

    study = optuna.create_study(
        direction="minimize",
        study_name="komuter",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=30)
    print("Best params:")
    print(study.best_params)
    print("Best RMSE:", study.best_value)

    return study.user_attrs["best_booster"]

def main():
    cache_path = "../dataset/komuter_lightgbm_preprocessed.csv"
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["datetime"])
    else:
        df = preprocess(pd.read_csv("../dataset/komuter_combined.csv", parse_dates=["datetime"]))
        df.to_csv(cache_path, index=False)

    split_date = "2026-06-01"
    train_df = pd.DataFrame(df[df["datetime"] < split_date])
    test_df = pd.DataFrame(df[df["datetime"] >= split_date])

    train_df, test_df = target_encode(train_df, test_df)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(train_df[CAT_FEATURES])

    if MODE in ("predict", "visualize"):
        model = Booster(model_file="model.txt")
    elif MODE == "tune":
        tune(train_df, encoder)
        return
    else:
        model = train_model(train_df, encoder)

    if MODE == "visualize":
        visualize(model, test_df, encoder)
    else:
        evaluate(model, test_df, encoder)


if __name__ == "__main__":
    main()
