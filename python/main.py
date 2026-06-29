import pandas as pd
import numpy as np
import holidays
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor, Booster

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
CHUNKSIZE = 500000


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


def train_model(train_df: pd.DataFrame, encoder: OrdinalEncoder) -> LGBMRegressor:
    model = LGBMRegressor(random_state=42, verbose=-1, n_estimators=1, objective="quantile", alpha=0.8)
    for i in range(0, len(train_df), CHUNKSIZE):
        chunk = train_df.iloc[i:i + CHUNKSIZE]
        X = make_X(chunk, encoder)
        y = np.log1p(chunk["ridership"])
        model.fit(X, y, categorical_feature=CAT_FEATURES, init_model=model if i > 0 else None)

    model.booster_.save_model("model.txt")
    with open("encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    importance = pd.Series(model.feature_importances_, index=CAT_FEATURES + NUM_FEATURES).sort_values(ascending=False)
    print("\nFeature importance:")
    print(importance)

    return model


def evaluate(model: LGBMRegressor | Booster, test_df: pd.DataFrame, encoder: OrdinalEncoder) -> None:
    y_true = []
    y_pred = []
    for i in range(0, len(test_df), CHUNKSIZE):
        chunk = test_df.iloc[i:i + CHUNKSIZE]
        X = make_X(chunk, encoder)
        pred = np.expm1(np.asarray(model.predict(X)))
        y_true.extend(chunk["ridership"])
        y_pred.extend(pred)
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("R²:", r2_score(y_true, y_pred))


def visualize(model: LGBMRegressor | Booster, test_df: pd.DataFrame, encoder: OrdinalEncoder) -> None:
    y_true = []
    y_pred = []
    datetimes = []
    for i in range(0, len(test_df), CHUNKSIZE):
        chunk = test_df.iloc[i:i + CHUNKSIZE]
        X = make_X(chunk, encoder)
        pred = np.expm1(np.asarray(model.predict(X)))
        y_true.extend(chunk["ridership"])
        y_pred.extend(pred)
        datetimes.extend(chunk["datetime"])
    results = pd.DataFrame({"datetime": datetimes, "actual": y_true, "predicted": y_pred})
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


def main():
    df = preprocess(pd.read_csv("../dataset/komuter_combined.csv", parse_dates=["datetime"]))

    split_date = "2026-06-01"
    train_df = pd.DataFrame(df[df["datetime"] < split_date])
    test_df = pd.DataFrame(df[df["datetime"] >= split_date])

    train_df, test_df = target_encode(train_df, test_df)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(train_df[CAT_FEATURES])

    if MODE in ("predict", "visualize"):
        model = Booster(model_file="model.txt")
    else:
        model = train_model(train_df, encoder)

    if MODE == "visualize":
        visualize(model, test_df, encoder)
    else:
        evaluate(model, test_df, encoder)


if __name__ == "__main__":
    main()
