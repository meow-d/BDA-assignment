import pandas as pd
import numpy as np
import holidays
import pickle
import sys
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor, Booster

PREDICT = len(sys.argv) > 1 and sys.argv[1] == "predict"

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

df = preprocess(pd.read_csv("../dataset/komuter_combined.csv", parse_dates=["datetime"]))

split_date = "2026-06-01"
train_df = df[df["datetime"] < split_date]
test_df = df[df["datetime"] >= split_date]

for key, cols in [("origin", ["origin"]), ("destination", ["destination"]), ("route", ["origin", "destination"])]:
    stats = pd.DataFrame(train_df.groupby(cols)["ridership"].agg(["mean", "std"]))
    stats.columns = [f"{key}_mean", f"{key}_std"]
    train_df = train_df.merge(stats, on=cols, how="left")
    test_df = test_df.merge(stats, on=cols, how="left")

cat_features = ["origin", "destination"]
num_features = [
    "hour", "day", "is_holiday",
    "hour_sin", "hour_cos",
    "lag_1h", "lag_2h", "lag_3h",
    "rolling_mean_3", "rolling_mean_6", "rolling_mean_24", "rolling_mean_168", "rolling_mean_336", "rolling_mean_672",
    "rolling_std_168", "rolling_std_336",
    "origin_mean", "destination_mean", "destination_std", "route_mean", "route_std",
]

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
encoder.fit(train_df[cat_features])

chunksize = 500000

if PREDICT:
    model = Booster(model_file="model.txt")
else:
    model = LGBMRegressor(random_state=42, verbose=-1, n_estimators=1)
    for i in range(0, len(train_df), chunksize):
        chunk = train_df.iloc[i:i + chunksize]
        X_cat = encoder.transform(chunk[cat_features])
        X = pd.DataFrame(X_cat, columns=pd.Index(cat_features), index=chunk.index).join(chunk[num_features])
        y = np.log1p(chunk["ridership"])
        model.fit(X, y, categorical_feature=cat_features, init_model=model if i > 0 else None)
    model.booster_.save_model("model.txt")
    with open("encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    importance = pd.Series(model.feature_importances_, index=cat_features + num_features).sort_values(ascending=False)
    print("\nFeature importance:")
    print(importance)

y_true = []
y_pred = []

for i in range(0, len(test_df), chunksize):
    chunk = test_df.iloc[i:i + chunksize]
    X_cat = encoder.transform(chunk[cat_features])
    X = pd.DataFrame(X_cat, columns=pd.Index(cat_features), index=chunk.index).join(chunk[num_features])
    pred = np.expm1(np.asarray(model.predict(X)))
    y_true.extend(chunk["ridership"])
    y_pred.extend(pred)

print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
print("MAE:", mean_absolute_error(y_true, y_pred))
print("R²:", r2_score(y_true, y_pred))
