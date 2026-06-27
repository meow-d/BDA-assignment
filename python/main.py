import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["origin", "destination", "datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["week"] = df["datetime"].dt.isocalendar().week.astype(int)
    df["year"] = df["datetime"].dt.year
    df["day_of_month"] = df["datetime"].dt.day
    df["weekend"] = (df["day"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 7)
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)
    df["dom_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
    df["dom_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)
    years: list[int] = list(range(int(df["year"].to_numpy().min()), int(df["year"].to_numpy().max()) + 1))
    my_holidays = holidays.Malaysia(years=years)
    df["is_holiday"] = df["datetime"].dt.date.isin(my_holidays).astype(int)
    for lag, name in [(1, "lag_1h"), (2, "lag_2h"), (3, "lag_3h"), (24, "lag_24h"), (48, "lag_48h"), (72, "lag_72h"), (168, "lag_1w"), (336, "lag_2w"), (504, "lag_3w"), (672, "lag_4w")]:
        df[name] = df.groupby(["origin", "destination"])["ridership"].shift(lag)
    for window in [3, 6, 24, 168, 336, 672]:
        df[f"rolling_mean_{window}"] = df.groupby(["origin", "destination"])["ridership"].transform(
            lambda x: x.shift(1).rolling(window).mean()
        )
    for window in [24, 168, 336]:
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
    "hour", "day", "month", "week", "year", "day_of_month", "weekend", "is_holiday",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "week_sin", "week_cos", "dom_sin", "dom_cos",
    "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_48h", "lag_72h",
    "lag_1w", "lag_2w", "lag_3w", "lag_4w",
    "rolling_mean_3", "rolling_mean_6", "rolling_mean_24", "rolling_mean_168", "rolling_mean_336", "rolling_mean_672",
    "rolling_std_24", "rolling_std_168", "rolling_std_336",
    "origin_mean", "origin_std", "destination_mean", "destination_std", "route_mean", "route_std",
]

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
encoder.fit(train_df[cat_features])

chunksize = 500000
model = LGBMRegressor(random_state=42, verbose=-1, n_estimators=1)

for i in range(0, len(train_df), chunksize):
    chunk = train_df.iloc[i:i + chunksize]
    X_cat = encoder.transform(chunk[cat_features])
    X = pd.DataFrame(X_cat, columns=pd.Index(cat_features), index=chunk.index).join(chunk[num_features])
    y = np.log1p(chunk["ridership"])
    model.fit(X, y, categorical_feature=cat_features, init_model=model if i > 0 else None)

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
