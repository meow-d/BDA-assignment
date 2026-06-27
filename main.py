import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["origin", "destination", "datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["weekend"] = (df["day"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 7)
    for lag, name in [(1, "lag_1h"), (24, "lag_24h"), (168, "lag_1w")]:
        df[name] = df.groupby(["origin", "destination"])["ridership"].shift(lag)
    for window in [3, 6, 24]:
        df[f"rolling_mean_{window}"] = df.groupby(["origin", "destination"])["ridership"].transform(
            lambda x: x.shift(1).rolling(window).mean()
        )
    df = df.dropna()
    return df

df = preprocess(pd.read_csv("komuter_datetime.csv", parse_dates=["datetime"]))

split_date = "2026-06-01"
train_df = df[df["datetime"] < split_date]
test_df = df[df["datetime"] >= split_date]

cat_features = ["origin", "destination"]
num_features = [
    "hour", "day", "month", "weekend",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "lag_1h", "lag_24h", "lag_1w",
    "rolling_mean_3", "rolling_mean_6", "rolling_mean_24",
]

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_cat = encoder.fit_transform(train_df[cat_features])
X_test_cat = encoder.transform(test_df[cat_features])
cat_cols = encoder.get_feature_names_out(cat_features).tolist()

X_train = pd.DataFrame(X_train_cat, columns=cat_cols, index=train_df.index).join(train_df[num_features])
X_test = pd.DataFrame(X_test_cat, columns=cat_cols, index=test_df.index).join(test_df[num_features])

y_train = np.log1p(train_df["ridership"])
y_test = test_df["ridership"]

model = LGBMRegressor(random_state=42, verbose=-1)
model.fit(X_train, y_train)

pred = np.expm1(np.asarray(model.predict(X_test)))

print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("MAE:", mean_absolute_error(y_test, pred))
print("R²:", r2_score(y_test, pred))
