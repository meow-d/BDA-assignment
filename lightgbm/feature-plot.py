import pandas as pd
import matplotlib.pyplot as plt

importance = {
    "destination":         59,
    "hour":                45,
    "hour_sin":            34,
    "route_mean":          26,
    "day":                 25,
    "hour_cos":            24,
    "lag_1h":              23,
    "rolling_mean_168":    10,
    "rolling_mean_3":       9,
    "route_std":            8,
    "origin":               7,
    "rolling_mean_336":     7,
    "origin_mean":          6,
    "rolling_mean_24":      4,
    "rolling_mean_672":     3,
    "is_holiday":           3,
    "rolling_std_168":      1,
    "lag_3h":               1,
    "lag_2h":               1,
    "destination_mean":     1,
    "rolling_mean_6":       1,
    "rolling_std_336":      1,
    "destination_std":      1,
}

df = (
    pd.DataFrame(
        importance.items(),
        columns=["Feature", "Importance"]
    )
    .sort_values("Importance", ascending=True)
)

plt.figure(figsize=(8, 10), dpi=150)
plt.barh(df["Feature"], df["Importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance Ranking")
plt.tight_layout()
plt.show()
