import json
import numpy as np
import pandas as pd

CSV_PATH = "dataset_E408.csv"
WINDOW = 72

# nalaganje
df = pd.read_csv(CSV_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").set_index("datetime")

# zapolni manjkajoce vrednosti in ustvari casovne featureje

if "wind_direction" in df.columns:
    df = df.drop(columns=["wind_direction"])

# stevilski stolpci - interpolacija
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[num_cols] = df[num_cols].interpolate(method="time").ffill().bfill()

# zapolni kategoricne stolpce
df["clouds"] = df["clouds"].fillna("unknown")

# casovni features
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

df = df.drop(columns=["hour", "dayofweek", "month"])


input_cols = [
    "PM2.5", "temperature", "rain", "pressure", "precipitation", "wind_speed",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend",
    "clouds"
]


missing = [c for c in input_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in prepared df: {missing}")

def make_req(start_idx: int):
    chunk = df.iloc[start_idx:start_idx + WINDOW][input_cols].copy()
    if len(chunk) != WINDOW:
        raise ValueError("Not enough rows for this start_idx")
    return {"data": chunk.to_dict(orient="records")}

req1 = make_req(0)
req2 = make_req(500)
req3 = make_req(1500)

for i, req in enumerate([req1, req2, req3], start=1):
    with open(f"request_{i}.json", "w", encoding="utf-8") as f:
        json.dump(req, f, ensure_ascii=False, indent=2)

print("Created request_1.json, request_2.json, request_3.json")

