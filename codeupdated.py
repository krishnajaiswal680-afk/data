import os
import glob
import json
import pandas as pd
import numpy as np
import xgboost as xgb

# ==========================
# 1️⃣ Resolve Paths
# ==========================

MODEL_PATH = os.environ.get("MODEL_PATH")
METADATA_PATH = os.environ.get("METADATA_PATH")
CSV_PATH = os.environ.get("PREPARED_CSV_PATH", "/mnt/data/loaded_features.csv")

if not MODEL_PATH:
    candidates = sorted(glob.glob("/mnt/data/xgboost_model*.json"))
    MODEL_PATH = candidates[-1] if candidates else None

if not METADATA_PATH:
    candidates = sorted(glob.glob("/mnt/data/model_metadata*.json"))
    METADATA_PATH = candidates[-1] if candidates else None

if not MODEL_PATH or not os.path.exists(MODEL_PATH):
    raise Exception(json.dumps({
        "error": {"code": "model_load_failed", "message": "model file not found"}
    }))

if not METADATA_PATH or not os.path.exists(METADATA_PATH):
    raise Exception(json.dumps({
        "error": {"code": "model_load_failed", "message": "metadata file not found"}
    }))

if not os.path.exists(CSV_PATH):
    raise Exception(json.dumps({
        "error": {"code": "data_load_failed", "message": "prepared CSV not found"}
    }))

# ==========================
# 2️⃣ Load Metadata
# ==========================

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

FEATURES = metadata.get("features_used")

if not FEATURES or len(FEATURES) != 15:
    raise Exception(json.dumps({
        "error": {
            "code": "feature_list_unavailable",
            "message": "metadata.features_used invalid"
        }
    }))

# ==========================
# 3️⃣ Load Booster Model
# ==========================

booster = xgb.Booster()
booster.load_model(MODEL_PATH)

# ==========================
# 4️⃣ Load Prepared CSV
# ==========================

df = pd.read_csv(CSV_PATH)

STATION = os.environ.get("STATION")
RANK = os.environ.get("RANK")

if not STATION or not RANK:
    raise Exception(json.dumps({
        "error": {
            "code": "missing_filter_inputs",
            "message": "STATION or RANK not provided"
        }
    }))

filtered = df[(df["Station"] == STATION) & (df["Rank"] == RANK)]

if filtered.empty:
    raise Exception(json.dumps({
        "error": {
            "code": "empty_filter_result",
            "message": "no rows after Station/Rank filter"
        }
    }))

# ==========================
# 5️⃣ Select Latest 3 Dates
# ==========================

dates = pd.to_datetime(filtered["Date"], errors="coerce")
filtered = filtered.assign(_parsed_date=dates)

distinct_dates = sorted(filtered["_parsed_date"].dropna().unique())

if len(distinct_dates) < 3:
    raise Exception(json.dumps({
        "error": {
            "code": "fewer_than_3_dates",
            "message": "less than 3 distinct dates available"
        }
    }))

latest_3 = distinct_dates[-3:]
selected = filtered[filtered["_parsed_date"].isin(latest_3)]

# Preserve original order
selected = selected.drop(columns=["_parsed_date"])

# ==========================
# 6️⃣ Strict Feature Matrix
# ==========================

for col in FEATURES:
    if col not in selected.columns:
        raise Exception(json.dumps({
            "error": {
                "code": "missing_features",
                "message": f"missing feature column: {col}"
            }
        }))

if selected[FEATURES].isnull().any().any():
    raise Exception(json.dumps({
        "error": {
            "code": "missing_features",
            "message": "null values in required features"
        }
    }))

X = selected[FEATURES]

dtest = xgb.DMatrix(X, feature_names=FEATURES)

# ==========================
# 7️⃣ Predict
# ==========================

preds = booster.predict(dtest)
preds = np.clip(preds, 0, 1)

# ==========================
# 8️⃣ Output JSON
# ==========================

output = {
    "input": {
        "station": STATION,
        "rank": RANK,
        "dates": [str(pd.to_datetime(d).date()) for d in latest_3],
        "records": len(selected),
        "source": "data_preparation_agent(prepared_csv)"
    },
    "features_used": FEATURES,
    "predictions": []
}

for row, pred in zip(selected.to_dict(orient="records"), preds):
    output["predictions"].append({
        **row,
        "features": {f: row[f] for f in FEATURES},
        "pred_activation_rate": float(pred)
    })

print(json.dumps(output))
