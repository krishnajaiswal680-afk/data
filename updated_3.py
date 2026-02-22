import os
import glob
import json
import pandas as pd
import numpy as np
import xgboost as xgb

# ==============================
# 1️⃣ Resolve Model + Metadata
# ==============================

# ---- Find Model File Dynamically ----
model_candidates = sorted(glob.glob("/mnt/data/*xgboost_model*.json"))
if not model_candidates:
    print(json.dumps({
        "error": {
            "code": "model_load_failed",
            "message": "No model file found",
            "details": {"reason": "glob_no_match"}
        }
    }))
    raise SystemExit

MODEL_PATH = model_candidates[-1]

# ---- Find Metadata File Dynamically ----
metadata_candidates = sorted(glob.glob("/mnt/data/*model_metadata*.json"))
if not metadata_candidates:
    print(json.dumps({
        "error": {
            "code": "feature_list_unavailable",
            "message": "Metadata file not found"
        }
    }))
    raise SystemExit

METADATA_PATH = metadata_candidates[-1]

# ---- Load Metadata ----
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

FEATURES = metadata.get("features_used")

if not FEATURES or len(FEATURES) != 15:
    print(json.dumps({
        "error": {
            "code": "feature_list_unavailable",
            "message": "metadata.features_used invalid or not length 15"
        }
    }))
    raise SystemExit

# ---- Load Booster Model ----
try:
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
except Exception as e:
    print(json.dumps({
        "error": {
            "code": "model_load_failed",
            "message": "Failed to load model",
            "details": {"reason": str(e)}
        }
    }))
    raise SystemExit


# ==============================
# 2️⃣ Load Prepared CSV
# ==============================

CSV_PATH = "/mnt/data/loaded_features.csv"

if not os.path.exists(CSV_PATH):
    print(json.dumps({
        "error": {
            "code": "data_load_failed",
            "message": "Prepared CSV not found"
        }
    }))
    raise SystemExit

df = pd.read_csv(CSV_PATH)

STATION = os.environ.get("STATION")
RANK = os.environ.get("RANK")

if not STATION or not RANK:
    print(json.dumps({
        "error": {
            "code": "missing_filter_inputs",
            "message": "STATION or RANK not provided"
        }
    }))
    raise SystemExit

filtered = df[(df["Station"] == STATION) & (df["Rank"] == RANK)]

if filtered.empty:
    print(json.dumps({
        "error": {
            "code": "empty_filter_result",
            "message": "No rows after Station/Rank filter"
        }
    }))
    raise SystemExit


# ==============================
# 3️⃣ Select Latest 3 Dates
# ==============================

parsed_dates = pd.to_datetime(filtered["Date"], errors="coerce")
filtered = filtered.assign(_parsed_date=parsed_dates)

distinct_dates = sorted(filtered["_parsed_date"].dropna().unique())

if len(distinct_dates) < 3:
    print(json.dumps({
        "error": {
            "code": "fewer_than_3_dates",
            "message": "Less than 3 distinct dates available"
        }
    }))
    raise SystemExit

latest_3 = distinct_dates[-3:]
selected = filtered[filtered["_parsed_date"].isin(latest_3)]

selected = selected.drop(columns=["_parsed_date"])


# ==============================
# 4️⃣ Strict Feature Validation
# ==============================

for col in FEATURES:
    if col not in selected.columns:
        print(json.dumps({
            "error": {
                "code": "missing_features",
                "message": f"Missing feature column: {col}"
            }
        }))
        raise SystemExit

if selected[FEATURES].isnull().any().any():
    print(json.dumps({
        "error": {
            "code": "missing_features",
            "message": "Null values found in required features"
        }
    }))
    raise SystemExit

X = selected[FEATURES]
dtest = xgb.DMatrix(X, feature_names=FEATURES)


# ==============================
# 5️⃣ Predict
# ==============================

preds = booster.predict(dtest)
preds = np.clip(preds, 0, 1)


# ==============================
# 6️⃣ Build Output JSON
# ==============================

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
