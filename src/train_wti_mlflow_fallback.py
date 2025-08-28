import os
import tempfile
import time
import pandas as pd
import numpy as np
from minio import Minio

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# Prefer new RMSE API (scikit-learn >= 1.4), fallback otherwise
try:
    from sklearn.metrics import root_mean_squared_error as rmse_fn
    def compute_rmse(y_true, y_pred):
        return float(rmse_fn(y_true, y_pred))
except Exception:
    from sklearn.metrics import mean_squared_error
    def compute_rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

# ------------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------------
#CSV_PATH = os.getenv("CSV_PATH", "s3://oil/COMBINED_updated.csv")
CSV_PATH = os.getenv("CSV_PATH", "Data\COMBINED_updated.csv")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "WTI-Forecasting")
PREDICTION_HORIZON = int(os.getenv("PRED_H", "1"))
LAGS = tuple(int(x) for x in os.getenv("LAGS", "1,4,20").split(","))
ROLLS = tuple(int(x) for x in os.getenv("ROLLS", "6,20").split(","))
TRAIN_FRACTION = float(os.getenv("TRAIN_FRAC", "0.8"))
RIDGE_ALPHA = float(os.getenv("RIDGE_ALPHA", "1.0"))
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "")
TARGET_STAGE = os.getenv("TARGET_STAGE", "Staging")

# >>> MinIO/S3 parameters used by pandas s3fs (optional but recommended for explicitness)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")           # e.g. https://minio.prod.local
MINIO_VERIFY_SSL = os.getenv("MINIO_SECURE", "false").lower() == "true" 

# Optional: silence TLS warnings if using self-signed
if os.getenv("MLFLOW_TRACKING_INSECURE_TLS", "").lower() in ("1", "true", "yes"):
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



# Optional: disable TLS warnings if using self-signed certs
if os.getenv("MLFLOW_TRACKING_INSECURE_TLS", "").lower() in ("1", "true", "yes"):
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ------------------------------------------------------------------------------------
# Data loading & feature engineering
# ------------------------------------------------------------------------------------
def read_csv_anywhere(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        # Requires: pip install s3fs
        storage_options = {}
        if MINIO_ENDPOINT:
            storage_options["client_kwargs"] = {"endpoint_url": MINIO_ENDPOINT, "verify": MINIO_VERIFY_SSL}
        return pd.read_csv(path, storage_options=storage_options)
    return pd.read_csv(path)



df = read_csv_anywhere(CSV_PATH)

# --- Normalize headers ---
import re
def _norm_cols(cols):
    return [re.sub(r"\s+", " ", str(c).replace("\u00a0", " ")).strip() for c in cols]
df.columns = _norm_cols(df.columns)

# --- Map synonyms -> canonical column names expected by the pipeline ---
# Canonical names we want: Date, WTI, DJU, Gold, SP500, US10Y, USD_INDEX
synonyms = {
    "Date": ["Date", "DATE", "date", "Timestamp", "timestamp", "Datetime", "Day"],
    "WTI": ["WTI", "WTI Price", "WTI_price", "Crude Oil WTI", "WTI Crude"],
    "DJU": ["DJU", "Degres-Jours", "Degrés-Jours", "Degree Days", "HDD"],
    "Gold": ["Gold", "Gold Price", "XAU", "XAUUSD"],
    "SP500": ["SP500", "SP 500", "S&P 500", "S&P500", "SP-500"],
    "US10Y": ["US10Y", "US 10YR BOND", "10Y", "10-Year", "10Y Treasury", "US 10 Year"],
    "USD_INDEX": ["USD_INDEX", "US DOLLAR INDEX", "Dollar Index", "DXY"],
}

lower_map = {c.lower(): c for c in df.columns}
rename_map = {}
for target, cands in synonyms.items():
    for cand in cands:
        if cand.lower() in lower_map:
            rename_map[lower_map[cand.lower()]] = target
            break
df = df.rename(columns=rename_map)

# --- If no 'Date' after mapping, try to auto-detect a date-like column ---
if "Date" not in df.columns:
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.9:  # ~90% parseable -> assume it's the date
            df["Date"] = parsed
            break
if "Date" not in df.columns:
    raise ValueError(f"Could not find a date column. Available columns: {list(df.columns)}")

# --- Ensure numeric cols exist (after mapping) ---
needed = ["WTI", "DJU", "Gold", "SP500", "US10Y", "USD_INDEX"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns after normalization/mapping: {missing}. "
                     f"Available: {list(df.columns)}")

# --- Coerce types ---
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
for c in ["DJU", "Gold", "SP500", "US10Y", "USD_INDEX", "WTI"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# --- Clean & order ---
df = df.dropna(subset=["Date", "WTI"]).sort_values("Date").reset_index(drop=True)
df[["DJU", "Gold", "SP500", "US10Y", "USD_INDEX", "WTI"]] = (
    df[["DJU", "Gold", "SP500", "US10Y", "USD_INDEX", "WTI"]].ffill()
)

#df = df.rename(columns={
#    'SP 500': 'SP500',
#    'US 10YR BOND': 'US10Y',
#    'US DOLLAR INDEX': 'USD_INDEX'
#})
#df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#for c in ['DJU', 'Gold', 'SP500', 'US10Y', 'USD_INDEX', 'WTI']:
#    df[c] = pd.to_numeric(df[c], errors='coerce')

#df = df.dropna(subset=['Date', 'WTI']).sort_values('Date').reset_index(drop=True)
#df[['DJU', 'Gold', 'SP500', 'US10Y', 'USD_INDEX', 'WTI']] = df[['DJU', 'Gold', 'SP500', 'US10Y', 'USD_INDEX', 'WTI']].ffill()

def make_lags(frame, cols, lags=(1, 5, 20)):
    out = frame.copy()
    for c in cols:
        for L in lags:
            out[f"{c}_lag{L}"] = out[c].shift(L)
    return out

def make_rolls(frame, cols, windows=(5, 20)):
    out = frame.copy()
    for c in cols:
        for w in windows:
            out[f"{c}_rollmean{w}"] = out[c].rolling(w).mean()
            out[f"{c}_rollstd{w}"] = out[c].rolling(w).std()
    return out

base_cols = ['WTI', 'DJU', 'Gold', 'SP500', 'US10Y', 'USD_INDEX']
fe = df[['Date'] + base_cols].copy()
fe = make_lags(fe, base_cols, lags=LAGS)
fe = make_rolls(fe, base_cols, windows=ROLLS)
fe = fe.dropna().reset_index(drop=True)

fe['WTI_target'] = fe['WTI'].shift(-PREDICTION_HORIZON)
fe = fe.dropna().reset_index(drop=True)

feature_cols = [c for c in fe.columns if c not in ['Date', 'WTI_target']]
X = fe[feature_cols]
y = fe['WTI_target']

split_idx = int(len(fe) * TRAIN_FRACTION)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


# ------------------------------------------------------------------------------------
# Ensure experiment artifacts go to MinIO
# ------------------------------------------------------------------------------------
artifact_uri = os.getenv("MLFLOW_ARTIFACT_URI")  # e.g., s3://mlflow-artifacts/wti/
client = MlflowClient()
exp = client.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    if not artifact_uri:
        raise RuntimeError("MLFLOW_ARTIFACT_URI must be set to an s3:// path for MinIO artifact storage.")
    exp_id = client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_uri)
else:
    exp_id = exp.experiment_id
    if artifact_uri and exp.artifact_location != artifact_uri:
        # Existing experiments keep their original artifact_location; warn if different.
        print(f"[WARN] Experiment exists with artifact_location={exp.artifact_location}. "
              f"Ignoring MLFLOW_ARTIFACT_URI={artifact_uri}")


# ------------------------------------------------------------------------------------
# Model training
# ------------------------------------------------------------------------------------
preprocess = ColumnTransformer([("num", StandardScaler(), X.columns.tolist())])
model = Ridge(alpha=RIDGE_ALPHA)
pipe = Pipeline([("prep", preprocess), ("model", model)])

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name=f"ridge_wti_h{PREDICTION_HORIZON}") as run:
    mlflow.log_params({
        "model": "Ridge",
        "ridge_alpha": RIDGE_ALPHA,
        "train_fraction": TRAIN_FRACTION,
        "horizon": PREDICTION_HORIZON,
        "lags": LAGS,
        "rolls": ROLLS,
        "n_features": len(feature_cols),
        "data_path": CSV_PATH
    })

    # Log raw dataset (optional)
    # Log dataset path reference (dataset file is already in MinIO)
    try:
        if not CSV_PATH.startswith("s3://"):
            mlflow.log_artifact(CSV_PATH, artifact_path="datasets")
    except Exception as e:
        print(f"[WARN] Could not log dataset artifact: {e}")

    # Fit & predict
    pipe.fit(X_train, y_train)
    pred_test = pipe.predict(X_test)

    # Metrics
    mae = float(mean_absolute_error(y_test, pred_test))
    rmse = compute_rmse(y_test, pred_test)
    mape = float((np.abs((y_test - pred_test) / y_test.replace(0, np.nan))).median() * 100)
    r2 = float(r2_score(y_test, pred_test))

    mlflow.log_metrics({
        "test_mae": mae,
        "test_rmse": rmse,
        "test_median_mape_pct": mape,
        "test_r2": r2
    })

    # Signature & example
    X_example = X_train.iloc[:5].copy()
    y_pred_example = pipe.predict(X_example)
    signature = infer_signature(X_example, y_pred_example)

    # Save artifacts (metrics/preds)
    out_dir = "artifacts"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"metric": ["MAE", "RMSE", "MedianMAPE%", "R2"],
                  "value": [mae, rmse, mape, r2]}).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    pd.DataFrame({"Date": fe['Date'].iloc[split_idx:].values,
                  "WTI_actual": y_test.values,
                  "WTI_pred": pred_test}).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    mlflow.log_artifacts(out_dir)

    # Try log_model with registry; if it fails or no name given, we fallback + register
    did_register_via_log_model = False
    try:
        if REGISTERED_MODEL_NAME:
            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAME,
                signature=signature,
                input_example=X_example,
            )
            did_register_via_log_model = True
        else:
            # No registry name provided: just log as artifact
            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                signature=signature,
                input_example=X_example,
            )
    except Exception as e:
        print(f"[WARN] log_model failed: {e}")
        print("[INFO] Falling back to save_model + log_artifacts...")
        with tempfile.TemporaryDirectory() as tmp:
            local_model_dir = os.path.join(tmp, "model")
            mlflow.sklearn.save_model(
                sk_model=pipe,
                path=local_model_dir,
                signature=signature,
                input_example=X_example,
            )
            mlflow.log_artifacts(local_model_dir, artifact_path="model")

    client = MlflowClient()
    run_id = run.info.run_id
    name = REGISTERED_MODEL_NAME or "WTI_Ridge"

    if did_register_via_log_model:
        # --- We used log_model with registered_model_name ---
        # Find the Model Version created for THIS run, then tag & transition it.
        mvs = client.search_model_versions(f"name='{name}' and run_id='{run_id}'")
        if mvs:
            mv = max(mvs, key=lambda x: int(x.version))
            # Poll until READY
            for _ in range(20):
                mv = client.get_model_version(name=name, version=mv.version)
                if mv.status == "READY":
                    break
                time.sleep(1)
            # Set model-version tags
            client.set_model_version_tag(name, mv.version, "framework", "sklearn")
            client.set_model_version_tag(name, mv.version, "source_run", run_id)
            # Transition stage
            client.transition_model_version_stage(
                name=name, version=mv.version, stage=TARGET_STAGE, archive_existing_versions=False
            )
            print(f"[INFO] Tagged and transitioned {name} v{mv.version} to {TARGET_STAGE}")
        else:
            print(f"[WARN] No registry version found for run_id={run_id} (name={name}).")
    else:
        # --- We did NOT register via log_model (older server or no REGISTERED_MODEL_NAME) ---
        model_uri = f"runs:/{run_id}/model"
        mv = client.register_model(model_uri=model_uri, name=name)
        print(f"[INFO] Registered model: name={name}, version={mv.version}, status={mv.status}")
        # Poll until READY
        for _ in range(20):
            mv = client.get_model_version(name=name, version=mv.version)
            if mv.status == "READY":
                break
            time.sleep(1)
        # Set tags & transition stage
        client.set_model_version_tag(name, mv.version, "framework", "sklearn")
        client.set_model_version_tag(name, mv.version, "source_run", run_id)
        client.transition_model_version_stage(
            name=name, version=mv.version, stage=TARGET_STAGE, archive_existing_versions=False
        )
        print(f"[INFO] Tagged and transitioned {name} v{mv.version} to {TARGET_STAGE}")

print("Done. Run logged to MLflow and model registered with tags & stage.")
