# WTI Price Forecasting

This project trains a Ridge regression model to forecast West Texas Intermediate (WTI) crude oil prices and logs the run to [MLflow](https://mlflow.org/). The main entry point is [`train_wti_mlflow_fallback.py`](train_wti_mlflow_fallback.py), which performs feature engineering, model training and MLflow model registration.

## How the training script works

1. **Configuration** – runtime options are read from environment variables. Key ones include the input CSV path (`CSV_PATH`), the MLflow experiment name (`MLFLOW_EXPERIMENT`), the prediction horizon (`PRED_H`), feature lags (`LAGS`), rolling windows (`ROLLS`), and Ridge regression strength (`RIDGE_ALPHA`). Defaults are provided in the script for local runs.
2. **Data preparation** – the script loads the dataset from a local file or an S3/MinIO bucket, cleans up column names, maps synonyms to canonical names and builds lagged and rolling‑window features.
3. **Training** – a scikit‑learn pipeline consisting of a `StandardScaler` and `Ridge` model is fitted on the training portion of the data. Metrics such as MAE and RMSE are computed on the hold‑out set and logged to MLflow.
4. **Logging and registration** – metrics, predictions and the model artifact are logged. The model is then registered in the MLflow Model Registry under `REGISTERED_MODEL_NAME` (or `WTI_Ridge` if not provided) and transitioned to the stage specified by `TARGET_STAGE`.

## Running the training

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up the required environment variables (MLflow tracking URI, S3/MinIO credentials, etc.) and run:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_ARTIFACT_URI=s3://mlflow-artifacts/wti/
python train_wti_mlflow_fallback.py
```

## Calling the registered model

Once the run finishes, the model is available in the MLflow Model Registry. You can load it and make predictions with:

```python
import mlflow
import pandas as pd

model = mlflow.pyfunc.load_model("models:/WTI_Ridge/Staging")
# `input_df` should contain the same engineered feature columns used for training
predictions = model.predict(input_df)
```

The model can also be referenced by run ID:

```python
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
```

For serving, any MLflow tooling such as `mlflow models serve` can be used to expose the registered model as a REST API.


## Running super-linter locally

Developers can run the same checks locally using Docker:

```bash
docker run \
  -e RUN_LOCAL=true \
  -e VALIDATE_PYTHON=true \
  -e VALIDATE_JAVA=true \
  -e VALIDATE_JAVASCRIPT=true \
  -e VALIDATE_SCALA=true \
  -e VALIDATE_SQL=true \
  -e VALIDATE_JSON=true \
  -v "$PWD:/tmp/lint" \
  ghcr.io/github/super-linter:slim-latest
```

