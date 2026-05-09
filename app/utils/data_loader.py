from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import os
from databricks.sdk import WorkspaceClient

ROOT         = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "data" / "processed" / "features"
MODELS_DIR   = ROOT / "models" / "optimized"
METADATA_DIR = ROOT / "models" / "metadata"
RESULTS_DIR  = ROOT / "results" / "optimized"

_MODEL_DISPLAY_NAMES = {
    "bilstm":         "BiLSTM",
    "rf":             "Random Forest",
    "dt":             "Decision Tree",
    "xgb":            "XGBoost",
    "lgb":            "LightGBM",
    "catboost":       "CatBoost",
    "lstm":           "LSTM",
    "gru":            "GRU",
    "cnn_lstm":       "CNN-LSTM",
    "attention_lstm": "Attention-LSTM",
    "prophet_lgb":    "Prophet+LGB",
}

def get_available_models() -> dict[str, Path]:
    return {display: key for key, display in _MODEL_DISPLAY_NAMES.items()}


LOCATION_NAMES = {
    "hcm":      "TP. Hồ Chí Minh",
    "vungtau":  "Vũng Tàu",
    "cuchi":    "Củ Chi",
    "nhabe":    "Nhà Bè",
    "thuduc":   "Thủ Đức",
    "cangio":   "Cần Giờ",
    "longhai":  "Long Hải",
}

LOCATION_COLORS = {
    "hcm":      "#E74C3C",
    "vungtau":  "#3498DB",
    "cuchi":    "#2ECC71",
    "nhabe":    "#F39C12",
    "thuduc":   "#9B59B6",
    "cangio":   "#1ABC9C",
    "longhai":  "#E67E22",
}

UV_CATEGORY_COLORS = {
    "Low":       "#27ae60",
    "Moderate":  "#f1c40f",
    "High":      "#e67e22",
    "Very High": "#e74c3c",
    "Extreme":   "#8e44ad",
}

@st.cache_resource
def get_databricks_client():
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")

    if not host or not token:
        return None

    return WorkspaceClient(host=host, token=token)


def download_from_volume(volume_path: str, local_cache_path: Path) -> Path:
    client = get_databricks_client()
    if client is None:
        return local_cache_path
    import time
    import logging

    logger = logging.getLogger(__name__)

    local_cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_cache_path.exists():
        start_time = time.time()
        try:
            response = client.files.download(volume_path)
            with open(local_cache_path, "wb") as f:
                f.write(response.contents.read())

        except Exception as e:
            logger.error(f"Download from Databricks: {volume_path} failed")
            st.warning(f"Failed to download from Databricks: {e}")
            return local_cache_path

    return local_cache_path

# @st.cache_data()
def load_regression() -> pd.DataFrame:
    volume_base = os.getenv("VOLUME_PATH")
    regression_file = os.getenv("FEATURES_REGRESSION_PATH")
    use_databricks = all([
        os.getenv("DATABRICKS_HOST"),
        os.getenv("DATABRICKS_TOKEN"),
        volume_base
    ])

    local_path = FEATURES_DIR / "features_regression.csv"

    if use_databricks:
        volume_path = f"{volume_base}/features/{regression_file}"
        path = download_from_volume(volume_path, local_path)
    else:
        path = local_path

    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])

    if "location_id" in df.columns:
        df["location_name"] = df["location_id"].map(LOCATION_NAMES)

    return df

@st.cache_data(show_spinner="Đang tải kết quả mô hình…")
def load_regression_results() -> pd.DataFrame:
    volume_base = os.getenv("VOLUME_PATH")
    use_databricks = all([
        os.getenv("DATABRICKS_HOST"),
        os.getenv("DATABRICKS_TOKEN"),
        volume_base
    ])

    local_path = RESULTS_DIR / "consolidated_results.csv"
    if use_databricks:
        volume_path = f"{volume_base}/results/consolidated_results.csv"
        path = download_from_volume(volume_path, local_path)
    else:
        path = local_path
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_regression_results_pivot() -> pd.DataFrame:
    df = load_regression_results()
    metrics = ["mae", "rmse", "r2", "mape"]
    
    pivoted = df.pivot_table(
        index=["type", "model"],
        columns="split",
        values=metrics,
        aggfunc="first",
    )
    pivoted.columns = [f"{split}_{metric}" for metric, split in pivoted.columns]
    pivoted = pivoted.reset_index()
    
    col_order = ["type", "model"]
    for split in ["train", "val", "test"]:
        for metric in metrics:
            col_name = f"{split}_{metric}"
            if col_name in pivoted.columns:
                col_order.append(col_name)
    pivoted = pivoted[[c for c in col_order if c in pivoted.columns]]
    
    if "test_rmse" in pivoted.columns:
        pivoted = pivoted.sort_values("test_rmse").reset_index(drop=True)
    
    return pivoted


@st.cache_data
def load_model_metadata(model_key: str) -> dict:
    """Load Optuna best hyperparameters for a model."""
    local_path = METADATA_DIR / f"best_params_{model_key}.json"
    volume_base = os.getenv("VOLUME_PATH")
    use_databricks = all([
        os.getenv("DATABRICKS_HOST"),
        os.getenv("DATABRICKS_TOKEN"),
        volume_base
    ])
    
    if use_databricks:
        volume_path = f"{volume_base}/metadata/best_params_{model_key}.json"
        path = download_from_volume(volume_path, local_path)
    else:
        path = local_path
    
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_resource(show_spinner="Đang tải mô hình…")
def load_optimized_model(model_name: str):
    import torch
    available = get_available_models()
    if model_name not in available:
        st.error(f"Mô hình '{model_name}' không khả dụng. Có sẵn: {list(available.keys())}")
        return None

    model_key = available[model_name]
    model_path = MODELS_DIR /f"{model_key}_optimized.joblib"
    if not model_path.exists():
        st.info(f"su dung databrick naoooo")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        kwargs["map_location"] = device
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
    try:
        model = joblib.load(model_path)
    finally:
        torch.load = _original_torch_load

    if hasattr(model, "model") and hasattr(model, "device"):
        model.model = model.model.to(device)
        model.device = device

    return model
