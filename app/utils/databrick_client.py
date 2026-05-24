import os
import time
from typing import Dict, Optional, Any
import requests
import numpy as np
import pandas as pd
import streamlit as st

# Exact columns the uv-forecast-router endpoint schema requires
_ROUTER_SCHEMA_COLS = [
    'model_type', 'timestamp', 'location_id',
    'cos_solar_zenith', 'doy_sin', 'temperature_2m', 'relative_humidity_2m',
    'cloud_cover', 'solar_cloud_interaction', 'ozone_anomaly', 'pressure_msl',
    'wind_speed_10m', 'altitude_m', 'cos_zenith_squared', 'cloud_attenuation_exp',
    'temp_humidity_product', 'pressure_cloud_interaction', 'temperature_2m_ema',
    'cloud_cover_ema', 'ozone_ema', 'altitude_solar_interaction',
    'ozone_depletion_risk', 'air_quality_combined', 'is_raining', 'cloud_cover_change_1h'
]

class DataBrickModelClient:
    def __init__(self):
        self.host = os.getenv("DATABRICKS_HOST", "localhost")
        self.token = os.getenv("DATABRICKS_TOKEN", "")
        self.enabled = bool(self.host and self.token)

        if not self.enabled:
            st.warning("Databricks credential are not configured")

    def get_endpoint_url(self, model_name: str) -> str:
        return f"{self.host}/serving-endpoints/uv-forecast-router/invocations"

    def _get_model_key(self, model_name: str) -> str:
        endpoint_mapping = {
            "Random Forest": "rf",
            "Decision Tree": "dt",
            "XGBoost": "xgb",
            "LightGBM": "lgb",
            "CatBoost": "catboost",
            "LSTM": "lstm",
            "GRU": "gru",
            "BiLSTM": "bilstm",
            "CNN-LSTM": "cnn_lstm",
            "Attention-LSTM": "attention_lstm",
            "Prophet+LGB": "prophet_lgb",
        }
        return endpoint_mapping.get(model_name, model_name.lower())

    def health_check(self, model_name: str):
        if not self.enabled:
            return False
        return True

    def predict_batch(
            self,
            features_df: pd.DataFrame,
            model_name: str,
            max_retries: int = 3,
            timeout: int = 30
    ) -> Optional[np.ndarray]:

        if not self.enabled:
            return None

        endpoint_url = self.get_endpoint_url(model_name)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "content-type": "application/json",
        }
        features_df = features_df.copy()
        features_df['model_type'] = self._get_model_key(model_name)

        if 'timestamp' in features_df.columns:
            features_df['timestamp'] = features_df['timestamp'].astype(str)

        cols_to_send = [c for c in _ROUTER_SCHEMA_COLS if c in features_df.columns]
        clean_df = features_df[cols_to_send].astype(object)
        
        clean_df = clean_df.where(pd.notnull(clean_df), None)

        payload = {
            "dataframe_split": clean_df.to_dict(orient="split")
        }

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = requests.post(
                    endpoint_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                latency = time.time() - start_time
                if response.status_code == 200:
                    result = response.json()
                    predictions = result.get('predictions', [])

                    if predictions and isinstance(predictions[0], dict):
                        predictions = [p.get('uv_predicted', 0.0) for p in predictions]

                    st.session_state['last_serving_latency'] = latency
                    st.session_state['last_serving_status'] = 'success'

                    return np.array(predictions, dtype=float)
                elif response.status_code == 429:
                    wait_time = min(2 ** attempt, 16)
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = f"Http {response.status_code}: {response.text}"
                    st.session_state['last_serving_error'] = error_msg
                    return None

            except requests.Timeout as e:
                st.session_state['last_serving_error'] = f"Request timeout ({timeout}s)"
                return None
            except Exception as e:
                st.error(f"Error {e}")
                st.session_state['last_serving_error'] = str(e)
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 8)
                    time.sleep(wait_time)
                else:
                    return None
        return None

    def predict_single(self, features: Dict[str, float], model_name: str) -> Optional[float]:
        df = pd.DataFrame([features])
        predictions = self.predict_batch(df, model_name)
        if predictions is not None and len(predictions) > 0:
            return float(predictions[0])
        return None

@st.cache_data(ttl=300, show_spinner=False)
def get_serving_endpoint_status(model_name: str) -> Dict[str, Any]:
    client = DataBrickModelClient()

    if not client.enabled:
        return {
            "available": False,
            "error": "Databricks not configured",
            "latency": None
        }

    start = time.time()
    is_healthy = client.health_check(model_name)
    latency = time.time() - start

    return {
        "available": is_healthy,
        "latency": latency if is_healthy else None,
        "error": None if is_healthy else "Endpoint unavailable",
    }

def format_serving_status(status: Dict[str, Any]) -> str:
   if status['available']:
       return "Online"
   else:
       error = status.get('error', 'unknown error')
       return f"Error ({error})"