
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pvlib
import requests
import streamlit as st

from config import LOCATIONS, TIMEZONE, UV_CATEGORY_BINS, FINAL_22_FEATURES
from app.utils.databrick_client import DataBrickModelClient

def _get_model_type(model_name: str) -> str:
    if model_name is None:
        return 'traditional'
    name_lower = model_name.lower()
    sequence_keywords = ['lstm', 'gru', 'bilstm', 'cnn-lstm', 'attention']
    if any(kw in name_lower for kw in sequence_keywords):
        return 'sequence'
    if 'prophet' in name_lower:
        return 'prophet'

    return 'traditional'


# -- Open-Meteo endpoints (free, no API key) --------------------------------
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_AQ_URL       = "https://air-quality-api.open-meteo.com/v1/air-quality"

_WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m",
    "cloud_cover", "cloud_cover_high",
    "precipitation", "pressure_msl",
    "wind_speed_10m",
    "shortwave_radiation",       # kept for solar_cloud_interaction calc only
]
_AQ_VARS = ["ozone", "aerosol_optical_depth", "pm2_5"]

# The Open-Meteo Air Quality API only provides 5 days of forecast (not 7)
_AQ_MAX_FORECAST_DAYS = 5

# Minimum history window (hours) for sequence/prophet models to have enough lookback
_HISTORY_HOURS = 120  # 5 days

# Default fallback — will be overridden by dynamic computation from training data
_OZONE_Q25_DEFAULT = 56.0


@st.cache_data(ttl=86400, show_spinner=False)
def _get_ozone_q25() -> float:
    """Compute ozone Q25 from training data for consistency with BE pipeline.

    BE computes quantile(0.25) dynamically from training data (engr.py L340).
    This function replicates that to avoid train/infer mismatch.
    Falls back to 56.0 if data unavailable.
    """
    full_path = ROOT / "data" / "processed" / "features" / "features_full.csv"
    if not full_path.exists():
        return _OZONE_Q25_DEFAULT
    try:
        import pandas as _pd
        df = _pd.read_csv(full_path, usecols=["ozone"])
        ozone = df["ozone"].dropna()
        if ozone.empty:
            return _OZONE_Q25_DEFAULT
        return float(ozone.quantile(0.25))
    except Exception:
        return _OZONE_Q25_DEFAULT

def _fetch_historical(endpoint: str, loc_id: str, loc: dict,
                      past_hours: int = 48,
                      hourly_vars: list | None = None) -> pd.DataFrame:
    if hourly_vars is None:
        hourly_vars = _WEATHER_VARS
    params = {
        "latitude": loc["lat"],
        "longitude": loc["lon"],
        "timezone": TIMEZONE,
        "past_hours": past_hours,
        "hourly": ",".join(hourly_vars),
    }
    resp = requests.get(endpoint, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("hourly", {})
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"])
    df["location_id"] = loc_id
    return df.drop(columns=["time"])

@st.cache_data(ttl=7200, show_spinner=False)
def _fetch_weather_historical(loc_id: str, loc: dict, past_hours: int = 48) -> pd.DataFrame:
    return _fetch_historical(_FORECAST_URL, loc_id, loc, past_hours, hourly_vars=_WEATHER_VARS)

@st.cache_data(ttl=7200, show_spinner=False)
def _fetch_air_quality_historical(loc_id: str, loc: dict, past_hours: int = 48) -> pd.DataFrame:
    return _fetch_historical(_AQ_URL, loc_id, loc, past_hours, hourly_vars=_AQ_VARS)

@st.cache_data(ttl=3600, show_spinner=False)
def _get_ozone_monthly_means() -> dict[tuple, float]:
    full_path = ROOT / "data" / "processed" / "features" / "features_full.csv"
    if not full_path.exists():
        return {}
    df_full = pd.read_csv(full_path, parse_dates=["timestamp"],
                          usecols=["timestamp", "location_id", "ozone"])
    df_full = df_full.dropna(subset=["ozone"])
    df_full["month_period"] = df_full["timestamp"].dt.to_period("M").astype(str)
    means = (
        df_full.groupby(["location_id", "month_period"])["ozone"]
        .mean()
        .to_dict()
    )
    return means


def _fetch_weather(loc_id: str, loc: dict, forecast_days: int) -> pd.DataFrame:
    params = {
        "latitude":      loc["lat"],
        "longitude":     loc["lon"],
        "timezone":      TIMEZONE,
        "forecast_days": forecast_days,
        "hourly":        ",".join(_WEATHER_VARS),
    }
    resp = requests.get(_FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("hourly", {})
    df = pd.DataFrame(data)
    df["timestamp"]   = pd.to_datetime(df["time"])
    df["location_id"] = loc_id
    return df.drop(columns=["time"])


def _fetch_air_quality(loc_id: str, loc: dict, forecast_days: int) -> pd.DataFrame:
    params = {
        "latitude":      loc["lat"],
        "longitude":     loc["lon"],
        "timezone":      TIMEZONE,
        "forecast_days": forecast_days,
        "hourly":        ",".join(_AQ_VARS),
    }
    resp = requests.get(_AQ_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("hourly", {})
    df = pd.DataFrame(data)
    df["timestamp"]   = pd.to_datetime(df["time"])
    df["location_id"] = loc_id
    return df.drop(columns=["time"])

def _fetch_combined_data(loc_id: str, loc:dict, forecast_day: int, need_history: bool) -> pd.DataFrame:
    weather_fc = _fetch_weather(loc_id, loc, forecast_day)
    # AQ API only supports 5 days of forecast — cap to avoid silent truncation
    aq_forecast_days = min(forecast_day, _AQ_MAX_FORECAST_DAYS)
    aq_fc = _fetch_air_quality(loc_id, loc, aq_forecast_days)
    forecast_df = weather_fc.merge(
        aq_fc[["timestamp", "ozone", "aerosol_optical_depth", "pm2_5"]],
        on="timestamp", how="left"
    )
    if not need_history:
        return forecast_df
    try:
        weather_hist = _fetch_weather_historical(loc_id, loc, past_hours=_HISTORY_HOURS)
        aq_hist = _fetch_air_quality_historical(loc_id, loc, past_hours=_HISTORY_HOURS)

        historical_df = weather_hist.merge(
            aq_hist[["timestamp", "ozone", "aerosol_optical_depth", "pm2_5"]],
            on="timestamp", how="left"
        )
        combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["timestamp", "location_id"], keep="last")
        combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
        # Forward-fill AQ columns so gaps beyond day 5 don't propagate NaN into model inputs
        for col in ["ozone", "aerosol_optical_depth", "pm2_5"]:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].ffill()
        return combined_df
    except Exception as e:
        st.warning(f"could not fetch historical data for {loc['name']} : {e}")
        return forecast_df


def _compute_solar(df: pd.DataFrame, loc_id: str, loc: dict) -> pd.DataFrame:
    ts = pd.DatetimeIndex(df["timestamp"]).tz_localize(
        TIMEZONE, ambiguous="NaT", nonexistent="shift_forward"
    )
    solpos = pvlib.solarposition.get_solarposition(
        ts, latitude=loc["lat"], longitude=loc["lon"],
        altitude=loc.get("altitude_m", 0)
    )
    df = df.copy()
    df["solar_elevation"] = solpos["apparent_elevation"].values
    df["solar_zenith"]    = solpos["apparent_zenith"].values
    zenith_rad = np.radians(df["solar_zenith"].clip(lower=0))
    df["cos_solar_zenith"] = np.cos(zenith_rad).clip(lower=0)
    df.loc[df["solar_elevation"] <= 0, "cos_solar_zenith"] = 0.0
    return df


def _compute_all_features(df: pd.DataFrame, loc_id: str, loc: dict,
                           ozone_means: dict) -> pd.DataFrame:
    df = df.copy()

    df["hour"]     = df["timestamp"].dt.hour
    df["month"]    = df["timestamp"].dt.month
    df["doy"]      = df["timestamp"].dt.dayofyear
    df["doy_sin"]  = np.sin(2 * np.pi * df["doy"] / 365.25)

    cc = df.get("cloud_cover", pd.Series(0, index=df.index)).fillna(0) / 100
    df["solar_cloud_interaction"] = df["cos_solar_zenith"] * (1 - cc)

    month_str = df["timestamp"].dt.to_period("M").astype(str)
    monthly_baseline = month_str.map(
        lambda m: ozone_means.get((loc_id, m), np.nan)
    )
    raw_ozone = df.get("ozone", pd.Series(np.nan, index=df.index))
    df["ozone_anomaly"] = raw_ozone - monthly_baseline
    df["ozone_anomaly"] = df["ozone_anomaly"].fillna(0.0)

    df["altitude_m"] = loc.get("altitude_m", 0)

    df["cos_zenith_squared"] = df["cos_solar_zenith"] ** 2

    cc_raw = df.get("cloud_cover", pd.Series(50, index=df.index)).fillna(50)
    # Kasten & Czeplak (1980) Cloud Modification Factor
    # CMF = 1 - 0.75 * (CC/100)^3.4
    # Source: doi:10.1016/0038-092X(80)90391-6
    df["cloud_attenuation_exp"] = (1 - 0.75 * np.power(cc_raw / 100.0, 3.4)).clip(0.05, 1.0)

    temp = df.get("temperature_2m", pd.Series(np.nan, index=df.index))
    rh   = df.get("relative_humidity_2m", pd.Series(np.nan, index=df.index))
    df["temp_humidity_product"] = temp * rh / 100

    pressure = df.get("pressure_msl", pd.Series(1013, index=df.index))
    df["pressure_cloud_interaction"] = pressure * (1 - cc)

    for col in ["temperature_2m", "cloud_cover", "ozone"]:
        src = df.get(col, pd.Series(np.nan, index=df.index)).ffill().fillna(0)
        df[f"{col}_ema"] = src.ewm(alpha=0.3, adjust=False).mean()

    df["altitude_solar_interaction"] = df["altitude_m"] * df["cos_solar_zenith"] / 1000

    ozone_val = df.get("ozone", pd.Series(np.nan, index=df.index)).fillna(70)
    ozone_q25 = _get_ozone_q25()
    df["ozone_depletion_risk"] = (ozone_val < ozone_q25).astype(int)

    aod  = df.get("aerosol_optical_depth", pd.Series(0, index=df.index)).fillna(0)
    pm25 = df.get("pm2_5", pd.Series(0, index=df.index)).fillna(0)
    df["air_quality_combined"] = 0.6 * aod + 0.4 * (pm25 / 100)

    precip = df.get("precipitation", pd.Series(0, index=df.index)).fillna(0)
    df["is_raining"] = (precip > 0).astype(int)

    df["cloud_cover_change_1h"] = cc_raw.diff(1).fillna(0)

    return df


def _predict(
    df: pd.DataFrame,
    model,
    feat_cols: list[str],
    use_serving: bool = False,
    model_name: str | None = None,
    databricks_client = None,
) -> pd.DataFrame:

    df = df.copy()
    daytime = df["solar_elevation"] > 0

    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    X = df[feat_cols].fillna(0)

    preds = np.full(len(df), 0.0)
    if daytime.any():
        model_type = _get_model_type(model_name)
        if model_type in ('sequence', 'prophet'):
            required_cols = ['timestamp', 'location_id']
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"sequence/prophet model require columns {required_cols} "
                    f"but missing columns {missing_cols}"
                )
            model_input = df[daytime].copy()
        else:
            # Apply physics constraint: X *= cos_solar_zenith
            # This must match training pipeline (train_pipeline.py L55-59)
            # where apply_physics_constraint() multiplies all features by cos_z.
            X_day = X.values[daytime].copy()
            cos_zenith_idx = feat_cols.index('cos_solar_zenith') if 'cos_solar_zenith' in feat_cols else None
            if cos_zenith_idx is not None:
                cos_z = X_day[:, cos_zenith_idx].clip(min=0)
                X_day = X_day * cos_z[:, np.newaxis]
            model_input = X_day

        if use_serving and databricks_client and databricks_client.enabled:
            try:
                serving_preds = databricks_client.predict_batch(
                    df[daytime],
                    model_name,
                    max_retries=2,
                    timeout=30
                )
                if serving_preds is not None:
                    preds[daytime] = np.maximum(0, serving_preds)
                    st.session_state['prediction_source'] = 'databricks_serving'
                else:
                    if model is not None:
                        st.warning("Databricks serving failed, using local model")
                        preds[daytime] = np.maximum(0, model.predict(model_input))
                        st.session_state['prediction_source'] = 'local_fallback'
                    else:
                        st.error("Databricks serving failed, No local model")
                        st.session_state['prediction_source'] = 'failed'
            except Exception as e:
                if model is not None:
                    st.warning(f"Serving error: {e}, using local model")
                    preds[daytime] = np.maximum(0, model.predict(model_input))
                    st.session_state['prediction_source'] = 'local_fallback'
                else:
                    st.error("Serving error: No local model")
                    st.session_state['prediction_source'] = 'failed'
        elif model is not None:
            preds[daytime] = np.maximum(0, model.predict(model_input))
            st.session_state['prediction_source'] = 'local'
        else:
            st.error("Serving error: No local model")
            st.session_state['prediction_source'] = 'failed'
    df["uv_predicted"] = preds

    solar_scale = None
    if daytime.any():
        cos_z = df.loc[daytime, "cos_solar_zenith"].clip(lower=0)

        daily_peak_cos = (
            df[daytime].groupby(df[daytime]["timestamp"].dt.date)["cos_solar_zenith"]
            .transform("max")
            .replace(0, 1)
        )

        # Erythemal UV power law: UV ∝ cos(z)^n, n≈1.2
        # Source: Madronich (1993), doi:10.1201/9781351069847
        # n=1.2 accounts for increased atmospheric path through ozone at low sun
        solar_scale = (cos_z / daily_peak_cos) ** 1.2

        df.loc[daytime, "uv_predicted"] = df.loc[daytime, "uv_predicted"] * solar_scale

    # Prediction uncertainty (for tree-based ensemble models)
    if daytime.any():
        if hasattr(model, "estimators_"):
            tree_preds = np.array([
                tree.predict(X.values[daytime])
                for tree in model.estimators_
            ])
            df.loc[daytime, "uv_pred_q10"] = np.maximum(0, np.percentile(tree_preds, 10, axis=0))
            df.loc[daytime, "uv_pred_q90"] = np.maximum(0, np.percentile(tree_preds, 90, axis=0))
        else:
            uv_pred_day = preds[daytime]
            df.loc[daytime, "uv_pred_q10"] = uv_pred_day
            df.loc[daytime, "uv_pred_q90"] = uv_pred_day
        if solar_scale is not None:
            df.loc[daytime, "uv_pred_q10"] = df.loc[daytime, "uv_pred_q10"] * solar_scale
            df.loc[daytime, "uv_pred_q90"] = df.loc[daytime, "uv_pred_q90"] * solar_scale

    df["uv_pred_q10"] = df.get("uv_pred_q10", pd.Series(0.0, index=df.index)).fillna(0.0)
    df["uv_pred_q90"] = df.get("uv_pred_q90", pd.Series(0.0, index=df.index)).fillna(0.0)

    # Derive UV category from regression predictions via WHO standard bins
    df["uv_category"] = "Night"
    if daytime.any():
        uv_pred = df.loc[daytime, "uv_predicted"]
        df.loc[daytime, "uv_category"] = pd.cut(
            uv_pred,
            bins=UV_CATEGORY_BINS,
            labels=["Low", "Moderate", "High", "Very High", "Extreme"],
        ).astype(str).values

    return df


@st.cache_data(ttl=1800, show_spinner=False)
def get_live_forecast(
    forecast_days: int = 7,
    regression_model: str = "Random Forest",
    use_serving: bool = False,
) -> pd.DataFrame:
    from app.utils.data_loader import load_optimized_model
    databricks_client = DataBrickModelClient() if use_serving else None

    model = load_optimized_model(regression_model)
    if model is None and not use_serving:
        st.error(f"Không thể tải mô hình '{regression_model}'. Dự báo không khả dụng.")
        return pd.DataFrame()

    feat_cols = list(FINAL_22_FEATURES)
    ozone_means = _get_ozone_monthly_means()
    all_frames = []

    model_type = _get_model_type(regression_model)
    need_history = model_type in ("sequence", "prophet")

    for loc_id, loc in LOCATIONS.items():
        try:
            df = _fetch_combined_data(loc_id, loc, forecast_days, need_history)
        except Exception as e:
            st.warning(f"⚠️ Không thể tải dữ liệu cho {loc['name']}: {e}")
            continue

        df = _compute_solar(df, loc_id, loc)
        df = _compute_all_features(df, loc_id, loc, ozone_means)

        df = _predict(
            df, model, feat_cols,
            use_serving=use_serving,
            model_name=regression_model,
            databricks_client=databricks_client,
        )
        if need_history:
            # Trim to today's midnight so the full first day is always visible,
            # even when the page is viewed in the evening after UV hours have passed.
            cutoff_time = pd.Timestamp.now(tz=TIMEZONE).normalize().tz_localize(None)
            df = df[df["timestamp"] >= cutoff_time].copy()
        df["location_name"] = loc["name"]
        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    out = pd.concat(all_frames, ignore_index=True)
    out = out.sort_values(["location_id", "timestamp"]).reset_index(drop=True)
    return out
