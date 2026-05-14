from __future__ import annotations
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Nền tảng dự báo UV",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components import eda, forecast, model_results, recommendation, evaluation
from app.utils.data_loader import (
    LOCATION_NAMES,
    load_regression, load_regression_results,
    get_available_models,
)

from config import FINAL_FEATURES

SKIN_OPTIONS = {
    1: "1 - Rất trắng, dễ cháy nắng",
    2: "2 - Trắng, dễ cháy",
    3: "3 - Trung bình",
    4: "4 - Olive",
    5: "5 - Nâu",
    6: "6 - Nâu đậm, đen",
}



with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/sun.png", width=64)
    st.title("Dự báo UV")
    st.caption("Đồ án tốt nghiệp · Dự báo UV & Cảnh báo sức khỏe")
    st.divider()

    st.subheader("🗺️ Khu vực")
    all_locs = list(LOCATION_NAMES.keys())
    selected_locs = st.multiselect(
        "Chọn khu vực",
        options=all_locs,
        default=['hcm'],
        format_func=lambda x: LOCATION_NAMES.get(x, x),
    )
    if not selected_locs:
        selected_locs = all_locs

reg_results = load_regression_results()
df_preview  = load_regression()

test_results = reg_results[reg_results["split"] == "test"]
if not test_results.empty:
    best_row   = test_results.loc[test_results["r2"].idxmax()]
    best_model = best_row["model"]
    best_rmse  = best_row["rmse"]
    best_r2    = best_row["r2"]
else:
    best_model, best_rmse, best_r2 = "N/A", 0, 0

st.markdown("""
<div style='padding: 1rem 0 0.5rem 0;'>
  <h1 style='margin:0;'>☀️ Nền tảng Dự báo UV</h1>
  <p style='color:#666; margin-top:0.3rem;'>
    Dự báo UV · Cảnh báo sức khỏe · Gợi ý du lịch - Khu vực TP. Hồ Chí Minh
  </p>
</div>
""", unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Mô hình tốt nhất", best_model)
kpi2.metric("RMSE tốt nhất",  f"{best_rmse:.4f}")
kpi3.metric("R2 tốt nhất",    f"{best_r2:.4f}")
kpi4.metric("Khu vực",        f"{len(LOCATION_NAMES)}")
kpi5.metric("Dataset", f"{len(df_preview):,}")
st.divider()

tabs = [
    "🏠 Tổng quan",
    "📊 Phân tích dữ liệu",
    "🤖 Kết quả mô hình",
    "🔮 Dự báo (Trực tiếp)",
    "🌍 Gợi ý hoạt động",
    "🔬 Đánh giá hệ thống",
]
active_tab = st.radio("Chọn tính năng", tabs, horizontal=True, label_visibility="collapsed")

main_content = st.empty()

if active_tab == "🏠 Tổng quan":
    with main_content.container():
        st.header("Tổng quan dự án")
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.markdown(f"""
**Nền tảng Dự báo UV** là hệ thống AI toàn diện phục vụ:

1. **Dự báo chỉ số UV** bằng các mô hình Machine Learning truyền thống, Deep Learning và Hybrid.
2. **Phân loại mức nguy hiểm UV** theo danh mục WHO (Thấp -> Cực đoan) được suy ra trực tiếp
   từ giá trị UV dự báo liên tục bằng ngưỡng tiêu chuẩn WHO.
3. **Gợi ý hoạt động du lịch an toàn** dựa trên chỉ số UV, thời điểm trong ngày và điều kiện thời tiết.

**Khu vực nghiên cứu:** TP. Hồ Chí Minh + Vũng Tàu, Củ Chi, Nhà Bè, Thủ Đức, Cần Giờ, Long Hải.

**Nguồn dữ liệu:** Open-Meteo (thời tiết + UV), WeatherBit (quan trắc UV), Open-Meteo Air Quality (ozone, aerosol, PM2.5).
            """)
        with col_r:
            st.markdown("""
#### 🔑 Quyết định thiết kế chính
- **Regression-only inference:** Mô hình dự báo chỉ số UV liên tục
- **Danh mục UV** suy ra từ ngưỡng WHO chuẩn
- **22 đặc trưng tối ưu** qua phân tích đa cộng tuyến + Optuna tuning
- **Phân chia thời gian** (70/15/15) để tránh data leakage
- **Lọc ban đêm:** `solar_elevation <= 0 -> UV = 0`
- **Dự báo thực tế** qua Open-Meteo API (không dùng dữ liệu giả)
            """)

        st.subheader("Kiến trúc hệ thống & Dữ liệu")

        st.markdown(f"""
```text
[ Raw Data Sources ]
Open-Meteo (Weather, Forecast) --┐
WeatherBit (Realtime UV)     ----+---> [ Data Merger ]
OpenUV (Realtime & Safe Time) ---+      - Spatial-Temporal Join & Fallback (WB > OpenUV > OM)
Air Quality (Ozone, Aerosol) ----┘      - pvlib Solar Position (zenith, elevation, azimuth)
                                        - NOAA Heat Index & Stress Levels
                                                  |
                                                  v
                                       [ Feature Engineering ]
                                        - 22 Core Features (Temporal, Solar, Atmospheric, Lag)
                                        - Health Features (Fitzpatrick Safe Exposure, Risk)
                                        - Tourism Features (Beach & Outdoor Suitability)
                                                  |
                                                  v
                                      [ Model Inference ]
                                       {best_model} (R2={best_r2:.4f}, RMSE={best_rmse:.4f})
                                        - Dự báo UV liên tục
                                        - Lọc ban đêm (solar_elevation <= 0 -> UV = 0)
                                        - Phân loại mức độ WHO (Thấp -> Cực đoan)
                                                  |
                 ┌────────────────────────────────┼────────────────────────────────┐
                 v                                v                                v
        [ EDA & Results ]               [ 7-Day Forecast ]             [ Tourism Recommender ]
  - Phân tích tương quan           - API Trực tiếp (Realtime)     - Phân loại da (ScanSkinAI)
  - Biểu đồ đánh giá mô hình       - Cảnh báo chỉ số cực đoan     - Lọc địa điểm an toàn (Shade/Indoor)
  - Ma trận hệ số (Heatmap)        - Giao diện Card hiện đại      - Chấm điểm và Gợi ý tối ưu
```
    """)

        st.subheader("Chi tiết đặc trưng regression cốt lõi")
        st.info("Danh sách là các features dùng trực tiếp để train/inference regression. UV lag/rolling bị loại để tránh data leakage.")
        import pandas as pd
        feature_meta = {
            "cos_solar_zenith": ("1. Solar Geometry", "cos(zenith)", "Cosin góc thiên đỉnh, yếu tố vật lý quan trọng nhất của UV."),
            "doy_sin": ("1. Solar Geometry", "sin(2pi*doy/365)", "Mùa vụ theo chu kỳ năm."),
            "temperature_2m": ("2. Atmospheric Base", "raw", "Nhiệt độ không khí tại 2m."),
            "relative_humidity_2m": ("2. Atmospheric Base", "raw", "Độ ẩm tương đối tại 2m."),
            "cloud_cover": ("2. Atmospheric Base", "raw", "Độ che phủ mây tổng."),
            "solar_cloud_interaction": ("3. Interaction", "cos_solar_zenith*(1-cloud_cover)", "Tương tác vị trí mặt trời và mây"),
            "ozone_anomaly": ("2. Atmospheric Base", "ozone-monthly_baseline", "Độ lệch ozone so với trung bình tháng"),
            "pressure_msl": ("2. Atmospheric Base", "raw", "Áp suất mực nước biển"),
            "wind_speed_10m": ("2. Atmospheric Base", "raw", "Tốc độ gió 10m."),
            "altitude_m": ("4. Spatial", "location metadata", "Độ cao địa điểm."),
            "cos_zenith_squared": ("5. Nonlinear", "cos_solar_zenith^2", "Thành phần phi tuyến của góc thiên đỉnh."),
            "cloud_attenuation_exp": ("5. Nonlinear", "exp(-cloud_cover/100)", "Suy giảm UV theo mây dạng mũ"),
            "temp_humidity_product": ("3. Interaction", "temperature_2m*relative_humidity_2m", "Tương tác nhiệt độ-độ ẩm."),
            "pressure_cloud_interaction": ("3. Interaction", "pressure_msl*(1-cloud_cover/100)", "Tương tác áp suất và mây."),
            "temperature_2m_ema": ("6. Smoothed Trend", "EMA(alpha=0.3)", "Xu hương mượt nhiệt độ."),
            "cloud_cover_ema": ("6. Smoothed Trend", "EMA(alpha=0.3)", "Xu hướng mượt độ che mây."),
            "ozone_ema": ("6. Smoothed Trend", "EMA(alpha=0.3)", "Xu hướng mượt ozone."),
            "altitude_solar_interaction": ("3. Interaction", "altitude_m*cos_solar_zenith/1000", "Hiệu ứng độ cao theo vị trí mặt trời."),
            "ozone_depletion_risk": ("7. Binary Indicator", "(ozone<Q25).astype(int)", "Cờ rủi ro suy giảm ozone."),
            "air_quality_combined": ("7. Binary Indicator", "normalized air-quality blend", "Chỉ báo chất lượng không khí tổng hợp."),
            "is_raining": ("7. Binary Indicator", "(precipitation>0).astype(int)", "Cờ đang mưa."),
            "cloud_cover_change_1h": ("8. Momentum", "cloud_cover.diff(1)", "Biến thiên mây theo giờ."),
        }
        features_data = []
        for feat in FINAL_FEATURES:
            group, formula, meaning = feature_meta.get(feat, ("Unknown", "-", "Chua co mo ta"))
            features_data.append({"Nhom": group, "Dac Trung": feat, "Cong Thuc": formula, "Y nghia": meaning})
        df_features = pd.DataFrame(features_data)
        for group_name, group_df in df_features.groupby("Nhóm", sort=False):
            with st.expander(f"🔍 {group_name} ({len(group_df)} features)"):
                st.dataframe(group_df[["Đặc trưng", "Cong thuc", "Ý nghĩa"]], width='stretch', hide_index=True)
        st.caption(f"tong so feature dang dung: {len(FINAL_FEATURES)}")

elif active_tab == "📊 Phân tích dữ liệu":
    with main_content.container():
        with st.spinner("⏳ Đang tải dữ liệu..."):
            df_reg = load_regression()
            eda.render(df_reg, selected_locs)

elif active_tab == "🤖 Kết quả mô hình":
    with main_content.container():
        with st.spinner("⏳ Đang tải dữ liệu..."):
            model_results.render()

elif active_tab == "🔮 Dự báo (Trực tiếp)":
    with main_content.container():
        available_reg = get_available_models()
        if available_reg:
            col_model, col_source = st.columns([2, 2])
            with col_model:
                selected_regression_model = st.selectbox(
                    "Mô hình Regression",
                    options=list(available_reg.keys()),
                    key="forecast_model_selector",
                    help="Mô hình dùng để dự báo chỉ số UV liên tục (RMSE, R2 xem tab Kết quả mô hình)",
                )
            with st.spinner("⏳ Đang tải dữ liệu..."):
                forecast.render(selected_locs, selected_regression_model, use_serving=False)
        else:
            st.error("Không tìm thấy mô hình regression trong thư mục models/optimized/. Hãy train mô hình trước.")

elif active_tab == "🌍 Gợi ý hoạt động":
    with main_content.container():
        available_reg = get_available_models()
        if available_reg:
            col_m, col_s, col_d, col_r = st.columns([2, 3, 2, 2])
            with col_m:
                selected_rec_model = st.selectbox(
                    "Mô hình Regression",
                    options=list(available_reg.keys()),
                    key="rec_model_selector",
                    help="Mô hình dùng để dự báo UV cho gợi ý hoạt động",
                )
            with col_s:
                skin_label = st.radio(
                    "Loại da (Thang Fitzpatrick)",
                    options=list(SKIN_OPTIONS.values()),
                    index=2,
                    horizontal=True,
                    key="skin_type_radio",
                )
                skin_type = next(k for k, v in SKIN_OPTIONS.items() if v == skin_label)
            with col_d:
                activity_duration = st.slider(
                    "Thời gian hoạt động (phút)",
                    min_value=30, max_value=180, value=60, step=15,
                    key="activity_duration_slider",
                )
            with col_r:
                radius_km = st.slider(
                    "Bán kính tìm kiếm (km)",
                    min_value=10, max_value=50, value=30, step=5,
                    key="radius_km_slider",
                    help="Chỉ hiển thị địa điểm trong bán kính này tính từ khu vực đã chọn",
                )
            st.divider()
            with st.spinner("⏳ Đang tải dữ liệu..."):
                recommendation.render(selected_locs, skin_type, activity_duration, selected_rec_model, radius_km, use_serving=False)
        else:
            st.error("Không tìm thấy mô hình. Hãy train mô hình trước.")

elif active_tab == "🔬 Đánh giá hệ thống":
    with main_content.container():
        # st.info("PROCESSINGGGGGGGGGGGG")
        evaluation.render()
