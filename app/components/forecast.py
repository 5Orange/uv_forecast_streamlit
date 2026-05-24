from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils.data_loader import LOCATION_COLORS, LOCATION_NAMES
from app.utils.forecaster import get_live_forecast
from app.utils.databrick_client import get_serving_endpoint_status, format_serving_status

UV_BAND_COLORS = {
    "Low":       "rgba(39,174,96,0.15)",
    "Moderate":  "rgba(241,196,15,0.15)",
    "High":      "rgba(230,126,34,0.15)",
    "Very High": "rgba(231,76,60,0.15)",
    "Extreme":   "rgba(142,68,173,0.15)",
}

WHO_ADVICE_VI = {
    "Low":       "☀️ Không cần bảo vệ. Thoải mái hoạt động ngoài trời.",
    "Moderate":  "😎 Đeo kính râm. Dùng kem chống nắng SPF 30+ nếu ra ngoài > 30 phút.",
    "High":      "⚠️ Hạn chế phơi nắng giữa trưa. Nên dùng kem chống nắng, mũ và bóng râm.",
    "Very High": "🔴 Tránh ra ngoài 10-16 giờ. Cần bảo vệ UV đầy đủ.",
    "Extreme":   "🚨 Ở trong nhà vào giờ cao điểm. Phơi nắng ngoài trời rất nguy hiểm.",
}

CAT_VI = {
    "Low": "Thấp", "Moderate": "Trung bình",
    "High": "Cao", "Very High": "Rất cao", "Extreme": "Cực đoan",
}

CAT_COLORS = {
    "Low": "#27ae60", "Moderate": "#f1c40f", "High": "#e67e22",
    "Very High": "#e74c3c", "Extreme": "#8e44ad",
}

CAT_BG_COLORS = {
    "Low":       "rgba(39,174,96,0.12)",
    "Moderate":  "rgba(241,196,15,0.12)",
    "High":      "rgba(230,126,34,0.12)",
    "Very High": "rgba(231,76,60,0.12)",
    "Extreme":   "rgba(142,68,173,0.12)",
}

WEEKDAY_VI = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]


def _get_cat(uv_val: float) -> str:
    if uv_val <= 2: return "Low"
    if uv_val <= 5: return "Moderate"
    if uv_val <= 7: return "High"
    if uv_val <= 10: return "Very High"
    return "Extreme"


def _add_who_bands(fig: go.Figure) -> go.Figure:
    bands = [
        (0, 2,   "rgba(39,174,96,0.07)",   "Thấp (0-2)"),
        (2, 5,   "rgba(241,196,15,0.07)",  "Trung bình (3-5)"),
        (5, 7,   "rgba(230,126,34,0.07)",  "Cao (6-7)"),
        (7, 10,  "rgba(231,76,60,0.07)",   "Rất cao (8-10)"),
        (10, 14, "rgba(142,68,173,0.07)",  "Cực đoan (11+)"),
    ]
    for y0, y1, color, label in bands:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0,
                      annotation_text=label, annotation_position="right",
                      annotation_font_size=10)
    return fig


def _add_night_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Replace UV=0 nighttime rows with NaN so Plotly breaks the line
    at night instead of drawing a diagonal connector across the gap."""
    out = df.copy()
    out.loc[out["uv_predicted"] == 0, "uv_predicted"] = float("nan")
    return out


def _uv_icon(color: str) -> str:
    """Return an inline SVG sun icon colored by the UV category."""
    return f"""
    <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="5" fill="{color}"></circle>
      <line x1="12" y1="1" x2="12" y2="3"></line>
      <line x1="12" y1="21" x2="12" y2="23"></line>
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
      <line x1="1" y1="12" x2="3" y2="12"></line>
      <line x1="21" y1="12" x2="23" y2="12"></line>
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
      <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
    </svg>
    """


def _inject_card_css():
    """Inject CSS once to style forecast cards."""
    st.markdown("""
    <style>
    /* Hover effect triggered by the column button */
    div[data-testid="column"]:has(button:hover) .fc-card,
    div[data-testid="stColumn"]:has(button:hover) .fc-card {
        transform: translateY(-3px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    }
    .fc-card {
        height: 100%;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: space-between;
        background: #ffffff;
        border: 2px solid #e8ecf0;
        border-radius: 14px;
        padding: 14px 8px 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        transition: transform 0.15s, box-shadow 0.15s, border-color 0.15s;
    }
    /* Ensure the card content does not block clicks meant for the overlay button */
    .fc-card {
        pointer-events: none !important;
    }
    .fc-card * {
        pointer-events: none !important;
    }
    .fc-card.selected {
        border-width: 2.5px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .fc-card-day { font-size: 0.75em; font-weight: 600; color: #555; margin-bottom: 2px; }
    .fc-card-date { font-size: 0.65em; color: #999; margin-bottom: 10px; }
    .fc-card-icon { margin-bottom: 8px; line-height: 1; }
    .fc-uv-badge {
        border-radius: 8px;
        padding: 6px 4px 4px;
        margin-bottom: 10px;
        width: 100%;
        box-sizing: border-box;
    }
    .fc-uv-peak {
        font-size: 1.4em;
        font-weight: 800;
        line-height: 1;
        display: block;
    }
    .fc-uv-label {
        font-size: 0.62em;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        display: block;
        margin-top: 2px;
    }
    .fc-card-temp {
        font-size: 0.75em;
        font-weight: 500;
        color: #444;
    }
    .fc-card-temp .temp-max { color: #e74c3c; }
    .fc-card-temp .temp-min { color: #3498db; }
    </style>
    """, unsafe_allow_html=True)


def _render_day_card(
    date,
    peak_uv: float,
    cat_en: str,
    temp_min: float,
    temp_max: float,
    is_selected: bool,
    card_id: str,
) -> str:
    """Return the HTML string for a single forecast day card."""
    cat_vi = CAT_VI.get(cat_en, cat_en)
    badge_color = CAT_COLORS.get(cat_en, "#888")
    badge_bg = CAT_BG_COLORS.get(cat_en, "#f5f5f5")
    icon = _uv_icon(badge_color)
    day_name = WEEKDAY_VI[pd.Timestamp(date).weekday()]
    date_str = date.strftime("%d/%m")

    selected_cls = "selected" if is_selected else ""
    border_color = badge_color if is_selected else "#e8ecf0"

    return f"""
    <div class="fc-card {selected_cls}" id="{card_id}"
         style="border-color:{border_color};">
        <div>
            <div class="fc-card-day">{day_name}</div>
            <div class="fc-card-date">{date_str}</div>
        </div>
        <div class="fc-card-icon">{icon}</div>
        <div class="fc-uv-badge" style="background:{badge_bg};">
            <span class="fc-uv-peak" style="color:{badge_color};">{peak_uv:.1f}</span>
            <span class="fc-uv-label" style="color:{badge_color};">{cat_vi}</span>
        </div>
        <div class="fc-card-temp">
            <span class="temp-min">{temp_min:.0f}°</span>
            &nbsp;/&nbsp;
            <span class="temp-max">{temp_max:.0f}°</span>
        </div>
    </div>
    """


def _set_selected_date(date):
    """Callback to update selected date before rendering."""
    st.session_state["fc_selected_date"] = date

@st.fragment
def _render_cards_and_detail(data: pd.DataFrame, dates: list, locs_in_fc: list, detail_loc: str):
    """Fragment: card grid + inline detail panel.
    Only this section reruns when a card button is clicked.
    """
    # Initialise selected date in session state
    if "fc_selected_date" not in st.session_state or st.session_state["fc_selected_date"] not in dates:
        st.session_state["fc_selected_date"] = dates[0] if dates else None

    _inject_card_css()

    # Pre-compute per-day summaries for the selected location
    day_summaries: list[dict] = []
    for date in dates:
        day_data = data[(data["timestamp"].dt.date == date) & (data["location_id"] == detail_loc)]
        daytime = day_data[day_data["uv_predicted"] > 0]
        peak_uv = daytime["uv_predicted"].max() if not daytime.empty else 0.0
        cat_en = _get_cat(peak_uv)
        temp_min = day_data["temperature_2m"].min() if "temperature_2m" in day_data.columns else 0.0
        temp_max = day_data["temperature_2m"].max() if "temperature_2m" in day_data.columns else 0.0
        avg_cloud = day_data["cloud_cover"].mean() if "cloud_cover" in day_data.columns else 0.0
        total_rain = day_data["precipitation"].sum() if "precipitation" in day_data.columns else 0.0
        day_summaries.append(dict(
            date=date, peak_uv=peak_uv, cat_en=cat_en,
            temp_min=temp_min, temp_max=temp_max,
            avg_cloud=avg_cloud, total_rain=total_rain,
        ))

    # ── Card grid ────────────────────────────────────────────────────────────
    st.markdown('<div class="fc-grid-anchor"></div>', unsafe_allow_html=True)
    card_cols = st.columns(len(dates))
    for i, (s, col) in enumerate(zip(day_summaries, card_cols)):
        is_selected = (st.session_state["fc_selected_date"] == s["date"])
        with col:
            st.markdown(
                _render_day_card(
                    date=s["date"],
                    peak_uv=s["peak_uv"],
                    cat_en=s["cat_en"],
                    temp_min=s["temp_min"],
                    temp_max=s["temp_max"],
                    is_selected=is_selected,
                    card_id=f"fc-card-{i}",
                ),
                unsafe_allow_html=True,
            )
            # Visible button below the card
            button_type = "primary" if is_selected else "secondary"
            st.button("Chi tiết", key=f"fc_card_btn_{i}", 
                      on_click=_set_selected_date, args=(s["date"],),
                      help=f"Xem chi tiết ngày {s['date'].strftime('%d/%m')}",
                      width='stretch', type=button_type)

    # ── Inline detail panel ──────────────────────────────────────────────────
    selected_date = st.session_state.get("fc_selected_date")
    if selected_date is not None:
        sel_summary = next((s for s in day_summaries if s["date"] == selected_date), None)
        if sel_summary:
            cat_en = sel_summary["cat_en"]
            cat_vi = CAT_VI.get(cat_en, cat_en)
            cat_color = CAT_COLORS.get(cat_en, "#888")
            badge_bg = CAT_BG_COLORS.get(cat_en, "#f5f5f5")
            peak_uv = sel_summary["peak_uv"]
            day_name = WEEKDAY_VI[pd.Timestamp(selected_date).weekday()]

            day_data = data[
                (data["timestamp"].dt.date == selected_date)
                & (data["location_id"] == detail_loc)
            ]
            daytime = day_data[day_data["uv_predicted"] > 0]

            st.markdown(
                f"""
                <div style="
                    background: {badge_bg};
                    border-left: 4px solid {cat_color};
                    border-radius: 10px;
                    padding: 14px 18px 6px;
                    margin: 8px 0 4px 0;
                ">
                    <span style="font-size:0.85em;font-weight:600;color:{cat_color};">
                        📅 {day_name} {selected_date.strftime('%d/%m/%Y')}
                        &nbsp;·&nbsp; UV đỉnh: {peak_uv:.1f} ({cat_vi})
                        &nbsp;·&nbsp; {sel_summary['temp_min']:.0f}–{sel_summary['temp_max']:.0f}°C
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_chart, col_info = st.columns([3, 1])
            with col_chart:
                fig_day = go.Figure()
                if not daytime.empty:
                    fig_day.add_trace(go.Scatter(
                        x=daytime["timestamp"], y=daytime["uv_predicted"],
                        mode="lines+markers",
                        name="UV dự báo",
                        line=dict(color=cat_color, width=2.5),
                        marker=dict(size=5),
                    ))
                    if "uv_pred_q10" in daytime.columns:
                        fig_day.add_trace(go.Scatter(
                            x=pd.concat([daytime["timestamp"], daytime["timestamp"][::-1]]),
                            y=pd.concat([daytime["uv_pred_q90"], daytime["uv_pred_q10"][::-1]]),
                            fill="toself",
                            fillcolor=f"rgba({','.join(str(int(x)) for x in _hex_to_rgb(cat_color))},0.12)",
                            line=dict(width=0),
                            name="Khoảng tin cậy (10–90%)",
                            showlegend=True,
                        ))
                _add_who_bands(fig_day)
                fig_day.update_layout(
                    title=f"UV Index theo giờ — {LOCATION_NAMES.get(detail_loc, detail_loc)}",
                    xaxis_title="Giờ", yaxis_title="UV Index",
                    legend_title="Chú thích",
                    height=300, margin=dict(t=40, b=30),
                )
                idx = dates.index(selected_date)
                st.plotly_chart(fig_day, width='stretch', key=f"daily_uv_chart_{idx}_{detail_loc}")

            with col_info:
                st.markdown(
                    f"<div style='background:{cat_color};color:white;padding:12px;border-radius:8px;"
                    f"text-align:center;font-size:1.1em;margin-bottom:8px'>"
                    f"<b>{cat_vi}</b><br>UV đỉnh: {peak_uv:.1f}</div>",
                    unsafe_allow_html=True,
                )
                advice = WHO_ADVICE_VI.get(cat_en, "")
                st.info(advice)
                st.caption(f"☁️ Mây TB: {sel_summary['avg_cloud']:.0f}%")
                if sel_summary["total_rain"] > 0:
                    st.caption(f"🌧️ Mưa: {sel_summary['total_rain']:.1f} mm")



def render(selected_locs: list[str], regression_model: str = "Random Forest", use_serving: bool = False):
    st.header("🔮 Dự báo UV 7 ngày (Trực tiếp)")
    st.caption(
        f"Dự báo thời gian thực bằng **{regression_model}** "
    )
    if use_serving:
        status = get_serving_endpoint_status(regression_model)
        status_str = format_serving_status(status)
        st.info(f"*Serving Endpoint status* {status_str}")

    col_btn, col_info = st.columns([1, 5])
    with col_btn:
        if st.button("🔄 Làm mới dự báo", type="primary"):
            get_live_forecast.clear()
            st.rerun()
    # with col_info:
    #     st.caption("Dự báo được cache 30 phút")

    fc = get_live_forecast(
        forecast_days=7,
        regression_model=regression_model,
        use_serving=use_serving
    )

    if fc.empty:
        st.error("Không thể tải dữ liệu dự báo. Kiểm tra kết nối mạng.")
        return

    locs_in_fc = [l for l in selected_locs if l in fc["location_id"].unique()]
    if not locs_in_fc:
        locs_in_fc = list(fc["location_id"].unique())
    data = fc[fc["location_id"].isin(locs_in_fc)].copy()
    data["location_name"] = data["location_id"].map(LOCATION_NAMES)

    # ── Daily forecast section ─────────────────────────────────────────────
    st.subheader("📅 Dự báo theo ngày")

    # Location selector must stay outside the fragment so detail_loc is
    # determined before the fragment runs (fragments can't return values).
    detail_loc = st.selectbox(
        "Khu vực xem chi tiết",
        locs_in_fc,
        format_func=lambda x: LOCATION_NAMES.get(x, x),
        key="fc_detail_loc",
    )

    dates = sorted(data["timestamp"].dt.date.unique())[:7]

    # ── Fragment: card grid + inline detail (partial rerun only) ────────────
    _render_cards_and_detail(data, dates, locs_in_fc, detail_loc)

    st.divider()

    # ── 3. Overview chart — all locations ──────────────────────────────────
    st.subheader("📈 Tổng quan UV 7 ngày — Tất cả khu vực")
    fc_gapped = _add_night_gaps(data)
    fig_fc = px.line(
        fc_gapped,
        x="timestamp", y="uv_predicted", color="location_name",
        color_discrete_map={v: LOCATION_COLORS[k] for k, v in LOCATION_NAMES.items()},
        labels={"timestamp": "Ngày & Giờ", "uv_predicted": "UV Index dự báo",
                "location_name": "Khu vực"},
        title="Dự báo UV theo giờ — 7 ngày",
    )
    _add_who_bands(fig_fc)
    fig_fc.update_layout(hovermode="x unified", legend_title="Chú thích")
    st.plotly_chart(fig_fc, width='stretch', key="forecast_overview_chart")

    # ── 4. Weather context ─────────────────────────────────────────────────
    st.subheader("🌤️ Bối cảnh thời tiết")
    weath_loc = st.selectbox(
        "Khu vực", locs_in_fc,
        format_func=lambda x: LOCATION_NAMES.get(x, x),
        key="fc_weath_loc",
    )
    wdta = data[data["location_id"] == weath_loc].sort_values("timestamp")

    col_t, col_c = st.columns(2)
    with col_t:
        fig_temp = px.line(
            wdta, x="timestamp", y="temperature_2m",
            color_discrete_sequence=["#e74c3c"],
            labels={"temperature_2m": "Nhiệt độ (°C)", "timestamp": "Thời gian"},
            title="Dự báo nhiệt độ",
        )
        st.plotly_chart(fig_temp, width='stretch', key="forecast_temp_chart")
    with col_c:
        fig_cloud = px.area(
            wdta, x="timestamp", y="cloud_cover",
            color_discrete_sequence=["#95a5a6"],
            labels={"cloud_cover": "Độ che phủ mây (%)", "timestamp": "Thời gian"},
            title="Dự báo mây che phủ",
        )
        st.plotly_chart(fig_cloud, width='stretch', key="forecast_cloud_chart")

    # ── 5. Raw data table ──────────────────────────────────────────────────
    with st.expander("📋 Xem dữ liệu dự báo thô"):
        show_cols = ["timestamp", "location_name", "uv_predicted", "uv_category",
                     "temperature_2m", "cloud_cover", "precipitation", "solar_elevation"]
        st.dataframe(
            data[[c for c in show_cols if c in data.columns]]
              .sort_values(["location_name", "timestamp"])
              .rename(columns={
                  "uv_predicted":   "UV Dự báo",
                  "uv_category":    "Mức nguy hiểm",
                  "temperature_2m": "Nhiệt độ (°C)",
                  "cloud_cover":    "Mây (%)",
                  "solar_elevation":"Góc mặt trời (°)",
              }),
            width='stretch', hide_index=True,
        )


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert #RRGGBB to (R, G, B) integers."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
