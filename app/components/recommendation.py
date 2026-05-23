"""Cảnh báo sức khỏe & Gợi ý du lịch - với bản đồ tương tác (click để chọn vị trí)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium
    _FOLIUM_AVAILABLE = True
except ImportError:
    _FOLIUM_AVAILABLE = False

from app.utils.data_loader import LOCATION_NAMES
from app.utils.forecaster import get_live_forecast
from config import STATIC_DIR, LOCATIONS

# -- Constants -----------------------------------------------------------------
from src.recommendation.safe_time_policy import MED_VALUES_JM2, SKIN_TYPE_MULTIPLIER

WHO_CAT_COLORS = {
    "Low": "#27ae60", "Moderate": "#f1c40f", "High": "#e67e22",
    "Very High": "#e74c3c", "Extreme": "#8e44ad", "Night": "#34495e",
}

WHO_ADVICE = {
    "Low":       "Không cần bảo vệ. Thoải mái hoạt động ngoài trời.",
    "Moderate":  "Đeo kính râm. Dùng kem chống nắng SPF 30+ nếu ra ngoài > 30 phút.",
    "High":      "Hạn chế phơi nắng giữa trưa. Nên dùng kem chống nắng, mũ và bóng râm.",
    "Very High": "Tránh ra ngoài 10 giờ-16 giờ. Cần bảo vệ UV đầy đủ.",
    "Extreme":   "Ở trong nhà vào giờ cao điểm. Phơi nắng ngoài trời rất nguy hiểm.",
}

_UV_SAFE_COLORS = [
    (70, "green"),
    (40, "orange"),
    (20, "red"),
    (0,  "darkred"),
]


# -- Helpers -------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points in kilometres."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _find_nearest_station(lat: float, lon: float) -> tuple[str, float]:
    """Find the nearest UV weather station to a given lat/lon.
    
    Returns (station_id, distance_km).
    """
    best_id, best_dist = None, float("inf")
    for loc_id, loc in LOCATIONS.items():
        dist = haversine_km(lat, lon, loc["lat"], loc["lon"])
        if dist < best_dist:
            best_id, best_dist = loc_id, dist
    return best_id, best_dist


def _load_places() -> list[dict]:
    path = STATIC_DIR / "suggest_location.json"

    if not path.exists():
        return []
    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f).get("TOURIST_PLACES", [])


def _compute_safe_minutes(effective_uv: pd.Series, skin_type: int) -> pd.Series:
    med_value = MED_VALUES_JM2[skin_type]
    # Cap at 480 min (8h) for negligible UV; avoid division by near-zero
    safe = np.where(
        effective_uv <= 0.01,
        480.0,
        np.minimum(480.0, med_value / (effective_uv * 1.5))
    )
    return pd.Series(safe, index=effective_uv.index)


def _filter_nearby_places(
    places: list[dict],
    center_lat: float,
    center_lon: float,
    radius_km: float,
) -> list[dict]:
    """Return places within radius_km of a center point, sorted nearest-first."""
    nearby: list[dict] = []
    for place in places:
        try:
            dist = haversine_km(center_lat, center_lon, place["lat"], place["lon"])
        except (KeyError, TypeError):
            continue
        if dist <= radius_km:
            nearby.append({**place, "distance_km": round(dist, 1)})
    return sorted(nearby, key=lambda x: x["distance_km"])


def _get_marker_color(safe_pct: float) -> str:
    for threshold, color in _UV_SAFE_COLORS:
        if safe_pct >= threshold:
            return color
    return "darkred"


def _score_places(
    fc_loc: pd.DataFrame,
    places: list[dict],
    skin_type: int,
    activity_min: float,
    regression_model: str = "",
) -> list[dict]:
    """Score and rank nearby places."""
    results = []
    for place in places:
        if place["location_key"] not in fc_loc["location_id"].values:
            continue
        loc_fc = fc_loc[fc_loc["location_id"] == place["location_key"]].copy()
        loc_fc = loc_fc[loc_fc["uv_predicted"] > 0]
        if loc_fc.empty:
            continue

        # 1. Calculate Effective UVI (Indoor / Shade Attenuation)
        # Indoor places: standard glass transmits ~5% of erythemal UV (Tuchinda et al. 2006)
        # Shade: Tree canopy UPF ~3.3 transmits ~30% UV (Parisi et al. 1999)
        shade_pct = place.get("shade_coverage_pct", 50)
        is_indoor = place.get("indoor_option", False)

        if is_indoor:
            effective_uv = loc_fc["uv_predicted"] * 0.05
        else:
            shade_ratio = shade_pct / 100.0
            # Open area gets 100% UV, shaded area gets 30% UV
            transmission = (1.0 - shade_ratio) + (shade_ratio * 0.3)
            effective_uv = loc_fc["uv_predicted"] * transmission

        # 2. Calculate Safe Minutes using biological MED formula
        loc_fc["safe_minutes"] = _compute_safe_minutes(effective_uv, skin_type)
        
        # 3. Calculate Base Safe Ratio
        loc_fc["safe_ratio"] = (loc_fc["safe_minutes"] / max(1.0, activity_min)).clip(upper=1.0)

        # 4. Apply Thermal/Comfort Stressors (ISO 7243 WBGT limit)
        # If outdoor and hot (>= 35C) without full shade, scale down safety due to heat strain
        avg_temp = loc_fc["temperature"].mean() if "temperature" in loc_fc.columns else 30
        is_raining = False  # Forecast rain status is not easily accessible here without more data, evaluation handles rain

        thermal_modifier = 1.0
        if not is_indoor and avg_temp >= 35.0:
            thermal_modifier = 0.5  # Severe heat stress multiplier

        # Hours that are fully safe
        safe_hours = loc_fc[loc_fc["safe_ratio"] >= 1.0]
        if not safe_hours.empty:
            best_start = safe_hours["timestamp"].iloc[0]
            best_end   = safe_hours["timestamp"].iloc[-1]
            avg_uv     = safe_hours["uv_predicted"].mean()
        else:
            best_start = best_end = None
            avg_uv = loc_fc["uv_predicted"].mean()

        avg_safe_ratio = float(loc_fc["safe_ratio"].mean())
        score = avg_safe_ratio * thermal_modifier

        results.append({
            "name":           place["name"],
            "location_key":   place["location_key"],
            "type":           place["type"],
            "lat":            place.get("lat"),
            "lon":            place.get("lon"),
            "distance_km":    place.get("distance_km"),
            "score":          round(score, 3),
            "safe_hours_pct": round(avg_safe_ratio * 100, 1),
            "avg_uv":         round(avg_uv, 1),
            "best_start":     best_start,
            "best_end":       best_end,
            "has_shade":      place.get("has_shade", False),
            "indoor_option":  place.get("indoor_option", False),
            "maps_url":       place.get("maps_url", ""),
            "activities":     place.get("activities", []),
            "region_type":    place.get("region_type", ""),
            "image_url":      place.get("image_url", ""),
            "model_used":     regression_model,
        })
    return sorted(results, key=lambda x: x["score"], reverse=True)


def _handle_map_click(map_data: dict | None) -> tuple[float, float, str] | None:
    """Persist a Folium click and return the selected point with nearest station."""
    if not map_data or not map_data.get("last_clicked"):
        return None

    clicked = map_data["last_clicked"]
    lat = clicked["lat"]
    lon = clicked["lng"]

    old_lat = st.session_state.get("rec_click_lat")
    old_lon = st.session_state.get("rec_click_lon")
    is_new_click = (
        old_lat is None
        or abs(lat - old_lat) > 1e-6
        or abs(lon - old_lon) > 1e-6
    )

    st.session_state["rec_click_lat"] = lat
    st.session_state["rec_click_lon"] = lon
    station_id, dist = _find_nearest_station(lat, lon)
    st.session_state["rec_nearest_station"] = station_id
    st.session_state["rec_nearest_dist"] = dist

    if is_new_click:
        st.rerun()

    return lat, lon, station_id


def _get_previous_map_selection() -> tuple[float, float, str] | None:
    if "rec_click_lat" not in st.session_state or "rec_nearest_station" not in st.session_state:
        return None
    return (
        st.session_state["rec_click_lat"],
        st.session_state["rec_click_lon"],
        st.session_state["rec_nearest_station"],
    )


def _render_recommendation_map(
    center_lat: float | None = None,
    center_lon: float | None = None,
    station_id: str | None = None,
    scored: list[dict] | None = None,
    radius_km: float | None = None,
    *,
    allow_click: bool = True,
) -> tuple[float, float, str] | None:
    """Render the single tourism map used for both picking and results."""
    if not _FOLIUM_AVAILABLE:
        st.warning("📦 Cài đặt `folium` và `streamlit-folium` để xem bản đồ tương tác.")
        return None

    has_selection = center_lat is not None and center_lon is not None
    map_lat = center_lat if center_lat is not None else st.session_state.get("rec_click_lat", 10.7769)
    map_lon = center_lon if center_lon is not None else st.session_state.get("rec_click_lon", 106.7009)
    station = LOCATIONS.get(station_id, {}) if station_id else {}
    station_name = LOCATION_NAMES.get(station_id, station_id) if station_id else ""

    m = folium.Map(
        location=[map_lat, map_lon],
        zoom_start=12 if has_selection else 10,
        tiles="CartoDB positron",
    )

    if has_selection and radius_km is not None:
        folium.Circle(
            location=[center_lat, center_lon],
            radius=radius_km * 1000,
            color="#3498db",
            fill=True, fill_color="#3498db", fill_opacity=0.06,
            weight=1.5,
            tooltip=f"Bán kính tìm kiếm: {radius_km:.0f} km",
        ).add_to(m)

    if has_selection:
        folium.Marker(
            location=[center_lat, center_lon],
            tooltip="📍 Vị trí bạn chọn",
            icon=folium.Icon(color="red", icon="map-pin", prefix="fa"),
        ).add_to(m)

    if station:
        folium.Marker(
            location=[station["lat"], station["lon"]],
            tooltip=f"📡 Trạm UV: {station_name}",
            popup=folium.Popup(
                f"<b>{station_name}</b><br>Trạm UV gần nhất",
                max_width=200,
            ),
            icon=folium.Icon(color="blue", icon="cloud", prefix="fa"),
        ).add_to(m)
    elif not has_selection:
        for loc_id, loc in LOCATIONS.items():
            name = LOCATION_NAMES.get(loc_id, loc_id)
            folium.Marker(
                location=[loc["lat"], loc["lon"]],
                tooltip=f"📡 Trạm UV: {name}",
                icon=folium.Icon(color="blue", icon="cloud", prefix="fa"),
            ).add_to(m)

    for p in scored or []:
        if p.get("lat") is None or p.get("lon") is None:
            continue
        color = _get_marker_color(p["safe_hours_pct"])
        best_window = "-"
        if p["best_start"] and p["best_end"]:
            best_window = (
                f"{p['best_start'].strftime('%H:%M')} - {p['best_end'].strftime('%H:%M')}"
            )
        dist_str = f"{p['distance_km']} km" if p.get("distance_km") is not None else "N/A"
        activities_str = ", ".join(p["activities"])
        shade_badge = "✅ Bóng râm" if p["has_shade"] else "☀️ Ngoài trời"
        indoor_badge = " · 🏠 Trong nhà" if p["indoor_option"] else ""

        popup_html = f"""
        <div style='font-family:sans-serif;min-width:200px'>
          <b style='font-size:14px'>{p['name']}</b><br>
          <span style='color:#666'>{p['type'].replace('_',' ').title()}</span><br>
          <hr style='margin:4px 0'>
          📍 Cách bạn: <b>{dist_str}</b><br>
          ☀️ UV trung bình: <b>{p['avg_uv']}</b><br>
          🕐 Giờ an toàn: <b>{p['safe_hours_pct']}%</b><br>
          ⏰ Khung tốt nhất: <b>{best_window}</b><br>
          🎯 Hoạt động: {activities_str}<br>
          {shade_badge}{indoor_badge}<br>
          🤖 Mô hình: <i>{p.get('model_used','')}</i>
        </div>
        """
        folium.Marker(
            location=[p["lat"], p["lon"]],
            tooltip=f"{p['name']} ({p['safe_hours_pct']}% an toàn)",
            popup=folium.Popup(popup_html, max_width=260),
            icon=folium.Icon(color=color, icon="map-marker", prefix="fa"),
        ).add_to(m)

    map_data = st_folium(
        m,
        width='stretch',
        height=450,
        returned_objects=["last_clicked"] if allow_click else [],
        key="rec_recommendation_map",
    )
    return _handle_map_click(map_data) if allow_click else None


# -- Main render ---------------------------------------------------------------

def render(
    selected_locs: list[str],
    skin_type: int = 3,
    activity_duration: int = 60,
    regression_model: str = "Random Forest",
    radius_km: float = 30.0,
    use_serving:bool = False,
):
    st.header("🌍 Cảnh báo sức khỏe & Gợi ý du lịch")
    st.caption(
        f"Gợi ý cá nhân hoá bằng **{regression_model}** - "
        f"loại da {skin_type}, thời gian hoạt động {activity_duration} phút, "
        f"bán kính **{radius_km:.0f} km**, dự báo trực tiếp."
    )

    fc = get_live_forecast(forecast_days=7, regression_model=regression_model, use_serving=use_serving)
    if fc.empty:
        st.error("Không có dữ liệu dự báo.")
        return

    places = _load_places()
    today  = fc[fc["timestamp"].dt.date == fc["timestamp"].dt.date.min()]

    # -- Location selection: map click OR dropdown --------------------------
    st.subheader("📍 Chọn vị trí của bạn")
    
    input_method = st.radio(
        "Cách chọn vị trí",
        ["🗺️ Chọn trên bản đồ", "📋 Chọn từ danh sách"],
        horizontal=True,
        key="rec_input_method",
    )
    map_slot = st.empty()

    center_lat, center_lon, loc_sel = None, None, None

    if input_method == "🗺️ Chọn trên bản đồ":
        st.caption("👆 Nhấp vào bản đồ để chọn vị trí. Hệ thống sẽ tự động tìm trạm UV gần nhất.")
        click_result = _get_previous_map_selection()

        if not click_result:
            with map_slot.container():
                click_result = _render_recommendation_map(allow_click=True)
        
        if click_result:
            center_lat, center_lon, loc_sel = click_result
            station_dist = st.session_state.get("rec_nearest_dist", 0)

            if station_dist > 50:
                with map_slot.container():
                    _render_recommendation_map(center_lat, center_lon, loc_sel, radius_km=radius_km)
                st.warning(
                    f"⚠️ Vị trí bạn chọn cách trạm UV gần nhất ({LOCATION_NAMES.get(loc_sel, loc_sel)}) {station_dist:.1f} km. "
                    "Vui lòng chọn vị trí trong bán kính 50km so với trạm UV để đảm bảo độ chính xác."
                )
                return

            st.success(
                f"📍 Vị trí: ({center_lat:.4f}, {center_lon:.4f}) · "
                f"📡 Trạm UV gần nhất: **{LOCATION_NAMES.get(loc_sel, loc_sel)}** "
                f"(cách {station_dist:.1f} km)"
            )
        else:
            st.info("👆 Nhấp vào bản đồ để chọn vị trí của bạn.")
            return
    else:
        locs_in_fc = [l for l in selected_locs if l in fc["location_id"].unique()]
        if not locs_in_fc:
            locs_in_fc = list(fc["location_id"].unique())
        loc_sel = st.selectbox(
            "Khu vực",
            locs_in_fc,
            format_func=lambda x: LOCATION_NAMES.get(x, x),
            key="rec_loc_dropdown",
        )
        loc_cfg = LOCATIONS.get(loc_sel, {})
        center_lat = loc_cfg.get("lat", 10.7769)
        center_lon = loc_cfg.get("lon", 106.7009)

    # -- UV risk metrics ----------------------------------------------------
    st.subheader("📍 Mức rủi ro UV hiện tại")
    
    # Show UV for the selected station
    loc_today = today[today["location_id"] == loc_sel]
    if not loc_today.empty:
        peak_uv  = loc_today["uv_predicted"].max()
        peak_cat = loc_today.loc[loc_today["uv_predicted"].idxmax(), "uv_category"]
        multiplier = SKIN_TYPE_MULTIPLIER[skin_type]
        safe_min = (200.0 * multiplier) / (3.0 * max(1.0, peak_uv))
        
        col_uv, col_safe, col_advice = st.columns(3)
        with col_uv:
            st.metric(
                LOCATION_NAMES.get(loc_sel, loc_sel),
                f"UV {peak_uv:.1f}",
                delta=str(peak_cat),
                delta_color="off",
            )
        with col_safe:
            st.metric("Thời gian an toàn", f"{safe_min:.0f} phút", delta=f"Loại da {skin_type}", delta_color="off")
        with col_advice:
            advice = WHO_ADVICE.get(peak_cat, "")
            if peak_cat in ["Very High", "Extreme"]:
                st.error(advice)
            elif peak_cat == "High":
                st.warning(advice)
            else:
                st.success(advice)

    # -- Safe time chart ----------------------------------------------------
    st.subheader("🕐 Khung giờ an toàn hôm nay")
    loc_today_chart = today[today["location_id"] == loc_sel].copy()
    if not loc_today_chart.empty:
        loc_today_chart["safe_minutes"] = _compute_safe_minutes(loc_today_chart["uv_predicted"], skin_type)
        loc_today_chart["is_safe"] = loc_today_chart["safe_minutes"] >= activity_duration

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=loc_today_chart["timestamp"], y=loc_today_chart["uv_predicted"],
            mode="lines+markers", name="UV Dự báo",
            line={"color": "#e74c3c", "width": 2},
        ))
        safe_rows = loc_today_chart[loc_today_chart["is_safe"]]
        if not safe_rows.empty:
            fig.add_trace(go.Scatter(
                x=safe_rows["timestamp"], y=safe_rows["uv_predicted"],
                mode="markers", name=f"An toàn (≥{activity_duration} phút)",
                marker={"color": "#27ae60", "size": 10, "symbol": "circle"},
            ))
        fig.update_layout(
            title=f"UV - {LOCATION_NAMES.get(loc_sel, loc_sel)} (khung giờ an toàn được đánh dấu)",
            xaxis_title="Thời gian", yaxis_title="UV Index",
            legend_title="Chú thích",
        )
        st.plotly_chart(fig, width='stretch')

    st.subheader("🗺️ Địa điểm được gợi ý")
    if not places:
        st.info("Không có dữ liệu địa điểm du lịch.")
        return

    # Legend chips
    col_leg = st.columns(4)
    for col, (label, color) in zip(col_leg, [
        ("≥70% an toàn", "#27ae60"), ("40-70%", "#e67e22"),
        ("20-40%", "#e74c3c"), ("<20%", "#922b21"),
    ]):
        col.markdown(
            f"<span style='background:{color};color:white;padding:2px 8px;"
            f"border-radius:4px;font-size:12px'>⬤ {label}</span>",
            unsafe_allow_html=True,
        )

    # Filter places by proximity to user's selected point
    nearby = _filter_nearby_places(places, center_lat, center_lon, radius_km)

    if not nearby:
        with map_slot.container():
            _render_recommendation_map(
                center_lat,
                center_lon,
                loc_sel,
                scored=[],
                radius_km=radius_km,
                allow_click=input_method == "🗺️ Chọn trên bản đồ",
            )
        st.info(
            f"Không có địa điểm du lịch trong bán kính **{radius_km:.0f} km** "
            f"tính từ vị trí đã chọn. Thử mở rộng bán kính."
        )
        return

    # Score filtered places and limit to Top 10
    scored = _score_places(today, nearby, skin_type, activity_duration, regression_model)
    scored = scored[:10]  # Only show top 10 recommendations

    if not scored:
        with map_slot.container():
            _render_recommendation_map(
                center_lat,
                center_lon,
                loc_sel,
                scored=[],
                radius_km=radius_km,
                allow_click=input_method == "🗺️ Chọn trên bản đồ",
            )
        st.info("Không có gợi ý cho khu vực đã chọn (không đủ dữ liệu dự báo).")
        return

    with map_slot.container():
        _render_recommendation_map(
            center_lat,
            center_lon,
            loc_sel,
            scored,
            radius_km,
            allow_click=input_method == "🗺️ Chọn trên bản đồ",
        )

    st.markdown(
        f"**Top {len(scored)} địa điểm** trong bán kính **{radius_km:.0f} km** "
        f"- sắp xếp theo điểm UV an toàn"
    )

    # Place cards
    st.divider()
    grid_cols = st.columns(3)
    for i, p in enumerate(scored):
        with grid_cols[i % 3]:
            with st.container(border=True):
                if p.get("image_url"):
                    st.markdown(
                        f"""<img src="{p['image_url']}" style="width:100%; height:160px; object-fit:cover; border-radius:6px; margin-bottom:10px;">""",
                        unsafe_allow_html=True
                    )

                badges = []
                if p["has_shade"]: badges.append("🌿")
                if p["indoor_option"]: badges.append("🏠")
                badge_str = " ".join(badges)
                dist_str = f"📐 {p['distance_km']} km" if p.get("distance_km") else ""

                color = _get_marker_color(p["safe_hours_pct"])
                safe_badge = (
                    f"<span style='background:{color};color:white;padding:1px 7px;"
                    f"border-radius:4px;font-size:12px'>{p['safe_hours_pct']}% an toàn</span>"
                )

                st.markdown(f"**{i + 1}. {p['name']}** {badge_str}")
                st.markdown(f"<span style='color:#888;font-size:0.9rem'>{p['type'].replace('_', ' ').title()} · {dist_str}</span>", unsafe_allow_html=True)
                st.markdown(safe_badge, unsafe_allow_html=True)

                st.caption(
                    f"Hoạt động: {', '.join(p['activities'])}  \n"
                    f"UV trung bình: **{p['avg_uv']}** · Điểm: **{p['score']:.2f}**"
                )
                if p["best_start"] and p["best_end"]:
                    st.caption(f"⏰ {p['best_start'].strftime('%H:%M')} - {p['best_end'].strftime('%H:%M')}  \n🤖 {p.get('model_used', '')}")
                if p["maps_url"]:
                    st.markdown(f"[📍 Google Maps]({p['maps_url']})")

    # -- WHO health warnings ------------------------------------------------
    st.divider()
    st.subheader("⚠️ Cảnh báo sức khỏe tổng thể")
    locs_in_fc = [l for l in selected_locs if l in fc["location_id"].unique()]
    for loc_id in locs_in_fc:
        loc_today_w = today[today["location_id"] == loc_id]
        if loc_today_w.empty:
            continue
        peak_cat = loc_today_w.loc[loc_today_w["uv_predicted"].idxmax(), "uv_category"]
        advice   = WHO_ADVICE.get(peak_cat, "")
        mess     = f"**{LOCATION_NAMES.get(loc_id, loc_id)}** - {peak_cat}: {advice}"
        if peak_cat in ["Very High", "Extreme"]:
            st.error(mess)
        elif peak_cat == "High":
            st.warning(mess)
        else:
            st.success(mess)
