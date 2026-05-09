"""EDA / Phân tích dữ liệu."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils.data_loader import LOCATION_COLORS, LOCATION_NAMES, UV_CATEGORY_COLORS


def render(df: pd.DataFrame, selected_locs: list[str]):
    st.header("📊 Phân tích dữ liệu (EDA)")

    data = df[df["location_id"].isin(selected_locs)].copy() if selected_locs else df.copy()
    data["location_name"] = data["location_id"].map(LOCATION_NAMES)

    if data.empty:
        st.warning("Không có dữ liệu cho bộ lọc đã chọn.")
        return

    # -- 1. Dataset summary cards -------------------------------------------
    c1, c2 = st.columns(2)
    c2.metric("Địa điểm chọn:",          f"{data['location_id'].nunique()}")
    c1.metric("UV Index trung bình:", f"{data['uv_index'].mean():.2f}")
    st.divider()

    # -- 2. UV Index Distribution -------------------------------------------
    st.subheader("Phân bố UV Index")
    col_a, col_b = st.columns(2)

    with col_a:
        fig_hist = px.histogram(
            data, x="uv_index", color="location_name",
            color_discrete_map={v: LOCATION_COLORS[k] for k, v in LOCATION_NAMES.items()},
            nbins=60, barmode="overlay", opacity=0.65,
            labels={"uv_index": "UV Index", "location_name": "Khu vực"},
            title="Histogram UV Index (theo khu vực)"
        )
        fig_hist.update_layout(bargap=0.05, legend_title="Chú thích")
        st.plotly_chart(fig_hist, width='stretch')

    with col_b:
        fig_box = px.box(
            data, x="location_name", y="uv_index",
            color="location_name",
            color_discrete_map={v: LOCATION_COLORS[k] for k, v in LOCATION_NAMES.items()},
            labels={"location_name": "Khu vực", "uv_index": "UV Index"},
            title="Phân bố UV Index theo khu vực"
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, width='stretch')

    # -- 3. Class distribution ----------------------------------------------
    st.subheader("Phân bố danh mục UV (WHO)")
    bins  = [-0.1, 2,  5,  7, 10, float("inf")]
    names = ["Low", "Moderate", "High", "Very High", "Extreme"]
    names_vi = ["Thấp", "Trung bình", "Cao", "Rất cao", "Cực đoan"]
    data["uv_cat_label"] = pd.cut(data["uv_index"], bins=bins, labels=names).astype(str)
    cat_counts = data["uv_cat_label"].value_counts().reindex(names).fillna(0).reset_index()
    cat_counts.columns = ["Category", "Count"]
    cat_counts["Category_vi"] = cat_counts["Category"].map(dict(zip(names, names_vi)))
    fig_cat = px.bar(
        cat_counts,
        x="Category_vi",
        y="Count",
        color="Category",
        text = "Count",
        color_discrete_map=UV_CATEGORY_COLORS,
        title="Phân bố danh mục nguy hiểm UV",
        labels={"Category_vi": "Danh mục", "Count": "Số lượng"},
    )
    fig_cat.update_layout(showlegend=False)
    fig_cat.update_traces(textposition="outside")
    st.plotly_chart(fig_cat, width='stretch')

    # -- 4. UV Temporal Pattern ----------------------------------------------
    st.subheader("Xu hướng UV theo thời gian")
    col_hr, col_mo, col_yr = st.columns(3)

    with col_hr:
        hourly_avg = data.groupby(["hour", "location_name"])["uv_index"].mean().reset_index()
        fig_hour = px.line(
            hourly_avg, x="hour", y="uv_index", color="location_name",
            color_discrete_map={v: LOCATION_COLORS[k] for k, v in LOCATION_NAMES.items()},
            markers=True,
            labels={"hour": "Giờ trong ngày", "uv_index": "UV Index trung bình", "location_name": "Khu vực"},
            title="UV Index trung bình theo giờ"
        )
        fig_hour.add_vrect(x0=10, x1=14, fillcolor="rgba(231,76,60,0.08)",
                           annotation_text="Cao điểm (10-14h)", annotation_position="top left",
                           line_width=0)
        fig_hour.update_layout(xaxis=dict(tickmode="linear", dtick=1), legend_title="Chú thích")
        st.plotly_chart(fig_hour, width='stretch')

    month_names = {1:"Th1",2:"Th2",3:"Th3",4:"Th4",5:"Th5",6:"Th6",
                   7:"Th7",8:"Th8",9:"Th9",10:"Th10",11:"Th11",12:"Th12"}

    with col_mo:
        monthly_avg = data.groupby(["month", "location_name"])["uv_index"].mean().reset_index()
        fig_month = px.line(
            monthly_avg, x="month", y="uv_index", color="location_name",
            color_discrete_map={v: LOCATION_COLORS[k] for k, v in LOCATION_NAMES.items()},
            markers=True,
            labels={"month": "Tháng", "uv_index": "UV Index trung bình", "location_name": "Khu vực"},
            title="UV Index trung bình theo tháng"
        )
        fig_month.update_layout(xaxis=dict(tickmode="array", tickvals=list(range(1, 13)),
                                           ticktext=list(month_names.values())),
                                 legend_title="Chú thích")
        st.plotly_chart(fig_month, width='stretch')

    with col_yr:
        if "year" not in data.columns:
            data["year"] = data["timestamp"].dt.year
        yearly_avg = data.groupby(["year", "location_name"])["uv_index"].mean().reset_index()
        fig_year = px.line(
            yearly_avg, x="year", y="uv_index", color="location_name",
            color_discrete_map={v: LOCATION_COLORS[k] for k, v in LOCATION_NAMES.items()},
            markers=True,
            labels={"year": "Năm", "uv_index": "UV Index trung bình", "location_name": "Khu vực"},
            title="UV Index trung bình theo năm"
        )
        fig_year.update_layout(xaxis=dict(tickmode="linear", dtick=1), legend_title="Chú thích")
        st.plotly_chart(fig_year, width='stretch')

    # -- 5. UV by Hour x Month Heatmap ---------------------------------------
    st.subheader("Heatmap UV theo Giờ × Tháng")
    unique_locs = data["location_name"].unique()
    cols_hm = st.columns(min(len(unique_locs), 3))
    for i, loc_name in enumerate(unique_locs):
        loc_data = data[data["location_name"] == loc_name]
        pivot = loc_data.pivot_table(values="uv_index", index="hour", columns="month", aggfunc="mean")
        pivot.columns = [month_names.get(c, str(c)) for c in pivot.columns]

        fig_hm = px.imshow(
            pivot,
            labels=dict(x="Tháng", y="Giờ", color="UV trung bình"),
            x=pivot.columns,
            y=pivot.index,
            color_continuous_scale="YlOrRd",
            aspect="auto",
            text_auto=".1f",
            title=f"{loc_name}"
        )
        fig_hm.update_layout(coloraxis_showscale=False, title_font_size=14)
        with cols_hm[i % 3]:
            st.plotly_chart(fig_hm, width='stretch')

    # -- 6. Time Series - Daily UV Max ---------------------------------------
    st.subheader("Chuỗi thời gian - UV Max theo ngày")
    daily_max = data.groupby([data["timestamp"].dt.date.rename("date"), "location_name"])["uv_index"].max().reset_index()
    daily_max["date"] = pd.to_datetime(daily_max["date"])

    daily_max = daily_max.sort_values(by=["location_name", "date"])
    daily_max["30d_avg"] = daily_max.groupby("location_name")["uv_index"].transform(lambda x: x.rolling(30, min_periods=1).mean())

    fig_ts = go.Figure()
    for loc_name, grp in daily_max.groupby("location_name"):
        line_color = {v: LOCATION_COLORS[k] for k, v in LOCATION_NAMES.items()}.get(loc_name, "#888")
        fig_ts.add_trace(go.Scatter(
            x=grp["date"], y=grp["uv_index"],
            mode="lines", name=f"{loc_name} (ngày)",
            line=dict(width=1, color=line_color), opacity=0.3
        ))
        fig_ts.add_trace(go.Scatter(
            x=grp["date"], y=grp["30d_avg"],
            mode="lines", name=f"{loc_name} (TB 30 ngày)",
            line=dict(width=3, color=line_color)
        ))

    fig_ts.add_hline(y=11, line_dash="dash", line_color="purple", annotation_text="Ngưỡng cực đoan", opacity=0.4)
    fig_ts.update_layout(
        title="UV Max theo ngày (với đường trung bình 30 ngày)",
        xaxis_title="Ngày", yaxis_title="UV Index",
        legend_title="Chú thích",
    )
    st.plotly_chart(fig_ts, width='stretch')

    # -- 7. Correlation heatmap --------------------------------------------
    st.subheader("Ma trận tương quan đặc trưng")
    key_feats = [
        "cos_solar_zenith", "doy_sin", "temperature_2m", "relative_humidity_2m",
        "cloud_cover", "solar_cloud_interaction", "ozone_anomaly",
        "pressure_msl", "wind_speed_10m", "altitude_m",
    ]
    avail_feats = [f for f in key_feats if f in data.columns] + ["uv_index"]
    if len(avail_feats) > 2:
        corr = data[avail_feats].corr()
        fig_corr = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            labels=dict(color="Hệ số tương quan"),
            title="Ma trận tương quan (các đặc trưng chính vs UV Index)",
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, width='stretch')

    # -- 8. Feature scatter ------------------------------------------------
    st.subheader("Tương quan đặc trưng vs UV Index")
    scatter_feats = [f for f in key_feats if f in data.columns]
    if scatter_feats:
        sample = data[scatter_feats + ["uv_index"]].dropna().sample(min(3000, len(data)), random_state=42)
        sel_feat = st.selectbox("Chọn đặc trưng", scatter_feats, index=0)
        fig_scatter = px.scatter(
            sample, x=sel_feat, y="uv_index",
            opacity=0.3, trendline="ols",
            color_discrete_sequence=["#3498db"],
            labels={sel_feat: sel_feat, "uv_index": "UV Index"},
            title=f"{sel_feat} vs UV Index"
        )
        st.plotly_chart(fig_scatter, width='stretch')
