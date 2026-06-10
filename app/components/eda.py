"""EDA / Phân tích dữ liệu."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from app.utils.data_loader import LOCATION_COLORS, LOCATION_NAMES, UV_CATEGORY_COLORS
from config import FINAL_FEATURES


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
        st.plotly_chart(fig_hist, width='stretch', theme=None)

    with col_b:
        fig_box = px.box(
            data, x="location_name", y="uv_index",
            color="location_name",
            color_discrete_map={v: LOCATION_COLORS[k] for k, v in LOCATION_NAMES.items()},
            labels={"location_name": "Khu vực", "uv_index": "UV Index"},
            title="Phân bố UV Index theo khu vực"
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, width='stretch', theme=None)

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
        text="Count",
        title="Phân bố danh mục nguy hiểm UV",
        labels={"Category_vi": "Danh mục", "Count": "Số lượng"},
    )
    fig_cat.update_layout(showlegend=False, height=450)
    
    marker_colors = [UV_CATEGORY_COLORS.get(c, "#888") for c in cat_counts["Category"]]
    fig_cat.update_traces(textposition="outside", textfont_size=14, marker_color=marker_colors, width=0.5)
    
    col_left, col_center, col_right = st.columns([1, 4, 1])
    with col_center:
        st.plotly_chart(fig_cat, width='stretch', theme=None)

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
    key_feats = FINAL_FEATURES
    avail_feats = [f for f in key_feats if f in data.columns]

    if len(avail_feats) > 1 and "uv_index" in data.columns:
        valid_feats = []
        for feat in avail_feats:
            if data[feat].nunique() > 1 and data[feat].notna().sum() > 0:
                valid_feats.append(feat)

        valid_feats.append("uv_index")
        if len(valid_feats) > 2:
            corr_data = data[valid_feats].dropna()
            if len(corr_data) > 0:
                corr = corr_data.corr()
                lower_triangle_mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
                corr_lower = corr.mask(lower_triangle_mask)
                corr_height = max(1200, len(corr_lower) * 52 + 260)

                fig_corr = px.imshow(
                    corr_lower,
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    text_auto=".2f",
                    labels=dict(color="Hệ số tương quan"),
                    title="Ma trận tương quan tam giác dưới (các đặc trưng chính vs UV index)",
                )
                fig_corr.update_traces(
                    hoverongaps=False,
                    textfont=dict(size=19, family="Time New Roman"),
                )
                fig_corr.update_layout(
                    height=corr_height,
                    margin=dict(l=230, r=50, t=140, b=90),
                    title=dict(font=dict(size=24)),
                    coloraxis_colorbar=dict(
                        title=dict(font=dict(size=17)),
                        tickfont=dict(size=15),
                        x=0.93,
                        xanchor="left",
                        xpad=0,
                    ),
                )
                fig_corr.update_xaxes(
                    tickangle=45,
                    tickfont=dict(size=18),
                    automargin=True,
                )
                fig_corr.update_yaxes(
                    tickfont=dict(size=18),
                    automargin=True,
                )
                st.plotly_chart(fig_corr, width='stretch')

    # -- time series validation

    st.divider()
    st.subheader("Time series Validation")

    # 9a
    if "timestamp" in data.columns:
        data_range = f"{data['timestamp'].min().strftime('%Y-%m-%d')} -> {data['timestamp'].max().strftime('%Y-%m-%d')}"
        time_diff = data.groupby("location_id")["timestamp"].diff()
        most_common_freq = time_diff.mode()[0] if len(time_diff.mode()) > 0 else pd.Timedelta(hours=1)
        freq_str = f"{most_common_freq.total_seconds()/3600:.0f}h"

        st.text(f"Timeline: {data_range} | Observations: {len(data):,} | Frequency: {freq_str}")
    # 9b
    st.subheader("Autocorrelation Function (ACF & PACF)")

    # selec location for ACF/PACF
    unique_locs_ids = data["location_id"].unique()
    if len(unique_locs_ids) > 0:
        acf_loc = st.selectbox(
            "Chon dia diem de xem ACF/PACF",
            unique_locs_ids,
            format_func=lambda x: LOCATION_NAMES.get(x, x)
        )

        loc_ts_data = data[data["location_id"] == acf_loc].copy()
        if "timestamp" in loc_ts_data.columns:
            loc_ts_data = loc_ts_data.sort_values("timestamp")
            uv_series = loc_ts_data["uv_index"].dropna()

            if len(uv_series) > 50:
                col_acf, col_pacf = st.columns(2)

                with col_acf:
                    try:
                        fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
                        plot_acf(uv_series, lags=min(72, len(uv_series)//2), ax=ax_acf, alpha=0.05)
                        ax_acf.set_title(f"ACF - {LOCATION_NAMES.get(acf_loc, acf_loc)}")
                        ax_acf.set_xlabel("Lag (Giờ)")
                        ax_acf.set_ylabel("Autocorrelation")
                        st.pyplot(fig_acf)
                        plt.close(fig_acf)
                    except:
                        pass

                with col_pacf:
                    try:
                        fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
                        plot_pacf(uv_series, lags=min(72, len(uv_series)//2), ax=ax_pacf, alpha=0.05, method="ywm")
                        ax_pacf.set_title(f"PACF - {LOCATION_NAMES.get(acf_loc, acf_loc)}")
                        ax_pacf.set_xlabel("Lag (Giờ)")
                        ax_pacf.set_ylabel("Partial Autocorrelation")
                        st.pyplot(fig_pacf)
                        plt.close(fig_pacf)
                    except:
                        pass

    # 9c
    st.subheader("Seasonal Decomposition")

    if len(unique_locs_ids) > 0 and "timestamp" in data.columns:
        decomp_loc = st.selectbox(
            "Chọn địa điểm",
            unique_locs_ids,
            format_func=lambda x: LOCATION_NAMES.get(x, x),
            key="decomp_loc"
        )

        loc_decomp_data = data[data["location_id"] == decomp_loc].copy()
        loc_decomp_data = loc_decomp_data.sort_values("timestamp")

        # resample to daily
        if len(loc_decomp_data) > 100:
            try:
                daily_uv = loc_decomp_data.set_index("timestamp")["uv_index"].resample("D").mean().dropna()

                if len(daily_uv) > 30:
                    decomposition = seasonal_decompose(daily_uv, model="additive", period=30, extrapolate_trend="freq")

                    #plot
                    fig_decomp = go.Figure()

                    #trend
                    fig_decomp.add_trace(go.Scatter(
                        x=decomposition.trend.index,
                        y=decomposition.trend.values,
                        name="Trend (Xu hướng dài hạn)",
                        line=dict(color="#E74C3C", width=2)
                    ))

                    #seasonal
                    fig_decomp.add_trace(go.Scatter(
                        x=decomposition.seasonal.index,
                        y=decomposition.seasonal.values,
                        name="Seasonal (Chu kỳ theo mùa)",
                        line=dict(color="#3498DB", width=2)
                    ))

                    # #Residual
                    # fig_decomp.add_trace(go.Scatter(
                    #     x=decomposition.resid.index,
                    #     y=decomposition.resid.values,
                    #     name="Residual (Nhiễu ngẫu nhiên)",
                    #     line=dict(color="#95A5A6", width=1),
                    #     opacity=0.6
                    # ))

                    fig_decomp.update_layout(
                        title=f"Seasonal Decomposition - {LOCATION_NAMES.get(decomp_loc, decomp_loc)}",
                        xaxis_title="Time",
                        yaxis_title="UV Index",
                        height=500,
                        legend_title="Thành phần"
                    )

                    st.plotly_chart(fig_decomp, width='stretch')
            except Exception as e:
                print(e)
                pass

        # -- 8. Feature scatter ------------------------------------------------
    st.subheader("Tương quan đặc trưng vs UV Index")
    scatter_feats = [f for f in key_feats if f in data.columns]

    valid_scatter_feats = []
    for feat in scatter_feats:
        if data[feat].nunique() > 1 and data[feat].notna().sum() > 10:
            valid_scatter_feats.append(feat)

    if valid_scatter_feats and "uv_index" in data.columns:
        sample_data = data[valid_scatter_feats + ["uv_index"]].dropna()

        if len(sample_data) > 10:
            sample = sample_data.sample(min(3000, len(sample_data)), random_state=42)
            sel_feat = st.selectbox("Chon dac trung", valid_scatter_feats, index=0)

            fig_scatter = px.scatter(
                sample, x=sel_feat, y="uv_index",
                opacity=0.3, trendline="ols",
                color_continuous_scale=["#3498db"],
                labels={sel_feat: sel_feat, "uv_index": "UV Index"},
                title=f"{sel_feat} vs UV Index",
            )
            st.plotly_chart(fig_scatter, width='stretch')
