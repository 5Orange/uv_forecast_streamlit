"""Kết quả mô hình - Tất cả mô hình tối ưu (ML · DL · Hybrid)."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils.data_loader import (
    load_regression_results,
    load_regression_results_pivot,
    load_model_metadata,
)

# Colour per group
GROUP_COLORS = {
    "Traditional ML":   "#3498db",
    "Deep Learning":    "#9b59b6",
    "Hybrid":           "#27ae60",
    "Ensemble":         "#1abc9c",
}


def render():
    st.subheader("🤖 Kết quả mô hình")

    raw_df = load_regression_results()
    pivot_df = load_regression_results_pivot()

    # -- KPI: best model per group by test RMSE -----------------------------
    test_df = raw_df[raw_df["split"] == "test"].sort_values("rmse")
    best_per_group = test_df.groupby("type", sort=False).first().reset_index()
    top3 = best_per_group.sort_values("rmse").head(3)
    cols = st.columns(len(top3))
    medals = ["🥇", "🥈", "🥉"]
    for col, medal, (_, row) in zip(cols, medals, top3.iterrows()):
        col.metric(
            f"{medal} {row['model']} ({row['type']})",
            f"RMSE {row['rmse']:.4f}",
            delta=f"R2 {row['r2']:.4f}",
            delta_color="off",
        )

    # -- Full leaderboard table (all splits) --------------------------------
    st.markdown("#### 📋 Bảng xếp hạng đầy đủ (Train / Val / Test)")
    st.markdown("*Optuna tuned: n = 30 trials*")
    
    # Format numeric columns
    fmt_cols = [c for c in pivot_df.columns if any(m in c for m in ["mae", "rmse", "r2", "mape"])]
    
    # Add Optuna params column with HTML details expander
    params_list = []
    for model_name in pivot_df["model"]:
        meta_key = model_name.lower().replace(" ", "_").replace("-", "_").replace("+", "_")
        params = load_model_metadata(meta_key)
        if not params:
            alt_keys = {"random forest": "rf", "decision tree": "dt",
                        "xgboost": "xgb", "lightgbm": "lgb"}
            meta_key = alt_keys.get(model_name.lower(), meta_key)
            params = load_model_metadata(meta_key)
        
        if params:
            param_html = "<br>".join([f"<b>{k}</b>: {v}" for k, v in params.items()])
            details = f"<details><summary>Xem tham số</summary><div style='text-align:left; font-size: 0.9em; padding:4px;'>{param_html}</div></details>"
            params_list.append(details)
        else:
            params_list.append("")
    
    pivot_df["optuna_params"] = params_list
    
    # Rename columns for display
    col_rename = {}
    for c in pivot_df.columns:
        parts = c.split("_", 1)
        if len(parts) == 2 and parts[0] in ("train", "val", "test"):
            split_vi = {"train": "Train", "val": "Val", "test": "Test"}[parts[0]]
            metric = parts[1].upper()
            col_rename[c] = f"{split_vi} {metric}"
        elif c == "type":
            col_rename[c] = "Nhóm"
        elif c == "model":
            col_rename[c] = "Mô hình"
        elif c == "optuna_params":
            col_rename[c] = "Tham số Optuna"
    
    display_df = pivot_df.rename(columns=col_rename)
    
    def highlight_ranks(col):
        if col.name not in [v for k, v in col_rename.items() if k in fmt_cols]:
            return [""] * len(col)
        
        is_r2 = "R2" in col.name
        # Rank: 1 is best
        ranks = col.rank(method="min", ascending=not is_r2)
        
        styles = []
        for r in ranks:
            if r == 1:
                styles.append("font-weight: bold")
            elif r == 2:
                styles.append("text-decoration: underline")
            elif r == 3:
                styles.append("font-style: italic")
            else:
                styles.append("")
        return styles
    
    styled = display_df.style.apply(highlight_ranks, axis=0).format("{:.4f}", subset=[v for k, v in col_rename.items() if k in fmt_cols]).hide(axis='index')
    
    # Render table as HTML to support <details> tag and exact CSS mapping
    table_css = """
    <style>
    .results-table { width: 100%; border-collapse: collapse; font-family: sans-serif; margin-bottom: 2rem; }
    .results-table th, .results-table td { border: 1px solid #ddd; padding: 8px 12px; text-align: right; }
    .results-table th { background-color: #f8f9fa; font-weight: 600; text-align: center; }
    .results-table td:nth-child(1), .results-table td:nth-child(2), .results-table td:last-child { text-align: left; }
    .results-table details { cursor: pointer; }
    .results-table summary { outline: none; font-weight: 500; color: #E67E22; }
    </style>
    """
    html_table = styled.to_html(escape=False, table_uuid="results", table_attributes='class="results-table"')
    st.markdown(f"<div>{table_css}{html_table}</div>", unsafe_allow_html=True)

    # -- RMSE bar - test split, coloured by group --------------------------
    st.markdown("#### 📊 So sánh RMSE (Test)")
    sorted_test_df_rmse = test_df.sort_values("rmse")
    marker_colors_rmse = sorted_test_df_rmse["type"].map(GROUP_COLORS).tolist()
    
    fig_rmse = px.bar(
        sorted_test_df_rmse,
        x="rmse", y="model",
        orientation="h",
        text="rmse",
        labels={"rmse": "RMSE (thấp hơn = tốt hơn)", "model": "Mô hình"},
        title="So sánh RMSE - Tất cả mô hình tối ưu",
    )
    fig_rmse.update_traces(texttemplate="%{text:.4f}", textposition="outside", textfont_size=14, marker_color=marker_colors_rmse)
    fig_rmse.update_layout(
        xaxis_range=[0, test_df["rmse"].max() * 1.18],
        legend_title="Chú thích",
        height=420,
    )
    # Add dummy traces for legend
    for g_name, g_color in GROUP_COLORS.items():
        if g_name in sorted_test_df_rmse["type"].unique():
            fig_rmse.add_trace(go.Bar(x=[None], y=[None], marker_color=g_color, name=g_name))
            
    st.plotly_chart(fig_rmse, width='stretch', theme=None)

    # -- R2 bar ------------------------------------------------------------
    st.markdown("#### 📊 So sánh R2 (Test)")
    sorted_test_df_r2 = test_df.sort_values("r2", ascending=False)
    marker_colors_r2 = sorted_test_df_r2["type"].map(GROUP_COLORS).tolist()

    fig_r2 = px.bar(
        sorted_test_df_r2,
        x="r2", y="model",
        orientation="h",
        text="r2",
        labels={"r2": "R2 (cao hơn = tốt hơn)", "model": "Mô hình"},
        title="So sánh R2 - Tất cả mô hình tối ưu",
    )
    fig_r2.update_traces(texttemplate="%{text:.4f}", textposition="outside", textfont_size=14, marker_color=marker_colors_r2)
    fig_r2.update_layout(
        xaxis_range=[max(0, test_df["r2"].min() - 0.05), 1.05],
        legend_title="Chú thích",
        height=420,
    )
    # Add dummy traces for legend
    for g_name, g_color in GROUP_COLORS.items():
        if g_name in sorted_test_df_r2["type"].unique():
            fig_r2.add_trace(go.Bar(x=[None], y=[None], marker_color=g_color, name=g_name))
            
    st.plotly_chart(fig_r2, width='stretch', theme=None)

    # -- MAE vs RMSE scatter -----------------------------------------------
    st.markdown("#### 🔍 MAE vs RMSE - Phân tích đánh đổi (Test)")
    fig_scatter = px.scatter(
        test_df,
        x="mae", y="rmse",
        color="type",
        text="model",
        size_max=14,
        color_discrete_map=GROUP_COLORS,
        labels={"mae": "MAE", "rmse": "RMSE", "type": "Nhóm"},
        title="MAE vs RMSE (điểm tốt = gần góc dưới-trái)",
    )
    fig_scatter.update_traces(textposition="top center", marker_size=10)
    mn = test_df[["mae", "rmse"]].min().min() * 0.9
    mx = test_df[["mae", "rmse"]].max().max() * 1.05
    fig_scatter.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                          line=dict(dash="dot", color="#ccc"))
    fig_scatter.update_layout(legend_title="Chú thích")
    st.plotly_chart(fig_scatter, width='stretch', theme=None)

    # -- Train vs Test R2 - overfitting check ------------------------------
    st.markdown("#### 🔬 Kiểm tra Overfitting (Train R2 vs Test R2)")
    train_df = raw_df[raw_df["split"] == "train"][["model", "type", "r2"]].rename(columns={"r2": "train_r2"})
    test_r2_df = test_df[["model", "r2"]].rename(columns={"r2": "test_r2"})
    overfit_df = train_df.merge(test_r2_df, on="model")
    overfit_df["gap"] = overfit_df["train_r2"] - overfit_df["test_r2"]
    
    fig_overfit = px.scatter(
        overfit_df, x="train_r2", y="test_r2",
        color="type", text="model",
        color_discrete_map=GROUP_COLORS,
        labels={"train_r2": "Train R2", "test_r2": "Test R2", "type": "Nhóm"},
        title="Train R2 vs Test R2 (điểm gần đường chéo = ít overfitting)",
    )
    fig_overfit.update_traces(textposition="top center", marker_size=10)
    fig_overfit.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dot", color="#ccc"))
    fig_overfit.update_layout(legend_title="Chú thích")
    st.plotly_chart(fig_overfit, width='stretch', theme=None)

    # -- Group-level summary -----------------------------------------------
    st.markdown("#### 📊 Tổng hợp theo nhóm mô hình (Test)")
    summary = (
        test_df.groupby("type")[["mae", "rmse", "r2", "mape"]]
        .agg(["mean", "min"])
        .round(4)
    )
    summary.columns = [" ".join(c) for c in summary.columns]
    st.dataframe(summary, width='stretch')

