"""
Evaluation Dashboard - Đánh giá hệ thống gợi ý
=================================================
Streamlit component that renders quantitative evidence of the
UV tourism recommendation system's effectiveness for thesis defence.

Sections:
  1. Metric overview     - Precision@5, Recall@5, NDCG@5, Coverage
  2. Baseline comparison - horizontal bar chart vs random/popular/distance
  3. Diversity analysis  - type distribution pie + stacked bar + entropy
  4. Test case results   - pass/fail per scenario with expandable details
  5. Score breakdown     - score component transparency for any scenario
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -- Project imports ------------------------------------------------------------
from src.recommendation.evaluation import run_evaluation
from config import STATIC_DIR

# -- Paths ----------------------------------------------------------------------
_SCENARIOS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation" / "test_scenarios.json"
_PLACES_PATH    = STATIC_DIR / "suggest_location.json"

# -- Vietnamese place-type labels -----------------------------------------------
_TYPE_LABELS = {
    "beach":             "Bãi biển",
    "urban_park":        "Công viên đô thị",
    "museum":            "Bảo tàng",
    "shopping_mall":     "Trung tâm thương mại",
    "indoor_attraction": "Điểm tham quan trong nhà",
    "observation_deck":  "Đài quan sát",
    "temple":            "Chùa / Đền",
    "mangrove_tour":     "Du lịch rừng ngập mặn",
    "coastal_viewpoint": "Điểm ngắm biển",
    "restaurant":        "Nhà hàng",
}

_METRIC_HELP = {
    "precision_at_k": "Tỷ lệ địa điểm phù hợp trong top-5 gợi ý",
    "recall_at_k":    "Tỷ lệ loại địa điểm tốt được bao phủ trong top-5",
    "ndcg_at_k":      "Chất lượng xếp hạng (0-1, càng cao càng tốt)",
    "coverage":       "Tỷ lệ danh mục địa điểm xuất hiện ít nhất một lần",
}


# -- Caching --------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner="Đang đánh giá hệ thống…")
def _load_results() -> dict:
    """Run evaluation and cache results for 10 minutes."""
    with open(_PLACES_PATH, 'r', encoding="utf-8") as f:
        catalog = json.load(f).get("TOURIST_PLACES", [])
    return run_evaluation(_SCENARIOS_PATH, catalog, k=5)


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 - Metric Overview
# ═══════════════════════════════════════════════════════════════════════════════

def _render_metric_overview(overall: dict, baseline: dict) -> None:
    st.subheader("📊 Tổng quan chỉ số hệ thống")

    metrics = [
        ("Precision@5", "precision_at_k", "🎯"),
        ("Recall@5",    "recall_at_k",    "📡"),
        ("NDCG@5",      "ndcg_at_k",      "📈"),
        ("Coverage",    "coverage",        "🗂️"),
    ]
    cols = st.columns(4)
    for col, (label, key, icon) in zip(cols, metrics):
        val      = overall.get(key, 0)
        rnd_val  = baseline.get("random",  {}) if isinstance(baseline.get("random"), dict) else {}
        # baseline comparison is stored as flat averages in BaselineComparator.compare_all
        # delta vs random precision (best available proxy)
        random_prec = baseline.get("random", 0) if isinstance(baseline.get("random"), (int, float)) else 0
        current_prec = baseline.get("current", 0) if isinstance(baseline.get("current"), (int, float)) else 0
        delta_pct = (current_prec - random_prec) * 100 if key == "precision_at_k" else None

        display_val = _pct(val) if key != "ndcg_at_k" else f"{val:.3f}"
        delta_str   = f"+{delta_pct:.1f}% vs ngẫu nhiên" if delta_pct is not None else None

        with col:
            st.metric(
                label=f"{icon} {label}",
                value=display_val,
                delta=delta_str,
                help=_METRIC_HELP.get(key, ""),
            )

    # Diversity + Pass rate in a second row
    div_col, pass_col, mrr_col = st.columns(3)
    with div_col:
        st.metric("🎨 Diversity Score", f"{overall.get('diversity', 0):.3f}",
                  help="Entropy phân phối loại địa điểm (0-1)")
    with pass_col:
        st.metric("✅ Tỷ lệ vượt ngưỡng", _pct(overall.get("pass_rate", 0)),
                  help="% kịch bản đạt precision AND recall target")
    with mrr_col:
        st.metric("🔍 MRR", f"{overall.get('mrr', 0):.3f}",
                  help="Mean Reciprocal Rank - vị trí trung bình của kết quả đúng đầu tiên")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 - Baseline Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def _render_baseline_comparison(scenario_results: list[dict]) -> None:
    st.subheader("📊 So sánh với Baseline")
    st.caption("Hệ thống hiện tại so với 3 phương pháp cơ sở - chỉ số Precision@5 trung bình")

    # Aggregate per-scenario baseline comparisons
    methods = ["current", "random", "popular", "distance_only"]
    method_labels = {
        "current":       "🤖 Hệ thống hiện tại",
        "random":        "🎲 Ngẫu nhiên",
        "popular":       "⭐ Phổ biến nhất",
        "distance_only": "📐 Gần nhất",
    }

    # Re-compute from scenario_results metrics (current system only available here)
    # We already have per-scenario metrics; show averages of key metrics per method
    from src.recommendation.evaluation import AccuracyMetrics, BaselineComparator

    with open(_PLACES_PATH, 'r', encoding="utf-8") as f:
        catalog = json.load(f).get("TOURIST_PLACES", [])
    comparator = BaselineComparator(catalog)
    popularity = {p["name"]: max(1, len(catalog) - i) for i, p in enumerate(catalog)}

    agg: dict[str, dict[str, list]] = {m: {"precision": [], "recall": [], "ndcg": []} for m in methods}

    for sc_res in scenario_results[:25]:
        sc = {
            "user_profile": sc_res["user_profile"],
            "context":      sc_res["context"],
        }
        gt       = sc_res["ground_truth"]
        user_lat = sc_res["context"].get("lat", 10.7769)
        user_lon = sc_res["context"].get("lon", 106.7009)

        recs = sc_res["top5"]  # already-scored current system top-5

        comp = comparator.compare_all(recs, gt, user_lat, user_lon, 5, popularity)
        for method in methods:
            m = comp.get(method, {})
            agg[method]["precision"].append(m.get("precision_at_k", 0))
            agg[method]["recall"].append(m.get("recall_at_k", 0))
            agg[method]["ndcg"].append(m.get("ndcg_at_k", 0))

    bar_data = []
    metric_display = {"precision": "Precision@5", "recall": "Recall@5", "ndcg": "NDCG@5"}
    for method in methods:
        for metric, mlabel in metric_display.items():
            val = float(np.mean(agg[method][metric])) if agg[method][metric] else 0
            bar_data.append({
                "Phương pháp": method_labels[method],
                "Chỉ số": mlabel,
                "Giá trị": round(val, 4),
            })

    import pandas as pd
    df = pd.DataFrame(bar_data)

    color_map = {
        "🤖 Hệ thống hiện tại": "#2ecc71",
        "🎲 Ngẫu nhiên":        "#95a5a6",
        "⭐ Phổ biến nhất":     "#3498db",
        "📐 Gần nhất":          "#e67e22",
    }

    fig = px.bar(
        df, x="Giá trị", y="Phương pháp", color="Phương pháp",
        facet_col="Chỉ số", orientation="h",
        color_discrete_map=color_map,
        text_auto=".3f",
    )
    fig.update_layout(
        showlegend=False,
        height=280,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 - Diversity Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _render_diversity_analysis(scenario_results: list[dict]) -> None:
    st.subheader("🎨 Phân tích đa dạng gợi ý")

    # Collect all recommended types across all scenarios
    type_counts: dict[str, int] = {}
    top10_by_scenario: dict[str, dict[str, int]] = {}

    for sc in scenario_results:
        sc_label = sc["id"]
        top10_by_scenario[sc_label] = {}
        for rec in sc.get("top5", []):
            t = rec.get("type", "unknown")
            lbl = _TYPE_LABELS.get(t, t)
            type_counts[lbl] = type_counts.get(lbl, 0) + 1
            top10_by_scenario[sc_label][lbl] = top10_by_scenario[sc_label].get(lbl, 0) + 1

    col_pie, col_bar = st.columns(2)

    with col_pie:
        labels = list(type_counts.keys())
        values = list(type_counts.values())
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.4,
            textinfo="label+percent",
            marker_colors=px.colors.qualitative.Set3,
        ))
        fig_pie.update_layout(
            title="Phân phối loại địa điểm (tất cả kịch bản)",
            height=320,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, width='stretch')

        # Entropy score
        total = sum(values) or 1
        probs = [v / total for v in values]
        import math
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        max_entropy = math.log2(len(values)) if len(values) > 1 else 1.0
        norm_entropy = entropy / max_entropy
        st.metric("🎲 Diversity Entropy Score", f"{norm_entropy:.3f}",
                  help="0 = chỉ một loại, 1 = phân phối đều hoàn toàn")

    with col_bar:
        import pandas as pd
        rows = []
        for sc_id, t_counts in list(top10_by_scenario.items())[:10]:
            for t, cnt in t_counts.items():
                rows.append({"Kịch bản": sc_id, "Loại": t, "Số lượng": cnt})
        if rows:
            df_bar = pd.DataFrame(rows)
            fig_stack = px.bar(
                df_bar, x="Kịch bản", y="Số lượng", color="Loại",
                title="Phân phối loại địa điểm top-10 kịch bản đầu",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_stack.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_stack, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 - Test Case Results
# ═══════════════════════════════════════════════════════════════════════════════

def _render_test_case_results(scenario_results: list[dict]) -> None:
    st.subheader("🧪 Kết quả kịch bản kiểm thử")

    n_pass  = sum(1 for s in scenario_results if s["passed"])
    n_total = len(scenario_results)
    rate    = n_pass / max(1, n_total)

    # Visual pass-rate bar
    col_rate, col_bar = st.columns([1, 3])
    with col_rate:
        color = "#2ecc71" if rate >= 0.7 else ("#e67e22" if rate >= 0.5 else "#e74c3c")
        st.markdown(
            f"<div style='text-align:center;padding:12px;background:{color};border-radius:10px;"
            f"color:white;font-size:1.6rem;font-weight:bold'>{n_pass}/{n_total}</div>"
            f"<div style='text-align:center;color:{color};margin-top:4px'>Kịch bản đạt</div>",
            unsafe_allow_html=True,
        )
    with col_bar:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rate * 100,
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0,  50], "color": "#fadbd8"},
                    {"range": [50, 70], "color": "#fdebd0"},
                    {"range": [70, 100], "color": "#d5f5e3"},
                ],
            },
            title={"text": "Tỷ lệ vượt ngưỡng"},
        ))
        fig_gauge.update_layout(height=160, margin=dict(l=20, r=20, t=20, b=0),
                                paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, width='stretch')

    # Filter radio
    view = st.radio("Hiển thị kịch bản", ["Tất cả", "✅ Đạt", "❌ Không đạt"], horizontal=True)

    filtered = scenario_results
    if view == "✅ Đạt":
        filtered = [s for s in scenario_results if s["passed"]]
    elif view == "❌ Không đạt":
        filtered = [s for s in scenario_results if not s["passed"]]

    for sc in filtered:
        emoji = "✅" if sc["passed"] else "❌"
        prec  = sc["metrics"]["precision_at_k"]
        rec   = sc["metrics"]["recall_at_k"]
        with st.expander(f"{emoji} **{sc['id']}** - {sc['description']}  |  P@5={prec:.2f}  R@5={rec:.2f}"):
            col_in, col_out = st.columns(2)
            with col_in:
                st.markdown("**Input (User Profile & Context)**")
                combined = {**sc["user_profile"], **sc["context"]}
                st.json(combined, expanded=False)
            with col_out:
                st.markdown("**Top-5 Gợi ý**")
                for i, r in enumerate(sc["top5"], 1):
                    badge = "🏠" if r.get("indoor_option") else ("🌿" if r.get("has_shade") else "☀️")
                    t_lbl = _TYPE_LABELS.get(r.get("type", ""), r.get("type", ""))
                    safe_pct = r.get("safe_pct", 0)
                    safe_color = "#27ae60" if safe_pct >= 70 else ("#e67e22" if safe_pct >= 30 else "#e74c3c")
                    st.markdown(
                        f"{i}. {badge} **{r['name']}** - {t_lbl} "
                        f"<span style='background:{safe_color};color:white;padding:1px 6px;"
                        f"border-radius:4px;font-size:11px'>{safe_pct:.0f}%</span>",
                        unsafe_allow_html=True,
                    )

            st.markdown("---")
            st.markdown(f"**Ground Truth:** {sc['ground_truth'].get('explanation', '')}")
            m = sc["metrics"]
            st.markdown(
                f"**Metrics ->** Precision@5: `{m['precision_at_k']:.3f}` · "
                f"Recall@5: `{m['recall_at_k']:.3f}` · "
                f"NDCG@5: `{m['ndcg_at_k']:.3f}` · "
                f"MRR: `{m['mrr']:.3f}`"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 - Score Breakdown
# ═══════════════════════════════════════════════════════════════════════════════

def _render_score_breakdown(scenario_results: list[dict]) -> None:
    st.subheader("🔬 Chi tiết điểm số (Score Breakdown)")
    st.caption("Chọn một kịch bản để xem cách hệ thống tính điểm từng địa điểm")

    sc_ids = [sc["id"] for sc in scenario_results]
    selected_id = st.selectbox("Chọn kịch bản", sc_ids, key="eval_score_breakdown_select")
    sc = next((s for s in scenario_results if s["id"] == selected_id), None)
    if sc is None:
        return

    from src.recommendation.safe_time_policy import get_safe_exposure_time

    skin_type    = sc["user_profile"]["skin_type"]
    activity_min = sc["user_profile"]["activity_duration_minutes"]
    uv           = sc["context"]["uv_forecast"]
    has_rain     = sc["context"].get("is_raining", False)
    temperature  = sc["context"].get("temperature", 28)
    safe_minutes = get_safe_exposure_time(skin_type, uv)
    is_safe_hour = safe_minutes >= activity_min

    st.info(
        f"**Loại da:** {skin_type} · **UV:** {uv} · "
        f"**Thời gian an toàn:** {safe_minutes:.1f} phút · "
        f"**Hoạt động:** {activity_min} phút · "
        f"**Trời mưa:** {'Có' if has_rain else 'Không'} · "
        f"**Nhiệt độ:** {temperature}°C"
    )

    rows = []
    for r in sc["top5"]:
        shade_cov  = 50  # default from place
        shade_bon  = 1.0 + (shade_cov / 200.0)
        indoor_bon = 1.3 if r.get("indoor_option") else 1.0
        rain_pen   = 0.7 if (has_rain and not r.get("indoor_option")) else 1.0
        heat_pen   = 0.8 if (temperature > 36 and not r.get("has_shade") and not r.get("indoor_option")) else 1.0
        safe_ratio = 1.0 if is_safe_hour else (0.2 if r.get("indoor_option") else 0.0)

        rows.append({
            "name":       r["name"],
            "type":       _TYPE_LABELS.get(r.get("type",""), r.get("type","")),
            "safe_ratio": safe_ratio,
            "shade_bon":  shade_bon,
            "indoor_bon": indoor_bon,
            "rain_pen":   rain_pen,
            "heat_pen":   heat_pen,
            "score":      r.get("score", 0),
        })

    import pandas as pd
    df = pd.DataFrame(rows)

    fig = go.Figure()
    components = [
        ("UV Safety Ratio", "safe_ratio",  "#2ecc71"),
        ("Shade Bonus",     "shade_bon",   "#3498db"),
        ("Indoor Bonus",    "indoor_bon",  "#9b59b6"),
        ("Rain Penalty",    "rain_pen",    "#e67e22"),
        ("Heat Penalty",    "heat_pen",    "#e74c3c"),
    ]
    for label, col, color in components:
        fig.add_trace(go.Bar(
            name=label,
            y=df["name"],
            x=df[col],
            orientation="h",
            marker_color=color,
            hovertemplate=f"{label}: %{{x:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title="Score components per recommended place",
        xaxis_title="Component value",
        height=max(250, 60 * len(rows)),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width='stretch')

    # Table of exact values
    display_df = df[["name", "type", "safe_ratio", "shade_bon", "indoor_bon", "rain_pen", "heat_pen", "score"]].copy()
    display_df.columns = ["Địa điểm", "Loại", "UV Safety", "Shade Bonus", "Indoor Bonus", "Rain Penalty", "Heat Penalty", "Score"]
    st.dataframe(display_df.style.format({
        "UV Safety": "{:.3f}", "Shade Bonus": "{:.3f}", "Indoor Bonus": "{:.3f}",
        "Rain Penalty": "{:.3f}", "Heat Penalty": "{:.3f}", "Score": "{:.4f}",
    }), width='stretch', hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ═══════════════════════════════════════════════════════════════════════════════

def render() -> None:
    st.header("🔬 Đánh giá hệ thống gợi ý")
    st.caption(
        "Đánh giá định lượng chứng minh hiệu quả hệ thống gợi ý du lịch UV - "
        "dành cho bảo vệ đồ án tốt nghiệp. **Không thay đổi thuật toán cốt lõi.**"
    )

    if not _SCENARIOS_PATH.exists():
        st.error(f"Không tìm thấy file kịch bản: `{_SCENARIOS_PATH}`")
        return
    if not _PLACES_PATH.exists():
        st.error(f"Không tìm thấy dữ liệu địa điểm: `{_PLACES_PATH}`")
        return

    results = _load_results()
    overall          = results["overall"]
    baseline_comp    = results["baseline_comparison"]
    scenario_results = results["scenario_results"]

    # -- Target thresholds banner -----------------------------------------------
    targets = {
        "Precision@5 > 70%": overall["precision_at_k"] > 0.70,
        "Recall@5 > 60%":    overall["recall_at_k"]    > 0.60,
        "NDCG@5 > 0.75":     overall["ndcg_at_k"]      > 0.75,
        "Diversity > 0.6":   overall["diversity"]       > 0.60,
    }
    passed_targets = sum(targets.values())
    banner_color = "#2ecc71" if passed_targets == 4 else ("#e67e22" if passed_targets >= 2 else "#e74c3c")
    badges = " · ".join(f"{'✅' if v else '❌'} {k}" for k, v in targets.items())
    st.markdown(
        f"<div style='background:{banner_color}22;border-left:4px solid {banner_color};"
        f"padding:10px 14px;border-radius:6px;margin-bottom:8px'>"
        f"<b>Mục tiêu:</b> {badges}</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # -- Tab navigation inside the evaluation page ------------------------------
    t1, t2, t3, t4, t5 = st.tabs([
        "📊 Tổng quan chỉ số",
        "🏆 So sánh Baseline",
        "🎨 Đa dạng gợi ý",
        "🧪 Kết quả kịch bản",
        "🔬 Chi tiết điểm số",
    ])

    with t1:
        _render_metric_overview(overall, baseline_comp)
    with t2:
        _render_baseline_comparison(scenario_results)
    with t3:
        _render_diversity_analysis(scenario_results)
    with t4:
        _render_test_case_results(scenario_results)
    with t5:
        _render_score_breakdown(scenario_results)
