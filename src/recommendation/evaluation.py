"""
Recommendation System Evaluation Module
=========================================
Provides quantitative metrics to prove the UV tourism recommendation
system's effectiveness for thesis defence.

Three metric groups:
  1. AccuracyMetrics  - Precision@K, Recall@K, NDCG@K, MRR
  2. SystemMetrics    - Catalog coverage, diversity, novelty
  3. BaselineComparator - random / popular / distance-only baselines
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

# -- Type aliases --------------------------------------------------------------
Recommendations = list[dict]   # list of place dicts with 'name', 'type', 'score', etc.
GroundTruth     = dict         # from test_scenarios.json ground_truth section


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ACCURACY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class AccuracyMetrics:

    @staticmethod
    def precision_at_k(
        recommendations: Recommendations,
        ground_truth: GroundTruth,
        k: int = 5,
    ) -> float:
        should = set(ground_truth.get("should_recommend", []))
        if not should:
            return 0.0
        top_k = recommendations[:k]
        relevant = sum(1 for r in top_k if r.get("type") in should)
        return relevant / max(1, len(top_k))

    @staticmethod
    def recall_at_k(
        recommendations: Recommendations,
        ground_truth: GroundTruth,
        k: int = 5,
    ) -> float:
        should = set(ground_truth.get("should_recommend", []))
        if not should:
            return 0.0
        top_k = recommendations[:k]
        found_types = {r.get("type") for r in top_k} & should
        return len(found_types) / len(should)

    @staticmethod
    def ndcg_at_k(
        recommendations: Recommendations,
        ground_truth: GroundTruth,
        k: int = 5,
    ) -> float:
        should   = set(ground_truth.get("should_recommend", []))
        avoid    = set(ground_truth.get("should_avoid", []))
        top_k    = recommendations[:k]

        def gain(r: dict) -> int:
            t = r.get("type", "")
            if t in should:
                return 2
            if t not in avoid:
                return 1
            return 0

        dcg  = sum(gain(r) / math.log2(i + 2) for i, r in enumerate(top_k))
        # Ideal DCG from the same candidate list and gain definition as DCG.
        # This keeps NDCG normalized even when multiple items share one place type.
        ideal_gains = sorted((gain(r) for r in recommendations), reverse=True)[:k]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
        return (dcg / idcg) if idcg > 0 else 0.0

    @staticmethod
    def mean_reciprocal_rank(
        recommendations: Recommendations,
        ground_truth: GroundTruth,
    ) -> float:
        should = set(ground_truth.get("should_recommend", []))
        for i, r in enumerate(recommendations[:10]):
            if r.get("type") in should:
                return 1.0 / (i + 1)
        return 0.0

    @classmethod
    def evaluate(
        cls,
        recommendations: Recommendations,
        ground_truth: GroundTruth,
        k: int = 5,
    ) -> dict[str, float]:
        return {
            "precision_at_k": cls.precision_at_k(recommendations, ground_truth, k),
            "recall_at_k":    cls.recall_at_k(recommendations, ground_truth, k),
            "ndcg_at_k":      cls.ndcg_at_k(recommendations, ground_truth, k),
            "mrr":            cls.mean_reciprocal_rank(recommendations, ground_truth),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SYSTEM METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class SystemMetrics:

    @staticmethod
    def catalog_coverage(
        all_recommendations: list[Recommendations],
        catalog: list[dict],
    ) -> float:
        catalog_names = {p["name"] for p in catalog}
        recommended   = {r["name"] for recs in all_recommendations for r in recs}
        return len(recommended & catalog_names) / max(1, len(catalog_names))

    @staticmethod
    def diversity_score(recommendations: Recommendations) -> float:
        types = [r.get("type", "unknown") for r in recommendations]
        if not types:
            return 0.0
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        n = len(types)
        probs = [c / n for c in type_counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        max_entropy = math.log2(len(type_counts)) if len(type_counts) > 1 else 1.0
        return entropy / max_entropy

    @staticmethod
    def novelty_score(
        recommendations: Recommendations,
        popularity: dict[str, int],
    ) -> float:
        total_pop = max(1, sum(popularity.values()))
        scores = []
        for r in recommendations:
            pop = popularity.get(r.get("name", ""), 1)
            prob = pop / total_pop
            scores.append(-math.log2(prob))
        return float(np.mean(scores)) if scores else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BASELINE COMPARATOR
# ═══════════════════════════════════════════════════════════════════════════════

class BaselineComparator:

    def __init__(self, catalog: list[dict]):
        self.catalog = catalog

    def random_baseline(self, k: int = 5) -> Recommendations:
        """Random selection from catalog."""
        sample = random.sample(self.catalog, len(self.catalog))
        return [{"name": p["name"], "type": p.get("type", ""), "score": random.random()} for p in sample]

    def popular_baseline(
        self,
        popularity: dict[str, int],
        k: int = 5,
    ) -> Recommendations:

        sorted_catalog = sorted(
            self.catalog,
            key=lambda p: popularity.get(p["name"], 0),
            reverse=True,
        )
        return [
            {"name": p["name"], "type": p.get("type", ""), "score": popularity.get(p["name"], 0)}
            for p in sorted_catalog
        ]

    def distance_only_baseline(
        self,
        user_lat: float,
        user_lon: float,
        k: int = 5,
    ) -> Recommendations:
        """Nearest places only - ignores UV / safety."""
        import math

        def dist(p: dict) -> float:
            dlat = math.radians(p.get("lat", 0) - user_lat)
            dlon = math.radians(p.get("lon", 0) - user_lon)
            a = math.sin(dlat / 2) ** 2 + (
                math.cos(math.radians(user_lat))
                * math.cos(math.radians(p.get("lat", 0)))
                * math.sin(dlon / 2) ** 2
            )
            return 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        sorted_catalog = sorted(self.catalog, key=dist)
        return [
            {
                "name": p["name"],
                "type": p.get("type", ""),
                "score": 1.0 / (1.0 + dist(p)),
                "distance_km": round(dist(p), 1),
            }
            for p in sorted_catalog
        ]

    def compare_all(
        self,
        current_recommendations: Recommendations,
        ground_truth: GroundTruth,
        user_lat: float = 10.7769,
        user_lon: float = 106.7009,
        k: int = 5,
        popularity: dict[str, int] | None = None,
    ) -> dict[str, dict[str, float]]:

        pop = popularity or {}
        results = {
            "current":       AccuracyMetrics.evaluate(current_recommendations, ground_truth, k),
            "random":        AccuracyMetrics.evaluate(self.random_baseline(k), ground_truth, k),
            "popular":       AccuracyMetrics.evaluate(self.popular_baseline(pop, k), ground_truth, k),
            "distance_only": AccuracyMetrics.evaluate(
                self.distance_only_baseline(user_lat, user_lon, k), ground_truth, k
            ),
        }
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FULL SCENARIO EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _make_fake_recommendations_from_scenario(
    scenario: dict,
    catalog: list[dict],
) -> Recommendations:

    from src.recommendation.safe_time_policy import get_safe_exposure_time

    skin_type    = scenario["user_profile"]["skin_type"]
    activity_min = scenario["user_profile"]["activity_duration_minutes"]
    uv           = scenario["context"]["uv_forecast"]
    has_rain     = scenario["context"].get("is_raining", False)
    temperature  = scenario["context"].get("temperature", 28)
    location_id  = scenario["context"].get("location_id", "")

    # Filter catalog to places matching the scenario's location
    local_catalog = [p for p in catalog if p.get("location_key", "") == location_id]
    if not local_catalog:
        # Fallback: use full catalog if no places match (should not happen with clean data)
        local_catalog = catalog

    scored = []
    for place in local_catalog:
        shade_pct = place.get("shade_coverage_pct", 50)
        is_indoor = place.get("indoor_option", False)

        if is_indoor:
            effective_uv = uv * 0.05
        else:
            shade_ratio = shade_pct / 100.0
            transmission = (1.0 - shade_ratio) + (shade_ratio * 0.3)
            effective_uv = uv * transmission

        safe_minutes = get_safe_exposure_time(skin_type, effective_uv)
        safe_ratio = min(1.0, safe_minutes / max(1.0, activity_min))

        thermal_modifier = 1.0
        rain_modifier = 1.0

        if not is_indoor:
            if temperature >= 35.0:
                thermal_modifier = 0.5
            if has_rain:
                rain_modifier = 0.3

        score = safe_ratio * thermal_modifier * rain_modifier

        scored.append({
            "name":              place["name"],
            "type":              place.get("type", ""),
            "score":             round(score, 4),
            "safe_pct":          round(safe_ratio * 100, 1),
            "indoor_option":     is_indoor,
            "has_shade":         place.get("has_shade", False),
            "lat":               place.get("lat"),
            "lon":               place.get("lon"),
            "location_key":      place.get("location_key", ""),
            # -- Scoring component breakdown for UI transparency --
            "effective_uv":      round(effective_uv, 4),
            "safe_minutes":      round(safe_minutes, 1),
            "safe_ratio":        round(safe_ratio, 4),
            "thermal_modifier":  thermal_modifier,
            "rain_modifier":     rain_modifier,
            "shade_coverage_pct": shade_pct,
        })
    return sorted(scored, key=lambda x: x["score"], reverse=True)


def run_evaluation(
    scenarios_path: Path,
    catalog: list[dict],
    k: int = 5,
) -> dict[str, Any]:

    with open(scenarios_path) as f:
        data = json.load(f)
    scenarios = data.get("scenarios", [])

    comparator = BaselineComparator(catalog)

    # Build naive popularity from catalog position (first items are most visited)
    popularity = {p["name"]: max(1, len(catalog) - i) for i, p in enumerate(catalog)}
    random.seed(42)

    all_recs:          list[Recommendations] = []
    scenario_results:  list[dict]            = []
    agg_prec = agg_rec = agg_ndcg = agg_mrr = 0.0
    all_baseline_comp: dict[str, list[float]] = {
        "current":       [], "random": [], "popular": [], "distance_only": []
    }

    passed = 0
    for sc in scenarios:
        recs = _make_fake_recommendations_from_scenario(sc, catalog)
        gt   = sc.get("ground_truth", {})

        metrics = AccuracyMetrics.evaluate(recs, gt, k)
        agg_prec += metrics["precision_at_k"]
        agg_rec  += metrics["recall_at_k"]
        agg_ndcg += metrics["ndcg_at_k"]
        agg_mrr  += metrics["mrr"]
        all_recs.append(recs)

        # Per-scenario pass/fail
        target_prec = sc.get("expected_metrics", {}).get("target_precision", 0.6)
        target_rec  = sc.get("expected_metrics", {}).get("target_recall",    0.5)
        sc_pass = (
            metrics["precision_at_k"] >= target_prec
            and metrics["recall_at_k"] >= target_rec
        )
        if sc_pass:
            passed += 1

        # Baseline comparison for this scenario
        user_lat = sc["context"].get("lat", 10.7769)
        user_lon = sc["context"].get("lon", 106.7009)
        comp = comparator.compare_all(recs, gt, user_lat, user_lon, k, popularity)
        for method, m in comp.items():
            all_baseline_comp[method].append(m["precision_at_k"])

        scenario_results.append({
            "id":          sc.get("id"),
            "description": sc.get("description", ""),
            "user_profile": sc.get("user_profile", {}),
            "context":     sc.get("context", {}),
            "top5":        recs[:5],
            "metrics":     metrics,
            "passed":      sc_pass,
            "ground_truth": gt,
        })

    n = max(1, len(scenarios))
    diversity_scores = [SystemMetrics.diversity_score(r) for r in all_recs]

    overall = {
        "precision_at_k": round(agg_prec / n, 4),
        "recall_at_k":    round(agg_rec  / n, 4),
        "ndcg_at_k":      round(agg_ndcg / n, 4),
        "mrr":            round(agg_mrr  / n, 4),
        "coverage":       round(SystemMetrics.catalog_coverage(all_recs, catalog), 4),
        "diversity":      round(float(np.mean(diversity_scores)), 4),
        "pass_rate":      round(passed / n, 4),
    }

    baseline_comparison = {
        method: round(float(np.mean(vals)), 4) if vals else 0.0
        for method, vals in all_baseline_comp.items()
    }

    return {
        "overall":              overall,
        "baseline_comparison":  baseline_comparison,
        "scenario_results":     scenario_results,
        "popularity":           popularity,
    }
