"""
Data Siege defense.

Strategy:
- use exactly one metered toolkit call per event type (private budget supports this),
- alert on published baseline violations,
- add conservative near-boundary and robust-history checks for subtle private faults,
- keep all learned state in ctx.state; no file I/O, no unsupported imports, no seed/event hardcoding.
"""
from api import Verdict


MIN_HISTORY = 12
MAX_HISTORY = 80


def register(ctx):
    ctx.state.setdefault("series", {})
    ctx.state.setdefault("sets", {})
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def _median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _num(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _base(ctx, key, default=None):
    return _num(ctx.baseline.get(key), default)


def _range_pressure(value, lo, hi):
    value = _num(value)
    lo = _num(lo)
    hi = _num(hi)
    if value is None or lo is None or hi is None or hi <= lo:
        return 0.0
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    if half <= 0:
        return 0.0
    return abs(value - center) / half


def _high_pressure(value, limit):
    value = _num(value)
    limit = _num(limit)
    if value is None or limit is None or limit <= 0:
        return 0.0
    return value / limit


def _add_history(ctx, key, value, ok=True):
    value = _num(value)
    if not ok or value is None:
        return
    series = ctx.state.setdefault("series", {}).setdefault(key, [])
    series.append(value)
    if len(series) > MAX_HISTORY:
        del series[: len(series) - MAX_HISTORY]


def _robust_outlier(ctx, key, value, *, high_only=False, low_only=False, min_n=MIN_HISTORY, z=3.2):
    value = _num(value)
    if value is None:
        return False, 0.0
    vals = ctx.state.setdefault("series", {}).get(key, [])
    if len(vals) < min_n:
        return False, 0.0
    m = _median(vals)
    deviations = [abs(v - m) for v in vals]
    mad = _median(deviations)
    # Keep the denominator sane while the run is still short or naturally low-variance.
    floor = max(abs(m) * 0.025, 0.001)
    scale = max(1.4826 * mad, floor)
    if high_only:
        score = (value - m) / scale
    elif low_only:
        score = (m - value) / scale
    else:
        score = abs(value - m) / scale
    return score >= z, score


def _verdict(alert, pillar, reasons, confidence=0.5):
    clean_reasons = [r for r in reasons if r]
    if alert:
        return Verdict(alert=True, confidence=max(0.55, min(0.99, confidence)), reason="; ".join(clean_reasons), pillar=pillar)
    return Verdict(alert=False, confidence=0.25, reason="within baseline and learned envelope", pillar=pillar)


def _tool_error(result):
    return not isinstance(result, dict) or "error" in result


def check_data_batch(payload, ctx):
    batch_id = payload.get("batch_id")
    prof = ctx.tools.batch_profile(batch_id)
    if _tool_error(prof):
        return _verdict(False, "checks", ["batch_profile_unavailable"], 0.1)

    row = _num(prof.get("row_count"))
    null_rate = _num((prof.get("null_rate") or {}).get("customer_id"))
    mean_amount = _num(prof.get("mean_amount"))
    std_amount = _num(prof.get("std_amount"))
    staleness = _num(prof.get("staleness_min"))

    row_min = _base(ctx, "row_count_min")
    row_max = _base(ctx, "row_count_max")
    null_max = _base(ctx, "null_rate_max")
    mean_min = _base(ctx, "mean_amount_min")
    mean_max = _base(ctx, "mean_amount_max")
    stale_max = _base(ctx, "staleness_min_max")

    reasons = []
    confidence = 0.55

    if row is not None and row_min is not None and row_max is not None and (row < row_min or row > row_max):
        reasons.append("row_count_outside_published_clean_band")
        confidence = max(confidence, 0.96)
    if null_rate is not None and null_max is not None and null_rate > null_max:
        reasons.append("customer_id_null_rate_above_published_clean_band")
        confidence = max(confidence, 0.96)
    if mean_amount is not None and mean_min is not None and mean_max is not None and (mean_amount < mean_min or mean_amount > mean_max):
        reasons.append("mean_amount_outside_published_clean_band")
        confidence = max(confidence, 0.96)
    if staleness is not None and stale_max is not None and staleness > stale_max:
        reasons.append("batch_staleness_above_published_clean_band")
        confidence = max(confidence, 0.96)

    # Subtle private faults may sit just inside mean+/-3sigma baselines. Use a
    # conservative near-boundary rule and require either an extreme single signal
    # or multiple moderately suspicious signals.
    row_p = _range_pressure(row, row_min, row_max)
    mean_p = _range_pressure(mean_amount, mean_min, mean_max)
    null_p = _high_pressure(null_rate, null_max)
    stale_p = _high_pressure(staleness, stale_max)
    suspicious = [row_p >= 0.86, mean_p >= 0.86, null_p >= 0.86, stale_p >= 0.88]
    if not reasons:
        if row_p >= 0.93:
            reasons.append("row_count_near_extreme_clean_boundary")
            confidence = max(confidence, 0.74)
        if mean_p >= 0.93:
            reasons.append("mean_amount_near_extreme_clean_boundary")
            confidence = max(confidence, 0.74)
        if null_p >= 0.92:
            reasons.append("customer_id_null_rate_near_extreme_clean_boundary")
            confidence = max(confidence, 0.74)
        if stale_p >= 0.92:
            reasons.append("staleness_near_extreme_clean_boundary")
            confidence = max(confidence, 0.74)
        if len([x for x in suspicious if x]) >= 2:
            reasons.append("multiple_batch_metrics_near_clean_boundary")
            confidence = max(confidence, 0.78)

    # Robust history, learned only from earlier non-alerting events, catches
    # phase-specific shifts without hardcoding event ids or schedule answers.
    dyn_checks = [
        ("checks.row_count", row, False),
        ("checks.mean_amount", mean_amount, False),
        ("checks.std_amount", std_amount, True),
        ("checks.null_rate", null_rate, True),
        ("checks.staleness", staleness, True),
    ]
    for key, value, high_only in dyn_checks:
        is_outlier, score = _robust_outlier(ctx, key, value, high_only=high_only, z=3.0 if high_only else 3.15)
        if is_outlier:
            reasons.append(key.split(".")[-1] + "_robust_history_outlier")
            confidence = max(confidence, min(0.93, 0.62 + score / 20.0))

    alert = bool(reasons)
    # Do not train the clean-history model on events we already consider faulty.
    for key, value, _ in dyn_checks:
        _add_history(ctx, key, value, ok=not alert)

    return _verdict(alert, "checks", reasons, confidence)


def check_contract_checkpoint(payload, ctx):
    diff = ctx.tools.contract_diff(payload.get("contract_id"), payload.get("checkpoint_batch_id"))
    if _tool_error(diff):
        return _verdict(False, "contracts", ["contract_diff_unavailable"], 0.1)

    delay = _num(diff.get("freshness_delay_min"))
    max_delay = _base(ctx, "freshness_delay_max_min")
    violations = diff.get("violations") or []
    reasons = []
    confidence = 0.55

    if violations:
        reasons.append("contract_violations:" + ",".join(str(v) for v in violations))
        confidence = max(confidence, 0.99)
    if delay is not None and max_delay is not None and delay > max_delay:
        reasons.append("contract_freshness_sla_exceeded")
        confidence = max(confidence, 0.96)
    elif delay is not None and max_delay is not None and delay >= 0.90 * max_delay:
        # One-sided baseline is clean mean+3sigma; 90% of it is still a high-SLA-risk tail.
        reasons.append("contract_freshness_near_sla_boundary")
        confidence = max(confidence, 0.72)

    key = "contracts.freshness_delay." + str(payload.get("contract_id", "all"))
    outlier, score = _robust_outlier(ctx, key, delay, high_only=True, z=3.0)
    if outlier:
        reasons.append("contract_freshness_robust_history_outlier")
        confidence = max(confidence, min(0.93, 0.62 + score / 20.0))

    alert = bool(reasons)
    _add_history(ctx, key, delay, ok=not alert)
    return _verdict(alert, "contracts", reasons, confidence)


def _payload_expected_set(payload):
    for name in ("expected_upstream", "expected_upstreams", "declared_upstream", "declared_upstreams"):
        if name in payload:
            val = payload.get(name)
            if isinstance(val, (list, tuple, set)):
                return set(str(x) for x in val)
            if isinstance(val, str) and val:
                return set([val])
    return None


def _lineage_key(payload):
    for name in ("transform", "transform_name", "job", "job_name", "asset", "pipeline", "task"):
        val = payload.get(name)
        if val:
            return "lineage." + str(val)
    return "lineage.all"


def check_lineage_run(payload, ctx):
    graph = ctx.tools.lineage_graph_slice(payload.get("run_id"))
    if _tool_error(graph):
        return _verdict(False, "lineage", ["lineage_graph_slice_unavailable"], 0.1)

    duration = _num(graph.get("duration_ms"))
    max_duration = _base(ctx, "lineage_duration_ms_max")
    upstream = graph.get("actual_upstream") or []
    if not isinstance(upstream, (list, tuple, set)):
        upstream = [upstream]
    upstream_set = set(str(x) for x in upstream if x is not None and str(x) != "")
    downstream_count = _num(graph.get("actual_downstream_count"))

    reasons = []
    confidence = 0.55

    if duration is not None and max_duration is not None and duration > max_duration:
        reasons.append("lineage_runtime_above_published_clean_band")
        confidence = max(confidence, 0.96)
    elif duration is not None and max_duration is not None and duration >= 0.90 * max_duration:
        reasons.append("lineage_runtime_near_extreme_clean_boundary")
        confidence = max(confidence, 0.72)

    expected_upstream = _payload_expected_set(payload)
    if expected_upstream is not None and upstream_set != expected_upstream:
        reasons.append("lineage_upstream_mismatch_vs_event_expectation")
        confidence = max(confidence, 0.97)
    elif not upstream_set:
        reasons.append("lineage_missing_all_upstream_edges")
        confidence = max(confidence, 0.91)

    for name in ("expected_downstream_count", "declared_downstream_count", "expected_outputs"):
        if name in payload:
            expected_downstream = _num(payload.get(name))
            if expected_downstream is not None and downstream_count is not None and downstream_count != expected_downstream:
                reasons.append("lineage_downstream_count_mismatch_vs_event_expectation")
                confidence = max(confidence, 0.94)
            break
    if downstream_count is not None and downstream_count <= 0:
        reasons.append("lineage_orphaned_output_count")
        confidence = max(confidence, 0.90)

    key = _lineage_key(payload)
    duration_key = key + ".duration_ms"
    downstream_key = key + ".downstream_count"
    outlier, score = _robust_outlier(ctx, duration_key, duration, high_only=True, z=3.0)
    if outlier:
        reasons.append("lineage_runtime_robust_history_outlier")
        confidence = max(confidence, min(0.93, 0.62 + score / 20.0))
    outlier, score = _robust_outlier(ctx, downstream_key, downstream_count, z=3.0)
    if outlier:
        reasons.append("lineage_downstream_count_robust_history_outlier")
        confidence = max(confidence, min(0.92, 0.62 + score / 20.0))

    alert = bool(reasons)
    _add_history(ctx, duration_key, duration, ok=not alert)
    _add_history(ctx, downstream_key, downstream_count, ok=not alert)
    return _verdict(alert, "lineage", reasons, confidence)


def check_feature_materialization(payload, ctx):
    drift = ctx.tools.feature_drift(payload.get("feature_view"), payload.get("batch_id"))
    if _tool_error(drift):
        return _verdict(False, "ai_infra", ["feature_drift_unavailable"], 0.1)

    shift = _num(drift.get("mean_shift_sigma"))
    max_shift = _base(ctx, "feature_mean_shift_sigma_max")
    reasons = []
    confidence = 0.55

    if shift is not None and max_shift is not None and shift > max_shift:
        reasons.append("feature_train_serving_skew_above_published_clean_band")
        confidence = max(confidence, 0.96)
    elif shift is not None and max_shift is not None and shift >= 0.75 * max_shift:
        reasons.append("feature_train_serving_skew_near_clean_boundary")
        confidence = max(confidence, 0.74)

    key = "feature.shift." + str(payload.get("feature_view", "all"))
    outlier, score = _robust_outlier(ctx, key, shift, high_only=True, z=2.9)
    if outlier:
        reasons.append("feature_shift_robust_history_outlier")
        confidence = max(confidence, min(0.93, 0.62 + score / 20.0))

    alert = bool(reasons)
    _add_history(ctx, key, shift, ok=not alert)
    return _verdict(alert, "ai_infra", reasons, confidence)


def check_embedding_batch(payload, ctx):
    drift = ctx.tools.embedding_drift(payload.get("corpus"), payload.get("chunk_batch_id"))
    if _tool_error(drift):
        return _verdict(False, "ai_infra", ["embedding_drift_unavailable"], 0.1)

    centroid = _num(drift.get("centroid_shift"))
    age = _num(drift.get("avg_doc_age_days"))
    max_centroid = _base(ctx, "embedding_centroid_shift_max")
    max_age = _base(ctx, "corpus_avg_doc_age_days_max")
    centroid_p = _high_pressure(centroid, max_centroid)
    age_p = _high_pressure(age, max_age)
    reasons = []
    confidence = 0.55

    if centroid is not None and max_centroid is not None and centroid > max_centroid:
        reasons.append("embedding_centroid_shift_above_published_clean_band")
        confidence = max(confidence, 0.96)
    if age is not None and max_age is not None and age > max_age:
        reasons.append("rag_corpus_age_above_published_clean_band")
        confidence = max(confidence, 0.96)

    if not reasons:
        if centroid_p >= 0.80:
            reasons.append("embedding_centroid_shift_near_clean_boundary")
            confidence = max(confidence, 0.74)
        if age_p >= 0.85:
            reasons.append("rag_corpus_age_near_clean_boundary")
            confidence = max(confidence, 0.72)
        if centroid_p >= 0.65 and age_p >= 0.65:
            reasons.append("combined_embedding_drift_and_corpus_staleness_pressure")
            confidence = max(confidence, 0.77)

    prefix = "embedding." + str(payload.get("corpus", "all"))
    for suffix, value in (("centroid", centroid), ("age", age)):
        outlier, score = _robust_outlier(ctx, prefix + "." + suffix, value, high_only=True, z=2.9)
        if outlier:
            reasons.append("embedding_" + suffix + "_robust_history_outlier")
            confidence = max(confidence, min(0.93, 0.62 + score / 20.0))

    alert = bool(reasons)
    _add_history(ctx, prefix + ".centroid", centroid, ok=not alert)
    _add_history(ctx, prefix + ".age", age, ok=not alert)
    return _verdict(alert, "ai_infra", reasons, confidence)
