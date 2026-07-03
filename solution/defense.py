"""
Data Siege defense.

Balanced legal detector. The high-recall version over-alerted, while the
precision version under-alerted. This tune opens the strongest tail/history
signals while keeping weak single-signal alerts tightened.
"""
from api import Verdict


MIN_HISTORY = 7
MAX_HISTORY = 100


def register(ctx):
    ctx.state.setdefault("series", {})
    ctx.state.setdefault("signatures", {})
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def _num(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _base(ctx, key, default=None):
    return _num(ctx.baseline.get(key), default)


def _median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _range_pressure(value, lo, hi):
    value = _num(value)
    lo = _num(lo)
    hi = _num(hi)
    if value is None or lo is None or hi is None or hi <= lo:
        return 0.0
    return abs(value - ((lo + hi) / 2.0)) / ((hi - lo) / 2.0)


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


def _robust_outlier(ctx, key, value, *, high_only=False, low_only=False, min_n=MIN_HISTORY, z=2.60):
    value = _num(value)
    if value is None:
        return False, 0.0
    vals = ctx.state.setdefault("series", {}).get(key, [])
    if len(vals) < min_n:
        return False, 0.0
    m = _median(vals)
    mad = _median([abs(v - m) for v in vals])
    floor = max(abs(m) * 0.022, 0.001)
    scale = max(1.4826 * mad, floor)
    if high_only:
        score = (value - m) / scale
    elif low_only:
        score = (m - value) / scale
    else:
        score = abs(value - m) / scale
    return score >= z, score


def _tool_error(result):
    return not isinstance(result, dict) or "error" in result


def _verdict(alert, pillar, reasons, confidence=0.5):
    if alert:
        return Verdict(alert=True, confidence=max(0.55, min(0.99, confidence)), reason="; ".join(reasons), pillar=pillar)
    return Verdict(alert=False, confidence=0.25, reason="within balanced legal envelope", pillar=pillar)


def _sig(values):
    if values is None:
        return "__unknown__"
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    clean = sorted(str(v) for v in values if v is not None and str(v) != "")
    if not clean:
        return "__empty__"
    return "|".join(clean)


def _signature_anomaly(ctx, key, sig, *, min_total=4, dominance=0.66):
    table = ctx.state.setdefault("signatures", {}).setdefault(key, {})
    total = sum(table.values())
    if total < min_total or not table:
        return False, ""
    dominant_sig = None
    dominant_n = -1
    for k, v in table.items():
        if v > dominant_n:
            dominant_sig = k
            dominant_n = v
    if dominant_sig is not None and sig != dominant_sig and (dominant_n / float(total)) >= dominance:
        return True, dominant_sig
    return False, dominant_sig or ""


def _add_signature(ctx, key, sig, ok=True):
    if not ok:
        return
    table = ctx.state.setdefault("signatures", {}).setdefault(key, {})
    table[sig] = table.get(sig, 0) + 1


def check_data_batch(payload, ctx):
    prof = ctx.tools.batch_profile(payload.get("batch_id"))
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
        confidence = max(confidence, 0.97)
    if null_rate is not None and null_max is not None and null_rate > null_max:
        reasons.append("customer_id_null_rate_above_published_clean_band")
        confidence = max(confidence, 0.97)
    if mean_amount is not None and mean_min is not None and mean_max is not None and (mean_amount < mean_min or mean_amount > mean_max):
        reasons.append("mean_amount_outside_published_clean_band")
        confidence = max(confidence, 0.97)
    if staleness is not None and stale_max is not None and staleness > stale_max:
        reasons.append("batch_staleness_above_published_clean_band")
        confidence = max(confidence, 0.97)

    row_p = _range_pressure(row, row_min, row_max)
    mean_p = _range_pressure(mean_amount, mean_min, mean_max)
    null_p = _high_pressure(null_rate, null_max)
    stale_p = _high_pressure(staleness, stale_max)
    pressure_hits = [row_p >= 0.78, mean_p >= 0.78, null_p >= 0.78, stale_p >= 0.78]

    if not reasons:
        if row_p >= 0.88:
            reasons.append("row_count_tail_pressure")
            confidence = max(confidence, 0.74)
        if mean_p >= 0.88:
            reasons.append("mean_amount_tail_pressure")
            confidence = max(confidence, 0.74)
        if null_p >= 0.86:
            reasons.append("customer_id_null_rate_tail_pressure")
            confidence = max(confidence, 0.74)
        if stale_p >= 0.86:
            reasons.append("batch_staleness_tail_pressure")
            confidence = max(confidence, 0.74)
        if len([x for x in pressure_hits if x]) >= 2:
            reasons.append("multiple_batch_metrics_in_tail")
            confidence = max(confidence, 0.80)

    dyn = [
        ("checks.row_count", row, False, 2.75),
        ("checks.mean_amount", mean_amount, False, 2.75),
        ("checks.std_amount", std_amount, True, 2.55),
        ("checks.null_rate", null_rate, True, 2.55),
        ("checks.staleness", staleness, True, 2.55),
    ]
    for key, value, high_only, z in dyn:
        outlier, score = _robust_outlier(ctx, key, value, high_only=high_only, z=z)
        if outlier:
            reasons.append(key.split(".")[-1] + "_robust_history_outlier")
            confidence = max(confidence, min(0.94, 0.62 + score / 18.0))

    alert = bool(reasons)
    for key, value, _, _ in dyn:
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
        confidence = max(confidence, 0.97)
    elif delay is not None and max_delay is not None and delay >= 0.84 * max_delay:
        reasons.append("contract_freshness_tail_pressure")
        confidence = max(confidence, 0.73)

    key = "contracts.freshness_delay." + str(payload.get("contract_id", "all"))
    outlier, score = _robust_outlier(ctx, key, delay, high_only=True, z=2.55)
    if outlier:
        reasons.append("contract_freshness_robust_history_outlier")
        confidence = max(confidence, min(0.94, 0.62 + score / 18.0))

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
    for name in ("transform", "transform_name", "job", "job_name", "asset", "pipeline", "task", "run_group"):
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
    downstream_count = _num(graph.get("actual_downstream_count"))
    upstream_set = set(str(x) for x in upstream if x is not None and str(x) != "") if isinstance(upstream, (list, tuple, set)) else set([str(upstream)])

    reasons = []
    confidence = 0.55

    if duration is not None and max_duration is not None and duration > max_duration:
        reasons.append("lineage_runtime_above_published_clean_band")
        confidence = max(confidence, 0.97)
    elif duration is not None and max_duration is not None and duration >= 0.84 * max_duration:
        reasons.append("lineage_runtime_tail_pressure")
        confidence = max(confidence, 0.72)

    expected_upstream = _payload_expected_set(payload)
    if expected_upstream is not None and upstream_set != expected_upstream:
        reasons.append("lineage_upstream_mismatch_vs_event_expectation")
        confidence = max(confidence, 0.97)
    elif not upstream_set:
        reasons.append("lineage_missing_all_upstream_edges")
        confidence = max(confidence, 0.92)

    if downstream_count is not None and downstream_count <= 0:
        reasons.append("lineage_orphaned_output_count")
        confidence = max(confidence, 0.92)

    key = _lineage_key(payload)
    upstream_sig = _sig(upstream_set)
    up_anom, _ = _signature_anomaly(ctx, key + ".upstream_sig", upstream_sig)
    if up_anom:
        reasons.append("lineage_upstream_signature_changed_from_stable_history")
        confidence = max(confidence, 0.86)

    down_sig = str(int(downstream_count)) if downstream_count is not None and downstream_count == int(downstream_count) else str(downstream_count)
    down_anom, _ = _signature_anomaly(ctx, key + ".downstream_sig", down_sig)
    if down_anom:
        reasons.append("lineage_downstream_count_changed_from_stable_history")
        confidence = max(confidence, 0.84)

    outlier, score = _robust_outlier(ctx, key + ".duration_ms", duration, high_only=True, z=2.55)
    if outlier:
        reasons.append("lineage_runtime_robust_history_outlier")
        confidence = max(confidence, min(0.94, 0.62 + score / 18.0))
    outlier, score = _robust_outlier(ctx, key + ".downstream_count", downstream_count, z=2.55)
    if outlier:
        reasons.append("lineage_downstream_count_robust_history_outlier")
        confidence = max(confidence, min(0.93, 0.62 + score / 18.0))

    alert = bool(reasons)
    _add_history(ctx, key + ".duration_ms", duration, ok=not alert)
    _add_history(ctx, key + ".downstream_count", downstream_count, ok=not alert)
    _add_signature(ctx, key + ".upstream_sig", upstream_sig, ok=not alert)
    _add_signature(ctx, key + ".downstream_sig", down_sig, ok=not alert)
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
        confidence = max(confidence, 0.97)
    elif shift is not None and max_shift is not None and shift >= 0.64 * max_shift:
        reasons.append("feature_train_serving_skew_tail_pressure")
        confidence = max(confidence, 0.74)

    key = "feature.shift." + str(payload.get("feature_view", "all"))
    outlier, score = _robust_outlier(ctx, key, shift, high_only=True, z=2.50)
    if outlier:
        reasons.append("feature_shift_robust_history_outlier")
        confidence = max(confidence, min(0.94, 0.62 + score / 18.0))

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
        confidence = max(confidence, 0.97)
    if age is not None and max_age is not None and age > max_age:
        reasons.append("rag_corpus_age_above_published_clean_band")
        confidence = max(confidence, 0.97)

    if not reasons:
        if centroid_p >= 0.68:
            reasons.append("embedding_centroid_shift_tail_pressure")
            confidence = max(confidence, 0.74)
        if age_p >= 0.72:
            reasons.append("rag_corpus_age_tail_pressure")
            confidence = max(confidence, 0.73)
        if centroid_p >= 0.58 and age_p >= 0.58:
            reasons.append("combined_embedding_and_corpus_tail_pressure")
            confidence = max(confidence, 0.79)

    prefix = "embedding." + str(payload.get("corpus", "all"))
    for suffix, value in (("centroid", centroid), ("age", age)):
        outlier, score = _robust_outlier(ctx, prefix + "." + suffix, value, high_only=True, z=2.50)
        if outlier:
            reasons.append("embedding_" + suffix + "_robust_history_outlier")
            confidence = max(confidence, min(0.94, 0.62 + score / 18.0))

    alert = bool(reasons)
    _add_history(ctx, prefix + ".centroid", centroid, ok=not alert)
    _add_history(ctx, prefix + ".age", age, ok=not alert)
    return _verdict(alert, "ai_infra", reasons, confidence)
