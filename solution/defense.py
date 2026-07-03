from api import Verdict

MIN_N = 5
MAX_N = 120


def register(ctx):
    ctx.state.setdefault("series", {})
    ctx.state.setdefault("sigs", {})
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def n(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def b(ctx, key):
    return n(ctx.baseline.get(key))


def med(xs):
    ys = sorted(xs)
    ln = len(ys)
    if ln == 0:
        return 0.0
    mid = ln // 2
    return ys[mid] if ln % 2 else (ys[mid - 1] + ys[mid]) / 2.0


def rng_pressure(v, lo, hi):
    v, lo, hi = n(v), n(lo), n(hi)
    if v is None or lo is None or hi is None or hi <= lo:
        return 0.0
    return abs(v - (lo + hi) / 2.0) / ((hi - lo) / 2.0)


def high_pressure(v, lim):
    v, lim = n(v), n(lim)
    if v is None or lim is None or lim <= 0:
        return 0.0
    return v / lim


def add_hist(ctx, key, value, ok=True):
    value = n(value)
    if not ok or value is None:
        return
    arr = ctx.state.setdefault("series", {}).setdefault(key, [])
    arr.append(value)
    if len(arr) > MAX_N:
        del arr[: len(arr) - MAX_N]


def outlier(ctx, key, value, high=False, z=2.25, min_n=MIN_N):
    value = n(value)
    arr = ctx.state.setdefault("series", {}).get(key, [])
    if value is None or len(arr) < min_n:
        return False, 0.0
    m = med(arr)
    mad = med([abs(x - m) for x in arr])
    scale = max(1.4826 * mad, abs(m) * 0.018, 0.001)
    score = (value - m) / scale if high else abs(value - m) / scale
    return score >= z, score


def check_hist(ctx, specs, reasons, conf):
    for key, value, high, z in specs:
        ok, score = outlier(ctx, key, value, high=high, z=z)
        if ok:
            reasons.append(key.split(".")[-1] + "_history_outlier")
            conf = max(conf, min(0.95, 0.62 + score / 17.0))
    return conf


def add_hists(ctx, specs, ok):
    for key, value, _, _ in specs:
        add_hist(ctx, key, value, ok)


def verdict(alert, pillar, reasons, conf=0.55):
    if alert:
        return Verdict(True, max(0.55, min(0.99, conf)), "; ".join(reasons), pillar)
    return Verdict(False, 0.25, "within envelope", pillar)


def bad(res):
    return not isinstance(res, dict) or "error" in res


def sig(values):
    if values is None:
        return "unknown"
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    vals = sorted(str(x) for x in values if x is not None and str(x) != "")
    return "|".join(vals) if vals else "empty"


def sig_changed(ctx, key, value, min_total=3, dominance=0.55):
    table = ctx.state.setdefault("sigs", {}).setdefault(key, {})
    total = sum(table.values())
    if total < min_total or not table:
        return False
    top = max(table.values())
    dom = None
    for k, v in table.items():
        if v == top:
            dom = k
            break
    return dom is not None and value != dom and top / float(total) >= dominance


def add_sig(ctx, key, value, ok=True):
    if not ok:
        return
    table = ctx.state.setdefault("sigs", {}).setdefault(key, {})
    table[value] = table.get(value, 0) + 1


def check_data_batch(payload, ctx):
    p = ctx.tools.batch_profile(payload.get("batch_id"))
    if bad(p):
        return verdict(False, "checks", ["batch_profile_unavailable"], 0.1)

    row = n(p.get("row_count"))
    null = n((p.get("null_rate") or {}).get("customer_id"))
    mean = n(p.get("mean_amount"))
    std = n(p.get("std_amount"))
    stale = n(p.get("staleness_min"))
    row_min, row_max = b(ctx, "row_count_min"), b(ctx, "row_count_max")
    mean_min, mean_max = b(ctx, "mean_amount_min"), b(ctx, "mean_amount_max")
    null_max, stale_max = b(ctx, "null_rate_max"), b(ctx, "staleness_min_max")

    reasons, conf = [], 0.55
    if row is not None and row_min is not None and row_max is not None and (row < row_min or row > row_max):
        reasons.append("row_count_outside_baseline"); conf = max(conf, 0.97)
    if null is not None and null_max is not None and null > null_max:
        reasons.append("null_rate_outside_baseline"); conf = max(conf, 0.97)
    if mean is not None and mean_min is not None and mean_max is not None and (mean < mean_min or mean > mean_max):
        reasons.append("mean_amount_outside_baseline"); conf = max(conf, 0.97)
    if stale is not None and stale_max is not None and stale > stale_max:
        reasons.append("staleness_outside_baseline"); conf = max(conf, 0.97)

    row_p = rng_pressure(row, row_min, row_max)
    mean_p = rng_pressure(mean, mean_min, mean_max)
    null_p = high_pressure(null, null_max)
    stale_p = high_pressure(stale, stale_max)
    if not reasons:
        hits = [row_p >= 0.64, mean_p >= 0.64, null_p >= 0.64, stale_p >= 0.64]
        if row_p >= 0.74:
            reasons.append("row_tail_pressure"); conf = max(conf, 0.74)
        if mean_p >= 0.74:
            reasons.append("mean_tail_pressure"); conf = max(conf, 0.74)
        if null_p >= 0.70:
            reasons.append("null_tail_pressure"); conf = max(conf, 0.74)
        if stale_p >= 0.70:
            reasons.append("staleness_tail_pressure"); conf = max(conf, 0.74)
        if len([x for x in hits if x]) >= 2:
            reasons.append("combined_batch_tail_pressure"); conf = max(conf, 0.81)

    specs = [("checks.row", row, False, 2.35), ("checks.mean", mean, False, 2.35),
             ("checks.std", std, True, 2.20), ("checks.null", null, True, 2.20),
             ("checks.stale", stale, True, 2.20)]
    conf = check_hist(ctx, specs, reasons, conf)
    alert = bool(reasons)
    add_hists(ctx, specs, not alert)
    return verdict(alert, "checks", reasons, conf)


def check_contract_checkpoint(payload, ctx):
    d = ctx.tools.contract_diff(payload.get("contract_id"), payload.get("checkpoint_batch_id"))
    if bad(d):
        return verdict(False, "contracts", ["contract_diff_unavailable"], 0.1)
    delay = n(d.get("freshness_delay_min"))
    max_delay = b(ctx, "freshness_delay_max_min")
    violations = d.get("violations") or []
    reasons, conf = [], 0.55
    if violations:
        reasons.append("contract_violation"); conf = max(conf, 0.99)
    if delay is not None and max_delay is not None and delay > max_delay:
        reasons.append("contract_delay_outside_baseline"); conf = max(conf, 0.97)
    elif delay is not None and max_delay is not None and delay >= 0.66 * max_delay:
        reasons.append("contract_delay_tail_pressure"); conf = max(conf, 0.74)
    cid = str(payload.get("contract_id", "all"))
    specs = [("contracts." + cid, delay, True, 2.20), ("contracts.global", delay, True, 2.30)]
    conf = check_hist(ctx, specs, reasons, conf)
    alert = bool(reasons)
    add_hists(ctx, specs, not alert)
    return verdict(alert, "contracts", reasons, conf)


def lineage_key(payload):
    for k in ("transform", "transform_name", "job", "job_name", "asset", "pipeline", "task", "run_group"):
        if payload.get(k):
            return "lineage." + str(payload.get(k))
    return "lineage.all"


def expected_upstream(payload):
    for k in ("expected_upstream", "expected_upstreams", "declared_upstream", "declared_upstreams"):
        if k in payload:
            v = payload.get(k)
            if isinstance(v, (list, tuple, set)):
                return set(str(x) for x in v)
            if isinstance(v, str) and v:
                return set([v])
    return None


def check_lineage_run(payload, ctx):
    g = ctx.tools.lineage_graph_slice(payload.get("run_id"))
    if bad(g):
        return verdict(False, "lineage", ["lineage_unavailable"], 0.1)
    duration = n(g.get("duration_ms"))
    max_duration = b(ctx, "lineage_duration_ms_max")
    upstream = g.get("actual_upstream") or []
    up_set = set(str(x) for x in upstream if x is not None and str(x) != "") if isinstance(upstream, (list, tuple, set)) else set([str(upstream)])
    down = n(g.get("actual_downstream_count"))
    reasons, conf = [], 0.55
    if duration is not None and max_duration is not None and duration > max_duration:
        reasons.append("runtime_outside_baseline"); conf = max(conf, 0.97)
    elif duration is not None and max_duration is not None and duration >= 0.66 * max_duration:
        reasons.append("runtime_tail_pressure"); conf = max(conf, 0.73)
    exp = expected_upstream(payload)
    if exp is not None and up_set != exp:
        reasons.append("upstream_mismatch"); conf = max(conf, 0.97)
    elif not up_set:
        reasons.append("missing_upstream"); conf = max(conf, 0.92)
    if down is not None and down <= 0:
        reasons.append("orphaned_output"); conf = max(conf, 0.92)

    lk = lineage_key(payload)
    up_sig = sig(up_set)
    down_sig = str(int(down)) if down is not None and down == int(down) else str(down)
    if sig_changed(ctx, lk + ".up", up_sig):
        reasons.append("upstream_signature_change"); conf = max(conf, 0.86)
    if sig_changed(ctx, lk + ".down", down_sig):
        reasons.append("downstream_signature_change"); conf = max(conf, 0.86)
    specs = [(lk + ".duration", duration, True, 2.20), (lk + ".down", down, False, 2.25),
             ("lineage.global.duration", duration, True, 2.35), ("lineage.global.down", down, False, 2.35)]
    conf = check_hist(ctx, specs, reasons, conf)
    alert = bool(reasons)
    add_hists(ctx, specs, not alert)
    add_sig(ctx, lk + ".up", up_sig, not alert)
    add_sig(ctx, lk + ".down", down_sig, not alert)
    return verdict(alert, "lineage", reasons, conf)


def check_feature_materialization(payload, ctx):
    d = ctx.tools.feature_drift(payload.get("feature_view"), payload.get("batch_id"))
    if bad(d):
        return verdict(False, "ai_infra", ["feature_drift_unavailable"], 0.1)
    shift = n(d.get("mean_shift_sigma"))
    max_shift = b(ctx, "feature_mean_shift_sigma_max")
    reasons, conf = [], 0.55
    if shift is not None and max_shift is not None and shift > max_shift:
        reasons.append("feature_shift_outside_baseline"); conf = max(conf, 0.97)
    elif shift is not None and max_shift is not None and shift >= 0.46 * max_shift:
        reasons.append("feature_shift_tail_pressure"); conf = max(conf, 0.75)
    fv = str(payload.get("feature_view", "all"))
    specs = [("feature." + fv, shift, True, 2.15), ("feature.global", shift, True, 2.25)]
    conf = check_hist(ctx, specs, reasons, conf)
    alert = bool(reasons)
    add_hists(ctx, specs, not alert)
    return verdict(alert, "ai_infra", reasons, conf)


def check_embedding_batch(payload, ctx):
    d = ctx.tools.embedding_drift(payload.get("corpus"), payload.get("chunk_batch_id"))
    if bad(d):
        return verdict(False, "ai_infra", ["embedding_drift_unavailable"], 0.1)
    centroid = n(d.get("centroid_shift"))
    age = n(d.get("avg_doc_age_days"))
    max_centroid = b(ctx, "embedding_centroid_shift_max")
    max_age = b(ctx, "corpus_avg_doc_age_days_max")
    cp = high_pressure(centroid, max_centroid)
    ap = high_pressure(age, max_age)
    reasons, conf = [], 0.55
    if centroid is not None and max_centroid is not None and centroid > max_centroid:
        reasons.append("centroid_outside_baseline"); conf = max(conf, 0.97)
    if age is not None and max_age is not None and age > max_age:
        reasons.append("corpus_age_outside_baseline"); conf = max(conf, 0.97)
    if not reasons:
        if cp >= 0.50:
            reasons.append("centroid_tail_pressure"); conf = max(conf, 0.75)
        if ap >= 0.55:
            reasons.append("age_tail_pressure"); conf = max(conf, 0.74)
        if cp >= 0.40 and ap >= 0.40:
            reasons.append("combined_embedding_tail_pressure"); conf = max(conf, 0.80)
    corpus = str(payload.get("corpus", "all"))
    specs = [("embed." + corpus + ".centroid", centroid, True, 2.15),
             ("embed." + corpus + ".age", age, True, 2.15),
             ("embed.global.centroid", centroid, True, 2.25),
             ("embed.global.age", age, True, 2.25)]
    conf = check_hist(ctx, specs, reasons, conf)
    alert = bool(reasons)
    add_hists(ctx, specs, not alert)
    return verdict(alert, "ai_infra", reasons, conf)
