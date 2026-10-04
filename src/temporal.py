"""
Timestamped temporal verification for smoke scores.
"""
import math
from collections import deque

DEFAULT_CONFIG = {
    "sample_hz": 2.0,
    "window_s": 5.0,
    "threshold": 0.70,
    "min_positive": 5,
    "model_version": "yolo11s_rapid-raccoon_v8.1.0",
}

MONITORING = "Monitoring"
CANDIDATE = "Candidate"
REVIEW_ALERT = "Review alert"
UNAVAILABLE = "Unavailable"


def new_temporal_state(clip_id=None, station_id=None, run_id=None):
    return {
        "clip_id": clip_id,
        "station_id": station_id,
        "run_id": run_id,
        "history": deque(),
        "coverage_start_s": None,
        "last_slot_s": None,
        "temporal_alert": False,
        "alert": None,
    }


def reset_temporal_state(state):
    """Clear history and the latch, e.g. on replay or source change."""
    state.update(new_temporal_state(state["clip_id"], state["station_id"], state["run_id"]))


def _clear_coverage(state):
    state["history"].clear()
    state["coverage_start_s"] = None


def _slot_time(sample, period):
    if sample.get("sample_time_s") is not None:
        return float(sample["sample_time_s"])
    return round(float(sample["source_time_s"]) / period) * period


def _invalid_reason(sample, slot, last_slot, period):
    score = sample.get("visual_score")
    if not sample.get("valid", True):
        return f"Inference failed: {sample.get('error') or 'unknown error'}"
    if score is None:
        return "No visual score"
    if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return f"Score {score!r} is not a number in [0, 1]"
    source = sample.get("source_time_s")
    if source is None or not math.isfinite(float(source)):
        return "Missing source timestamp"
    if abs(float(source) - slot) > period / 2:
        return f"Frame time {float(source):.2f}s is outside slot {slot:.2f}s"
    if last_slot is not None:
        step = round((slot - last_slot) / period)
        if step < 0:
            return f"Out-of-order time: slot {slot:.2f}s after {last_slot:.2f}s"
        if step == 0:
            return f"Duplicate slot {slot:.2f}s"
        if step > 1:
            return f"Gap: {step - 1} missing slot(s) before {slot:.2f}s"
    return None


def update_temporal(state, sample, config=DEFAULT_CONFIG):
    """Consume one scheduled sample and return the temporal decision."""
    for key in ("clip_id", "station_id", "run_id"):
        if sample.get(key) is not None and sample[key] != state[key]:
            state[key] = sample[key]
            reset_temporal_state(state)

    period = 1.0 / config["sample_hz"]
    slots_per_window = round(config["window_s"] * config["sample_hz"])
    slot = _slot_time(sample, period)
    last_slot = state["last_slot_s"]
    invalid = _invalid_reason(sample, slot, last_slot, period)

    if invalid:
        if last_slot is None or slot > last_slot:
            state["last_slot_s"] = slot
        _clear_coverage(state)
        if sample.get("valid", True) and sample.get("visual_score") is not None and "Gap" in invalid:
            # A valid sample after a gap starts a fresh coverage interval.
            state["coverage_start_s"] = max(0.0, slot - period)
            state["history"].append({"sample_time_s": slot, "score": float(sample["visual_score"])})
        return _decision(state, UNAVAILABLE, invalid, [], 0, slots_per_window, False)

    state["last_slot_s"] = slot
    score = float(sample["visual_score"])
    if state["coverage_start_s"] is None:
        # Slot s stands for the interval (s - period, s]; source time starts at 0.
        state["coverage_start_s"] = max(0.0, slot - period)
    history = state["history"]
    history.append({"sample_time_s": slot, "score": score})
    window_start = slot - config["window_s"]
    eps = period / 10
    while history and history[0]["sample_time_s"] <= window_start + eps:
        history.popleft()

    window = list(history)
    positive = sum(x["score"] >= config["threshold"] for x in window)
    complete = state["coverage_start_s"] <= window_start + eps and len(window) == slots_per_window

    event = False
    if complete and positive >= config["min_positive"] and not state["temporal_alert"]:
        state["temporal_alert"] = True
        event = True
        reason = (f"{positive} of {slots_per_window} samples >= {config['threshold']} "
                  f"in ({window_start:.1f}s, {slot:.1f}s]")
        state["alert"] = {
            "source_time_s": float(sample["source_time_s"]),
            "sample_time_s": slot,
            "visual_score": score,
            "window": window,
            "positive_samples": positive,
            "model_version": config.get("model_version"),
            "reason": reason,
        }

    if state["temporal_alert"]:
        status = REVIEW_ALERT
        reason = "Possible smoke requires review: " + state["alert"]["reason"]
    elif positive:
        status = CANDIDATE
        need = "complete window" if not complete else f"{config['min_positive']} qualifying samples"
        reason = f"{positive} of {len(window)} samples qualify; waiting for {need}"
    else:
        status = MONITORING
        reason = "No alert under current rule."
    return _decision(state, status, reason, window, positive, slots_per_window, event)


def _decision(state, status, reason, window, positive, slots_per_window, event):
    return {
        "state": status,
        "reason": reason,
        "window_samples": len(window),
        "positive_samples": positive,
        "window_coverage": len(window) / slots_per_window,
        "temporal_alert": state["temporal_alert"],
        "alert_event_temporal": event,
        "alert": state["alert"],
    }


if __name__ == "__main__":
    state = new_temporal_state("demo")
    for i in range(1, 13):
        t = i * 0.5
        d = update_temporal(state, {"source_time_s": t, "visual_score": 0.9, "valid": True})
        print(f"{t:4.1f}s {d['state']:12} {d['positive_samples']}/{d['window_samples']} {d['reason']}")
