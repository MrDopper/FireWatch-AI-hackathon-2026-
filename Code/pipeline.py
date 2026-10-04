import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Code.camera import CameraError, close_display, read_samples, show_frame  # noqa: E402
from Code.target_box import CLIPS, smoke_target_box  # noqa: E402
from src.detector import detect as model_detect  # noqa: E402
from src.detector import loadDetector  # noqa: E402
from src.temporal import DEFAULT_CONFIG, new_temporal_state, update_temporal  # noqa: E402

RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
STATE_COLOURS = {
    "Monitoring": (200, 200, 200),
    "Candidate": (0, 200, 255),
    "Review alert": RED,
    "Unavailable": (255, 120, 0),
}


def detect(engine, frame_bgr):
    """Run the Pyronear model on an OpenCV BGR frame and extract max visual score."""
    start = time.perf_counter()
    try:
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = model_detect(engine, image)

        boxes = result.get("boxes", [])

        # Max confidence score across all detected boxes in the frame
        if boxes:
            score = max(b[4] for b in boxes if len(b) >= 5)
        else:
            raw_score = result.get("visual_score")
            score = raw_score if raw_score is not None else 0.0

        valid, error = True, None
    except Exception as exc:
        score, boxes, valid, error = None, [], False, f"{type(exc).__name__}: {exc}"

    return {
        "visual_score": score,
        "boxes": boxes,
        "inference_ms": (time.perf_counter() - start) * 1000,
        "valid": valid,
        "error": error,
    }


def draw_overlay(frame, target_box, result, decision, record):
    """Draw the target box (red), model boxes (green) and the evidence panel."""
    out = frame.copy()
    scale = out.shape[1] / 1280
    thick = max(2, int(3 * scale))
    font = 0.8 * scale

    if target_box:
        x1, y1, x2, y2 = target_box
        cv2.rectangle(out, (x1, y1), (x2, y2), RED, thick)
        cv2.putText(
            out,
            "target: smoke plume",
            (x1 + thick * 3, y2 - thick * 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            RED,
            thick,
        )

    for x1, y1, x2, y2, conf in result["boxes"]:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, GREEN, thick)
        cv2.putText(
            out,
            f"model {conf:.3f}",
            (p1[0], max(int(30 * scale), p1[1] - thick * 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            GREEN,
            thick,
        )

    score = record["visual_score"]
    lines = [
        (
            f"t = {record['source_time_s']:.2f}s  (slot {record['sample_time_s']:.1f}s)",
            WHITE,
        ),
        (
            f"Visual model score: {'n/a' if score is None else f'{score:.3f}'}"
            f"  ({record['inference_ms']:.0f} ms)",
            WHITE,
        ),
        (f"Immediate alert: {'YES' if record['baseline_alert'] else 'no'}", WHITE),
        (
            f"Temporal: {decision['state']}  "
            f"({decision['positive_samples']} of {decision['window_samples']} samples qualify)",
            STATE_COLOURS[decision["state"]],
        ),
        (decision["reason"], STATE_COLOURS[decision["state"]]),
    ]
    line_h = int(34 * scale)
    pad = int(10 * scale)
    panel_w = int(820 * scale)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, pad * 2 + line_h * len(lines)), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(
            out,
            text,
            (pad, pad + line_h * (i + 1) - int(8 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font * 0.85,
            colour,
            max(1, thick - 1),
        )
    return out


def process_clip(video_path, engine, config=DEFAULT_CONFIG, show=True, annotated_dir=None):
    clip_id = os.path.basename(video_path)
    draw_target = CLIPS.get(clip_id) == "smoke"
    state = new_temporal_state(clip_id)
    baseline_alert = False
    records = []

    if annotated_dir:
        os.makedirs(annotated_dir, exist_ok=True)

    for sample in read_samples(video_path, sample_rate=config["sample_hz"]):
        frame = sample["frame"]
        result = detect(engine, frame)
        target_box = smoke_target_box(frame) if draw_target else None

        decision = update_temporal(
            state,
            {
                "clip_id": clip_id,
                "sample_time_s": sample["sample_time_s"],
                "source_time_s": sample["source_time_s"],
                "visual_score": result["visual_score"],
                "valid": result["valid"],
                "error": result["error"],
            },
            config,
        )

        alert_event_baseline = (
            result["valid"]
            and result["visual_score"] is not None
            and result["visual_score"] >= config["threshold"]
            and not baseline_alert
        )
        baseline_alert = baseline_alert or alert_event_baseline

        record = {
            "clip_id": clip_id,
            "frame_index": sample["frame_index"],
            "sample_time_s": round(sample["sample_time_s"], 2),
            "source_time_s": round(sample["source_time_s"], 3),
            "visual_score": result["visual_score"],
            "boxes": json.dumps(
                [[round(v, 1) for v in b[:4]] + [round(b[4], 4)] for b in result["boxes"]]
            ),
            "target_box": json.dumps(target_box) if target_box else "",
            "inference_ms": round(result["inference_ms"], 1),
            "valid": result["valid"],
            "error": result["error"],
            "baseline_alert": baseline_alert,
            "alert_event_baseline": alert_event_baseline,
            "temporal_alert": decision["temporal_alert"],
            "alert_event_temporal": decision["alert_event_temporal"],
            "state": decision["state"],
            "reason": decision["reason"],
            "window_samples": decision["window_samples"],
            "positive_samples": decision["positive_samples"],
        }
        records.append(record)

        print(
            f"Time={record['source_time_s']:.2f}s "
            f"Score={result['visual_score']} "
            f"State={decision['state']} "
            f"({decision['positive_samples']}/{decision['window_samples']})"
        )

        annotated = draw_overlay(frame, target_box, result, decision, record)
        if annotated_dir:
            cv2.imwrite(
                os.path.join(annotated_dir, f"{sample['sample_time_s']:05.1f}s.jpg"),
                cv2.resize(
                    annotated, (1280, int(annotated.shape[0] * 1280 / annotated.shape[1]))
                ),
            )
        if show and not show_frame(annotated):
            break

    return records


def save_csv(records, output_file="results/predictions.csv"):
    if not records:
        print("No records generated.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved CSV: {output_file}")


def run_dashboard():
    """Launch Streamlit Dashboard from pipeline.py."""
    import streamlit as st

    st.set_page_config(
        page_title="FireWatch AI - Wildfire Detection", page_icon="🔥", layout="wide"
    )

    st.title("🔥 FireWatch AI — Early Wildfire Smoke Detection")
    st.markdown(
        "Real-time video evaluation using Pyronear smoke detection and sliding-window temporal verification."
    )

    @st.cache_resource
    def get_detector_engine():
        return loadDetector()

    engine = get_detector_engine()

    if "running" not in st.session_state:
        st.session_state.running = False
    if "reset_requested" not in st.session_state:
        st.session_state.reset_requested = False
    if "events_log" not in st.session_state:
        st.session_state.events_log = []
    if "scores_history" not in st.session_state:
        st.session_state.scores_history = []

    st.sidebar.header("⚙️ Station & Video Controls")

    video_dir = Path("data")
    video_files = (
        list(video_dir.glob("*.mp4"))
        + list(video_dir.glob("*.avi"))
        + list(video_dir.glob("*.mov"))
    )
    video_options = (
        [str(f) for f in video_files] if video_files else ["data/Smoke Plume 4K Video.mp4"]
    )

    selected_video = st.sidebar.selectbox("Select Video Feed:", video_options)

    st.sidebar.subheader("Temporal Verification Parameters")
    threshold = st.sidebar.slider(
        "Visual Evidence Threshold", 0.01, 1.0, float(DEFAULT_CONFIG["threshold"]), 0.01
    )
    window_sec = st.sidebar.slider(
        "Verification Window (Sec)", 1, 10, int(DEFAULT_CONFIG["window_sec"]), 1
    )
    required_samples = st.sidebar.slider(
        "Required Positive Samples", 1, 10, int(DEFAULT_CONFIG["min_positive"]), 1
    )

    config = {
        **DEFAULT_CONFIG,
        "threshold": threshold,
        "window_sec": window_sec,
        "min_positive": required_samples,
    }

    col_b1, col_b2, col_b3 = st.sidebar.columns(3)
    if col_b1.button("▶️ Start"):
        st.session_state.running = True
    if col_b2.button("⏸️ Stop"):
        st.session_state.running = False
    if col_b3.button("🔄 Replay"):
        st.session_state.running = True
        st.session_state.reset_requested = True
        st.session_state.events_log = []
        st.session_state.scores_history = []

    elevenlabs_audio_path = "results/wildfire_warning_elevenlabs.mp3"

    top_col1, top_col2 = st.columns([2, 1])

    with top_col1:
        st.subheader("Live Video Stream & Detection Overlay")
        frame_placeholder = st.empty()

    with top_col2:
        st.subheader("System Status")
        status_metric = st.empty()
        evidence_metric = st.empty()
        alert_banner = st.empty()
        audio_placeholder = st.empty()

    st.divider()

    bottom_col1, bottom_col2 = st.columns([1, 1])

    with bottom_col1:
        st.subheader("📈 Real-Time Model Confidence Score")
        chart_placeholder = st.empty()

    with bottom_col2:
        st.subheader("📜 Review Event Log")
        log_placeholder = st.empty()

    if st.session_state.running and selected_video and os.path.exists(selected_video):
        clip_id = os.path.basename(selected_video)
        state = new_temporal_state(clip_id)
        audio_played = False

        for sample in read_samples(selected_video, sample_rate=config["sample_hz"]):
            if not st.session_state.running:
                break

            if st.session_state.reset_requested:
                st.session_state.reset_requested = False
                state = new_temporal_state(clip_id)

            frame = sample["frame"]
            result = detect(engine, frame)

            decision = update_temporal(
                state,
                {
                    "clip_id": clip_id,
                    "sample_time_s": sample["sample_time_s"],
                    "source_time_s": sample["source_time_s"],
                    "visual_score": result["visual_score"],
                    "valid": result["valid"],
                    "error": result["error"],
                },
                config,
            )

            annotated_frame = draw_overlay(
                frame,
                None,
                result,
                decision,
                {
                    "source_time_s": sample["source_time_s"],
                    "sample_time_s": sample["sample_time_s"],
                    "visual_score": result["visual_score"],
                    "inference_ms": result["inference_ms"],
                    "baseline_alert": result["valid"]
                    and (result["visual_score"] or 0) >= config["threshold"],
                },
            )

            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            current_state = decision["state"]
            if current_state == "Monitoring":
                status_metric.metric("Current State", "🟢 Monitoring", delta="Normal")
            elif current_state == "Candidate":
                status_metric.metric("Current State", "🟡 Candidate", delta="Verification Active")
            elif current_state == "Review alert":
                status_metric.metric(
                    "Current State",
                    "🔴 Review Alert",
                    delta="Smoke Confirmed",
                    delta_color="inverse",
                )
                alert_banner.error("🚨 WILDFIRE WARNING! PLEASE EVACUATE.")
                if os.path.exists(elevenlabs_audio_path) and not audio_played:
                    audio_placeholder.audio(elevenlabs_audio_path, autoplay=True)
                    audio_played = True
            else:
                status_metric.metric("Current State", "⚠️️ Unavailable", delta="Signal Lost")

            evidence_metric.markdown(
                f"**Evidence Samples:** {decision['positive_samples']} / {decision['window_samples']} required"
            )

            st.session_state.scores_history.append(
                {
                    "Time (s)": round(sample["source_time_s"], 2),
                    "Score": result["visual_score"]
                    if result["visual_score"] is not None
                    else 0.0,
                    "Threshold": config["threshold"],
                }
            )
            df_chart = pd.DataFrame(st.session_state.scores_history).set_index("Time (s)")
            chart_placeholder.line_chart(df_chart)

            if decision["alert_event_temporal"]:
                st.session_state.events_log.append(
                    {
                        "Timestamp (s)": round(sample["source_time_s"], 2),
                        "Event": "Review Alert Triggered",
                        "Score": round(result["visual_score"], 3),
                        "Reason": decision["reason"],
                    }
                )

            if st.session_state.events_log:
                log_placeholder.dataframe(
                    pd.DataFrame(st.session_state.events_log), use_container_width=True
                )
            else:
                log_placeholder.info("No alert events recorded yet.")

            time.sleep(0.01)
    else:
        if not st.session_state.running:
            status_metric.info("System Paused. Press ▶️ Start to begin monitoring.")


if __name__ == "__main__":
    # Check if executed via Streamlit CLI
    try:
        import streamlit as st

        if st.runtime.exists():
            run_dashboard()
            sys.exit(0)
    except Exception:
        pass

    # CLI Command Line Execution
    parser = argparse.ArgumentParser(description="Run FireWatch on one video clip.")
    parser.add_argument("video", nargs="?", default="data/Smoke Plume 4K Video.mp4")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIG["threshold"])
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG, "threshold": args.threshold}
    clip_stem = Path(args.video).stem.replace(" ", "_")

    try:
        records = process_clip(
            args.video,
            loadDetector(),
            config,
            show=not args.no_display,
            annotated_dir=f"results/annotated/{clip_stem}",
        )
    except CameraError as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        close_display()

    save_csv(records)
    print(f"Processed {len(records)} samples.")