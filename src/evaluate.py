"""Replay saved detector scores through the immediate and temporal rules.

Labels from the manifest are only used here, after both rules have decided.
Each clip is replayed with a fresh temporal history.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.temporal import DEFAULT_CONFIG, new_temporal_state, update_temporal  # noqa: E402

METHODS = ("baseline", "temporal")


def load_manifest(path, split="all"):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        onset = row.get("first_smoke_time_s", "").strip()
        row["first_smoke_time_s"] = float(onset) if onset else None
        if row["label"] not in ("smoke", "negative"):
            raise ValueError(f"{row['clip_id']}: label must be smoke or negative, got {row['label']!r}")
        if row["label"] == "smoke" and row["first_smoke_time_s"] is None:
            raise ValueError(f"{row['clip_id']}: smoke clips need first_smoke_time_s")
    return [r for r in rows if split == "all" or r["split"] == split]


def load_records(path):
    """Read detections.csv (run_detector.py) or predictions.csv (pipeline.py)."""
    records = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            valid = row["valid"] == "True"
            score = row.get("visual_score", "")
            records.setdefault(row.get("clip_id") or row["clip"], []).append({
                "sample_time_s": float(row["sample_time_s"]) if row.get("sample_time_s") else None,
                "source_time_s": float(row["source_time_s"]),
                "visual_score": float(score) if valid and score not in ("", "None") else None,
                "valid": valid,
                "error": row.get("error") or None,
                "inference_ms": float(row["inference_ms"]) if row.get("inference_ms") else None,
            })
    for rows in records.values():
        rows.sort(key=lambda r: r["source_time_s"])
    return records


def first_alerts(samples, config):
    """Return the source time of the first alert for each method."""
    state = new_temporal_state()
    first = {"baseline": None, "temporal": None}
    for s in samples:
        if first["baseline"] is None and s["valid"] and s["visual_score"] is not None \
                and s["visual_score"] >= config["threshold"]:
            first["baseline"] = s["source_time_s"]
        decision = update_temporal(state, s, config)
        if decision["alert_event_temporal"]:
            first["temporal"] = s["source_time_s"]
    return first


def categorise(label, onset, alert_time):
    if label == "negative":
        return ("false_alarm", None) if alert_time is not None else ("correct_no_alert", None)
    if alert_time is None:
        return "missed", None
    if alert_time < onset:
        return "premature", None
    return "detected", round(alert_time - onset, 2)


def summarise(per_clip, method):
    rows = [r for r in per_clip if not r["failed_run"]]
    count = {c: sum(r[f"{method}_category"] == c for r in rows)
             for c in ("detected", "premature", "missed", "false_alarm", "correct_no_alert")}
    smoke = sum(r["label"] == "smoke" for r in rows)
    negative = sum(r["label"] == "negative" for r in rows)
    alerts = count["detected"] + count["premature"] + count["false_alarm"]
    delays = [r[f"{method}_delay_s"] for r in rows if r[f"{method}_delay_s"] is not None]

    def ratio(num, den):
        return {"value": round(num / den, 3) if den else None, "numerator": num, "denominator": den,
                "note": None if den else "undefined (denominator is 0)"}

    return {
        "counts": count,
        "clips_evaluated": len(rows),
        "smoke_clips": smoke,
        "negative_clips": negative,
        # Premature alerts fire before smoke is visible, so they count as false alerts.
        "precision": ratio(count["detected"], alerts),
        "recall": ratio(count["detected"], smoke),
        "false_alarm_rate": ratio(count["false_alarm"], negative),
        "delay_s": {"mean": round(statistics.mean(delays), 2) if delays else None,
                    "max": max(delays) if delays else None, "n": len(delays)},
    }


def evaluate(manifest, records, config):
    per_clip = []
    for clip in manifest:
        samples = records.get(clip["clip_id"], [])
        valid = [s for s in samples if s["valid"]]
        timings = [s["inference_ms"] for s in valid if s["inference_ms"] is not None]
        row = {
            "clip_id": clip["clip_id"],
            "split": clip["split"],
            "label": clip["label"],
            "first_smoke_time_s": clip["first_smoke_time_s"],
            "samples": len(samples),
            "failed_samples": len(samples) - len(valid),
            "failed_run": not valid,
            "max_score": round(max(s["visual_score"] for s in valid), 4) if valid else None,
            "inference_ms_mean": round(statistics.mean(timings), 1) if timings else None,
            "inference_ms_max": round(max(timings), 1) if timings else None,
        }
        first = first_alerts(samples, config) if valid else {m: None for m in METHODS}
        for method in METHODS:
            category, delay = categorise(clip["label"], clip["first_smoke_time_s"], first[method])
            row[f"{method}_alert_s"] = first[method]
            row[f"{method}_category"] = "failed_run" if row["failed_run"] else category
            row[f"{method}_delay_s"] = delay
        per_clip.append(row)

    all_timings = [s["inference_ms"] for c in manifest for s in records.get(c["clip_id"], [])
                   if s["valid"] and s["inference_ms"] is not None]
    return {
        "config": config,
        "per_clip": per_clip,
        "summary": {m: summarise(per_clip, m) for m in METHODS},
        "failed_runs": sum(r["failed_run"] for r in per_clip),
        "inference_ms": {
            "mean": round(statistics.mean(all_timings), 1) if all_timings else None,
            "median": round(statistics.median(all_timings), 1) if all_timings else None,
            "max": round(max(all_timings), 1) if all_timings else None,
            "n": len(all_timings),
        },
    }


def print_report(report):
    cfg = report["config"]
    print(f"\nthreshold={cfg['threshold']}  window={cfg['window_s']}s  "
          f"min_positive={cfg['min_positive']}  model={cfg.get('model_version')}")
    print(f"{'clip':30} {'label':9} {'max score':>9}  {'immediate':24} {'temporal':24}")
    for r in report["per_clip"]:
        def cell(m):
            t = r[f"{m}_alert_s"]
            return r[f"{m}_category"] + (f" @ {t:.1f}s" if t is not None else "")
        print(f"{r['clip_id'][:30]:30} {r['label']:9} {r['max_score'] if r['max_score'] is not None else '-':>9}  "
              f"{cell('baseline'):24} {cell('temporal'):24}")
    for m in METHODS:
        s = report["summary"][m]

        def fmt(x):
            return f"{x['numerator']}/{x['denominator']}" + (f" = {x['value']}" if x["value"] is not None else " (undefined)")
        print(f"  {m:9} precision {fmt(s['precision']):18} recall {fmt(s['recall']):14} "
              f"false alarms {fmt(s['false_alarm_rate']):14} mean delay {s['delay_s']['mean']}")
    t = report["inference_ms"]
    print(f"  inference ms: mean {t['mean']}  median {t['median']}  max {t['max']}  (n={t['n']})  "
          f"failed runs: {report['failed_runs']}")


def save(report, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"threshold_{report['config']['threshold']}"
    with open(out_dir / f"per_clip_{tag}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report["per_clip"][0]))
        writer.writeheader()
        writer.writerows(report["per_clip"])
    with open(out_dir / f"summary_{tag}.json", "w") as f:
        json.dump({k: v for k, v in report.items() if k != "per_clip"}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    parser.add_argument("--records", default=str(ROOT / "results" / "detections" / "detections.csv"))
    parser.add_argument("--split", default="all", choices=["all", "dev", "test"])
    parser.add_argument("--threshold", type=float, nargs="+", default=[DEFAULT_CONFIG["threshold"]])
    parser.add_argument("--out", default=str(ROOT / "results" / "evaluation"))
    args = parser.parse_args()

    manifest = load_manifest(args.manifest, args.split)
    records = load_records(args.records)
    for threshold in args.threshold:
        report = evaluate(manifest, records, {**DEFAULT_CONFIG, "threshold": threshold})
        print_report(report)
        save(report, Path(args.out))
    print(f"\nSaved to {args.out}")
