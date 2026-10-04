import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Generator

BASE_DIR = Path(__file__).resolve().parent
PYRO_ENGINE_PATH = BASE_DIR / "external" / "pyro-engine"
if str(PYRO_ENGINE_PATH) not in sys.path:
    sys.path.append(str(PYRO_ENGINE_PATH))

def call_camera(video_source: str) -> Generator[Dict[str, Any], None, None]:
    cap = cv2.VideoCapture(str(video_source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        decoded_time_s = frame_index / fps
        yield {
            "frame": frame,                 
            "frame_index": frame_index,      
            "decoded_time_s": decoded_time_s 
        }
        frame_index += 1
    cap.release()

def call_detector(frame: np.ndarray, model_predictor: Any = None) -> Dict[str, Any]:
    if model_predictor is not None:
        results = model_predictor(frame)
        visual_score = results.get("score", 0.0)
        boxes = results.get("boxes", [])

    else:
        visual_score = 0.85
        boxes = [[100, 120, 300, 400]]

    return {
        "visual_score": visual_score,
        "boxes": boxes
    }

class TemporalModule:
    def __init__(self, score_threshold: float = 0.70, window_seconds: float = 2.0):
        self.score_threshold = score_threshold
        self.window_seconds = window_seconds
        self.history = []

    def call_temporal(self, visual_score: float, decoded_time_s: float) -> Dict[str, Any]:

        if visual_score is not None:
            self.history.append((decoded_time_s, visual_score))
        self.history = [item for item in self.history if item[0] >= (decoded_time_s - self.window_seconds)]
        scores = [score for t, score in self.history if score >= self.score_threshold]
        temporal_score = float(np.mean(scores)) if scores else 0.0
        persistence = (self.history[-1][0] - self.history[0][0]) if len(self.history) > 1 else 0.0
        decision = "ALERT" if temporal_score >= self.score_threshold and persistence >= 1.0 else "NO_ALERT"

        return {
            "temporal_score": temporal_score,
            "persistence": persistence,
            "decision": decision
        }


def run_pipeline(video_path: str):

    temporal = TemporalModule(score_threshold=0.70, window_seconds=2.0)
    predictor = None 

    for camera_output in call_camera(video_path):
        frame = camera_output["frame"]
        timestamp = camera_output["decoded_time_s"]
        frame_idx = camera_output["frame_index"]

        detector_output = call_detector(frame, model_predictor=predictor)
        visual_score = detector_output["visual_score"]
        boxes = detector_output["boxes"]
        temporal_output = temporal.call_temporal(
            visual_score=visual_score, 
            decoded_time_s=timestamp
        )
        print(
            f"Frame #{frame_idx:04d} | Time: {timestamp:.2f}s | "
            f"Visual Score: {visual_score:.2f} | "
            f"Temporal Score: {temporal_output['temporal_score']:.2f} | "
            f"Decision: {temporal_output['decision']}"
        )
        
if __name__ == "__main__":
    run_pipeline("test_video.mp4")