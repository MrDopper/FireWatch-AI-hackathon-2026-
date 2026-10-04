import logging
from pathlib import Path
from PIL import Image
from pyroengine.core import Engine

# Set up directory paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

def loadDetector():
    """Load the Pyronear Engine with low thresholds to catch weak smoke plumes."""
    return Engine(cache_folder=str(DATA_DIR), conf_thresh=0.03, model_conf_thresh=0.001)

def detect(engine, frame):
    """
    Run Pyronear model inference on a PIL Image.
    Returns visual_score, scaled bounding boxes, valid status, and optional error.
    """
    try:
        prediction = engine.model(frame, {})
        
        # Max confidence score across all detected boxes (default to 0.0 if empty)
        score = max((float(box[4]) for box in prediction), default=0.0)

        width, height = frame.size
        boxes = [
            [
                float(x1) * width, 
                float(y1) * height,
                float(x2) * width,
                float(y2) * height,
                float(confidence)
            ]
            for x1, y1, x2, y2, confidence in prediction
        ]
        return {
            "visual_score": score, 
            "boxes": boxes, 
            "valid": True, 
            "error": None
        }
    except Exception as exc:
        return {
            "visual_score": None, 
            "boxes": [], 
            "valid": False, 
            "error": f"{type(exc).__name__}: {exc}"
        }

if __name__ == "__main__":
    print("Testing detector on data folder:", DATA_DIR)
    engine = loadDetector()
    
    for image_path in sorted(DATA_DIR.glob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
            continue
            
        with Image.open(image_path) as source: 
            frame = source.convert("RGB")
            res = detect(engine, frame)
            print(f"\nImage: {image_path.name}")
            print("Visual Score:", res["visual_score"])
            print("Boxes:", res["boxes"])
            print("Valid:", res["valid"])