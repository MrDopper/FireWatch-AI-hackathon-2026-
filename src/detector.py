from pyroengine.core import Engine
from PIL import Image
from pathlib import Path
#Getting data path
dir_path = Path(__file__).resolve().parents[1] / "data"

def loadDetector():
    return Engine(cache_folder=str(dir_path), conf_thresh=0.03, model_conf_thresh= 0.001)

def detect(engine, frame):
    
    prediction = engine.model(frame, {})
    score = max((float(box[4]) for box in prediction), default = 0.0)

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
    return {"visual_score": score, "boxes": boxes}
    
    

if __name__ == "__main__":
    print(dir_path) #Checking the current directory
    engine = loadDetector()
    for image in sorted(dir_path.glob("*")):
        #Skip image files that are not in the right format
        if not image.is_file() or image.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
            continue
        #This code will process any valid image
        with Image.open(image) as source: 
            frame = source.convert("RGB")
            prediction = detect(engine, frame)
            print(image.name)

            print("prediction: ", prediction["visual_score"])
            print("boxes: ", prediction["boxes"])


 
    



