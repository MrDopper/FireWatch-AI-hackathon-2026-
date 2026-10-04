from pyroengine.core import Engine
from PIL import Image
from pathlib import Path
#Getting data path
input = Path(__file__).resolve().parents[1] / "data"

def loadDetector():
    return Engine(cache_folder=str(input), conf_thresh=0.03, model_conf_thresh= 0.001)

def detect(engine, frame):
    return engine.model(frame, {})

if __name__ == "__main__":
    print(input) #Checking the current directory
    engine = loadDetector()
    for image in sorted(input.glob("*")):
        if not image.is_file():
            continue
        with Image.open(image) as source:
            frame = source.convert("RGB")
            prediction = detect(engine, frame)
            confidence = engine.predict(source)
            
            print("prediction: ", prediction)
            print("final confidence: ", confidence)


 
    



