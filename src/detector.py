from pyroengine.core import Engine
from PIL import Image

engine = Engine()

im = Image.open("data/smoke.jpg").convert("RGB")

prediction = engine.predict(im)

