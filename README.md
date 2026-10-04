# FireWatch-AI-hackathon-2026-

How to clone our repository: 
```
git clone --recurse-submodules https://github.com/MrDopper/FireWatch-AI-hackathon-2026-.git
```

This code will allow you to interact with our subrepository that we are currently working: 
```
./.venv/bin/python -m pip install \
  ./external/pyro-engine/pyro-predictor \
  ./external/pyro-engine/pyro_camera_api/client \
  ./external/pyro-engine
```

Then to test the model use
```
./.venv/bin/python -m src.detector
```

To test the camera use:
```
./.venv/bin/python -m Code.pipeline "data/wildfire.mp4" --threshold 0.03
```