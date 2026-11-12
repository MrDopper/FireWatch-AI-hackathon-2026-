# FireWatch-AI-hackathon-2026-

### Description

This project was developed by our team during a 24-hour hackathon with the goal of building an early wildfire detection system.

Our system is designed to detect smoke at an early stage using a YOLO-based model provided by Pyronear. Rather than training an AI model from scratch, we used Pyronear's existing technology as the foundation and developed our own detection pipeline around it.

Despite the limited development time, we successfully built a working prototype that processes video frames, analyzes potential smoke, and identifies possible wildfire threats.

### Problem

The project is still in its early stages and requires further improvements. One of our biggest challenges was the model's low detection confidence, which was often below 20%.

Our system extracts individual frames from a video, converts them into RGB image data, and passes them to the YOLO-based detection model for analysis.

However, when processing large videos, our pipeline struggles to keep up with the incoming frame rate. This creates a processing bottleneck, causing some frames to be skipped.

As a result, the system may miss important visual information needed for consistent smoke detection. We are still investigating how much this processing limitation contributes to the model's low confidence scores.

### Future Goals

Our next goal is to optimize the video processing pipeline so that frame extraction and analysis can keep up with the video's frame rate.

We also plan to improve detection accuracy, reduce unnecessary frame skipping, and evaluate the model under different environmental conditions.

Ultimately, our goal is to develop a more reliable early wildfire detection system capable of identifying potential threats as quickly as possible.

# How to clone our repository: 
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
./.venv/bin/python -m src.pipeline "data/wildfire.mp4" --threshold 0.03
```
