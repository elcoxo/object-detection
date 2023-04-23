import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

vid = "./input/1.mp4"
# Inference
results = model(vid)

# Results
results.show()  # or .show(), .save(), .crop(), .pandas(), etc.