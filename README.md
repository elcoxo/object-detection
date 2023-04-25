# Object detection with YOLOv5 and OpenCV
## Install guide 
```py 
git clone --recurse-submodules https://github.com/ultralytics/yolov5.git # clone
cd yolov5
pip install -r requirements.txt # YOLOv5
```
For input files `input/`

For output files `result/`

## Run
```py
python run.py [-h] [filename] [objectname]
```
e. g.

```py
python run.py 1.mp4 cat
```