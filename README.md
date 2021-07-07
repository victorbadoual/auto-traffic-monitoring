# Visual scene understanding and tracking from traffic cameras

## Installation

* Install dependencies using conda from `environment.yml`
* Create `autonomous-traffic-monitoring/external/` folder
* [Install Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
  in `autonomous-traffic-monitoring/external/`
* Download and add Mask R-CNN model weights (R101-FPN)
  from [Detectron2 Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)
  in `autonomous-traffic-monitoring/detection/detetctron2-model-weights`
* Clone DeepSORT repository and put it in `autonomous-traffic-monitoring/external/`
* Clone IOU/VIOU Tracker repository and put it in `autonomous-traffic-monitoring/external/`
* Download and add [DeepSORT model weights](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6)
  in `autonomous-traffic-monitoring/tracking/deep-sort-model-weights/`
* [OPTIONAL (alternative camera calibration)] Install [monodepth2](https://github.com/nianticlabs/monodepth2) in and put
  model
  weights ([mono+stereo_640x192](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip))
  in `autonomous-traffic-monitoring/camera/models/`

## Run

### Fetch Videos

* `auto-traffic-monitoring/input/scripts/get_tfl_videos.py`
* `auto-traffic-monitoring/input/scripts/get_frames_from_video.py video.mp4`

### Camera Calibration

Get landmarks from location of camera in [Google Maps](http://maps.google.com/) and match points with frame.
Format `.csv` file as `u, v, X, Y, Y`

* Main method (recommended):
  `auto-traffic-monitoring/camera/run_camera.py --video_path input/videos/video.mp4 --landmarks_path input/landmarks/video_landmarks.csv --mode manual`
* Alternative method:
  `auto-traffic-monitoring/camera/run_camera.py --video_path input/videos/video.mp4 --landmarks_path input/landmarks/video_landmarks.csv --mode manual`

### Object Detection

`auto-traffic-monitoring/detection/run_detection.py --video_path input/videos/video.mp4 --camera_params results/video.mp4_camera_parameters.pickle`

### Object Tracking

`auto-traffic-monitoring/tracking/run_tracking.py --video_path input/videos/video.mp4 --detection results/video.mp4_detection.csv --camera_params results/video.mp4_camera_parameters.pickle`

### Speed Estimation

`auto-traffic-monitoring/tracking/run_speed_estimation.py --video_path  input/videos/video.mp4 --tracking results/video.mp4_tracking_iou.csv --camera_params results/video.mp4_camera_parameters.pickle`