from external.ioutracker.iou_tracker import *
from external.ioutracker.util import load_mot, save_to_csv


def track(detection_path, output_path):
    detections = load_mot(detection_path, with_classes=False)
    tracks = track_iou(detections, 0, 0.5, 0.5, 2)
    save_to_csv(output_path, tracks, fmt='motchallenge')
