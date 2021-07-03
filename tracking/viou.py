from external.ioutracker.viou_tracker import *
from external.ioutracker.util import load_mot, save_to_csv


def track(frames_path, detection_path, output_path):
    detections = load_mot(detection_path, with_classes=False)
    tracks = track_viou(frames_path, detections, 0, 0.5, 0.5, 2, 1, 'NONE', 1)
    save_to_csv(output_path, tracks, fmt='motchallenge')