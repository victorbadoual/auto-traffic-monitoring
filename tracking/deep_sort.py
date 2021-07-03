import os

import cv2
import numpy as np

from external.deep_sort import preprocessing
from external.deep_sort.detection import Detection
from external.deep_sort.generate_detections import create_box_encoder
from external.deep_sort.nn_matching import NearestNeighborDistanceMetric
from external.deep_sort.tracker import Tracker
from external.ioutracker.util import load_mot


def xyxy_to_xywh(bbox_xyxy):
    width = bbox_xyxy[2] - bbox_xyxy[0]
    height = bbox_xyxy[3] - bbox_xyxy[1]
    return bbox_xyxy[0], bbox_xyxy[1], width, height


def track(video_path, detection_path, output_path):
    video_capture = cv2.VideoCapture(video_path)

    detections_by_frames = load_mot(detection_path, with_classes=False)

    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 0.5

    model_filename = 'deep_sort_model_weights/mars-small128.pb'
    encoder = create_box_encoder(model_filename, batch_size=1)
    metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    frame_idx = 1
    for detections in detections_by_frames:
        success, frame = video_capture.read()
        if success is False:
            break
        bounding_boxes = np.asarray([detection['bbox'] for detection in detections])
        scores = np.asarray([detection['score'] for detection in detections])
        classes = np.asarray([detection['class'] for detection in detections], dtype=int)
        detection_ids = np.asarray([detection['detection_id'] for detection in detections], dtype=int)

        bboxes = []
        for box in bounding_boxes:
            bbox = xyxy_to_xywh(box)
            bboxes.append(bbox)

        bounding_boxes = np.asarray(bboxes)

        detections = [bounding_boxes, scores, classes]

        cur_path = os.path.dirname(__file__)
        class_labels_path = os.path.join(cur_path, '..', 'detection/labels', 'coco_labels.txt')
        class_labels = np.genfromtxt(class_labels_path, delimiter=',', dtype=str)
        allowed_classes = ['person', 'bicycle', 'car', 'motorbike', 'bus', 'truck']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(len(bounding_boxes)):
            class_indx = int(classes[i])
            class_name = class_labels[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        # delete detections that are not in allowed_classes
        bounding_boxes = np.delete(bounding_boxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bounding_boxes)
        detections = [Detection(bounding_box, score, detection_id, class_name, feature) for
                      bounding_box, score, detection_id, class_name, feature
                      in zip(bounding_boxes, scores, detection_ids, names, features) if score > 0.5]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            results.append(
                [frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3], track.class_name, track.detection_id])
        frame_idx += 1

    # Store results.
    f = open(output_path, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,-1,-1,-1,%s,%d' % (
            row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]), file=f)
    f.close()
