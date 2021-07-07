import argparse
import os
import pickle
import sys
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np

import deep_sort
import iou
import viou
from visualization.bounding_box_drawing import draw_3d_bounding_boxes

sys.path.append('../camera')
sys.path.append('../detection')

from camera.projection import back_projection

MINIMUM_TRACK_LENGTH = 10
MINIMUM_NON_STATIC_POSITION = 15


def get_trajectories(tracking_path, detection_3d_bounding_boxes_path, trajectories_output):
    # Read CSV file
    tracking_file = open(tracking_path)
    tracks = []
    while True:
        line = tracking_file.readline()
        split = line.split(",")
        if not line:
            break
        tracks.append([int(split[1]), float(split[2]), float(split[3]), float(split[10])])
    tracking_file.close()

    detection_3d_bounding_boxes_file = open(detection_3d_bounding_boxes_path)
    dict_3d_bounding_boxes = defaultdict()
    while True:
        line = detection_3d_bounding_boxes_file.readline()
        split = line.split(",")
        if not line:
            break
        dict_3d_bounding_boxes[int(split[0])] = [float(split[1]), float(split[2]), float(split[3]), float(split[4]),
                                                 float(split[5]), float(split[6]), float(split[7])]
    detection_3d_bounding_boxes_file.close()

    # Get list of unique tracking ids
    tracking_ids = np.array(tracks)[:, 0]
    unique_ids = np.unique(tracking_ids)

    # Iterate over tracking data for each tracking id
    for i in range(len(unique_ids)):
        # Get tracking data corresponding to current tracking id
        indexes = np.where(tracking_ids == unique_ids[i])[0]

        # Skip tracks with less than MINIMUM_TRACK_LENGTH frames
        if len(indexes) < MINIMUM_TRACK_LENGTH:
            continue

        tl_image_points = np.array(tracks)[indexes.astype(int), 1:3]
        detection_ids = np.array(tracks)[indexes.astype(int), 3]

        # Project top left corner of bounding boxes to road plane
        tl_world_points = []
        for tl_image_point in tl_image_points:
            tl_world_point = back_projection(tl_image_point, camera_parameters)
            # Skip points outside of tracking zone
            if abs(tl_world_point[0]) > 100 or abs(tl_world_point[1]) > 100:
                continue
            tl_world_points.append(back_projection(tl_image_point, camera_parameters))
        tl_world_points = np.asarray(tl_world_points)

        bounding_boxes_bottom_center = []
        for detection_id in detection_ids:
            x, y = dict_3d_bounding_boxes[int(detection_id)][1], dict_3d_bounding_boxes[int(detection_id)][2]
            if abs(x) > 50 or abs(y) > 100:
                continue
            bounding_boxes_bottom_center.append((x, y))
        bounding_boxes_bottom_center = np.asarray(bounding_boxes_bottom_center)

        # Uncomment to use 2D bounding boxes for trajectories
        # bounding_boxes_bottom_center = tl_world_points

        # Remove objects static for a long time
        if len(bounding_boxes_bottom_center) > 0:
            max_distance_x = max(bounding_boxes_bottom_center[:, 0]) - min(bounding_boxes_bottom_center[:, 0])
            max_distance_y = max(bounding_boxes_bottom_center[:, 1]) - min(bounding_boxes_bottom_center[:, 1])
            if abs(max_distance_x) < MINIMUM_NON_STATIC_POSITION and abs(max_distance_y) < MINIMUM_NON_STATIC_POSITION:
                continue

        # Scatter projected points on plot
        if len(bounding_boxes_bottom_center) > 0:
            plt.scatter(bounding_boxes_bottom_center[:, 0], bounding_boxes_bottom_center[:, 1])
            plt.text(bounding_boxes_bottom_center[0, 0], bounding_boxes_bottom_center[0, 1], str(unique_ids[i]))

    # Format plot
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("vehicle/pedestrian trajectories")
    plt.savefig(trajectories_output)
    plt.close()


# For 3D Bounding Boxes:
# with open(detection_ids_path, 'rb') as handle:
#     masks_dict = pickle.load(handle)
# detection_ids = np.array(tracks)[indexes.astype(int), 6]


def get_tracking_video(tracking_path, detection_3d_bounding_boxes_path, video_path, video_output_path):
    # Read input video
    video_capture = cv2.VideoCapture(video_path)
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up output tracking video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_output = cv2.VideoWriter(video_output_path, fourcc, 20, (video_width, video_height))

    # Read CSV file
    tracking_file = open(tracking_path)
    tracks, tracks_classes = [], []
    while True:
        line = tracking_file.readline()
        split = line.split(",")
        if not line:
            break
        tracks.append([int(split[1]), float(split[0]),
                       float(split[2]), float(split[3]), float(split[4]), float(split[5]),
                       float(split[10])])
        tracks_classes.append(str(split[9]))
    tracking_file.close()

    detection_3d_bounding_boxes_file = open(detection_3d_bounding_boxes_path)
    dict_3d_bounding_boxes = defaultdict()
    while True:
        line = detection_3d_bounding_boxes_file.readline()
        split = line.split(",")
        if not line:
            break
        dict_3d_bounding_boxes[int(split[0])] = [float(split[1]), float(split[2]), float(split[3]), float(split[4]),
                                                 float(split[5]), float(split[6]), float(split[7])]
    detection_3d_bounding_boxes_file.close()

    # Get list of unique frames
    frame_ids = np.array(tracks)[:, 1]
    unique_ids = np.unique(frame_ids)

    # Iterate over tracking data for each frame
    frame_count = 0
    for i in range(len(unique_ids)):
        # Read frame from input video
        while frame_count < unique_ids[i]:
            success, frame = video_capture.read()
            if success is False:
                break
            frame_count += 1

        # Get tracking data corresponding to current frame
        indexes = np.where(frame_ids == unique_ids[i])[0]
        bounding_boxes_tlbr = np.array(tracks)[indexes.astype(int), 2:6]
        tracking_ids = np.array(tracks)[indexes.astype(int), 0]
        class_names = np.array(tracks_classes)[indexes.astype(int)]
        detection_ids = np.array(tracks)[indexes.astype(int), 6]

        bounding_boxes_3d = []
        for detection_id in detection_ids:
            bounding_boxes_3d.append(dict_3d_bounding_boxes[int(detection_id)])

        output_frame = draw_3d_bounding_boxes(frame, bounding_boxes_tlbr, bounding_boxes_3d, tracking_ids, class_names,
                                              camera_parameters)

        # Draw 2d bounding boxes with tracking id and class on the output frame
        # output_frame = draw_2d_bounding_boxes(frame, bounding_boxes_tlbr, tracking_ids, class_names)
        video_output.write(output_frame)
        cv2.imshow("tracking", output_frame)
        cv2.waitKey(1)

    video_output.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--frames_path", type=str)
    parser.add_argument("--detection", type=str)
    parser.add_argument("--camera_params", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cur_path = os.path.dirname(__file__)
    detection_path = os.path.join(cur_path, '..', args.detection)

    pickle_path = os.path.join(cur_path, '..', args.camera_params)
    with open(pickle_path, 'rb') as handle:
        camera_parameters = pickle.load(handle)

    video_path = os.path.join(cur_path, '..', args.video_path)
    frames_path = os.path.join(cur_path, '..', args.frames_path)
    video_name = os.path.basename(video_path)
    tracking_path_iou = os.path.join(cur_path, '..', 'results', video_name + "_tracking_iou.csv")
    tracking_path_viou = os.path.join(cur_path, '..', 'results', video_name + "_tracking_viou.csv")
    tracking_path_deep_sort = os.path.join(cur_path, '..', 'results', video_name + "_tracking_deep_sort.csv")
    trajectories_output_iou = os.path.join(cur_path, '..', 'results', video_name + "_trajectories_iou.png")
    trajectories_output_viou = os.path.join(cur_path, '..', 'results', video_name + "_trajectories_viou.png")
    trajectories_output_deep_sort = os.path.join(cur_path, '..', 'results', video_name + "_trajectories_deep_sort.png")
    video_output_path = os.path.join(cur_path, '..', 'results', video_name + "_tracking.mp4")
    detection_3d_bounding_boxes_path = os.path.join(cur_path, '..', 'results',
                                                    video_name + "_detection_3d_bounding_boxes.csv")

    get_tracking_video(tracking_path_deep_sort, detection_3d_bounding_boxes_path, video_path, video_output_path)

    get_trajectories(tracking_path_iou, detection_3d_bounding_boxes_path, trajectories_output_iou)
    get_trajectories(tracking_path_viou, detection_3d_bounding_boxes_path, trajectories_output_viou)
    get_trajectories(tracking_path_deep_sort, detection_3d_bounding_boxes_path, trajectories_output_deep_sort)
