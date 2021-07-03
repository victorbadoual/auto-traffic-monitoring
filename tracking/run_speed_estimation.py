import argparse
import os
import pickle
import sys
import statistics

from camera.projection import *
from collections import defaultdict

sys.path.append('../camera')
sys.path.append('../detection')


def find_speed_estimates(tracking_path, speed_estimates_output, camera_parameters, fps):
    tracking_file = open(tracking_path)
    tracks = []
    while True:
        line = tracking_file.readline()
        split = line.split(",")
        if not line:
            break
        tracks.append([int(split[0]), int(split[1]), float(split[2]), float(split[3])])
    tracking_file.close()

    ids = np.array(tracks)[:, 1]
    unique_ids = np.unique(ids)

    instanteneous_speeds = defaultdict(list)
    for i in range(len(unique_ids)):
        indexes = np.where(ids == unique_ids[i])[0]
        image_points = np.array(tracks)[indexes.astype(int), 2:4]
        frames = np.array(tracks)[indexes.astype(int), 0]
        previous_position = None
        previous_speed = None
        last_frame = None
        for frame_id, image_point in zip(frames, image_points):
            if frame_id > 75:
                continue
            current_position = np.asarray(back_projection(image_point, camera_parameters))
            if previous_position is None:
                previous_position = current_position
                last_frame = frame_id
                continue
            time_dif = frame_id - last_frame
            speed = (np.linalg.norm(np.subtract(current_position, previous_position)) * fps) / time_dif

            if previous_speed is None:
                previous_speed = speed
                previous_position = current_position
                last_frame = frame_id
                continue
            speed = 0.65 * previous_speed + (1 - 0.65) * speed
            print("Speed: " + str(speed * 3.6) + " km/h for object: " + str(unique_ids[i]))
            instanteneous_speeds[unique_ids[i]].append(speed)
            previous_position = current_position
            previous_speed = speed
            last_frame = frame_id


    for object_id, instanteneous_speeds_for_object in instanteneous_speeds.items():
        avg_speed = statistics.median(instanteneous_speeds_for_object)
        print("Object ID: {:d} has average speed: {} m/s or {} km/h.".format(int(object_id), avg_speed,
                                                                             avg_speed * 3.6))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--tracking", type=str)
    parser.add_argument("--camera_params", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cur_path = os.path.dirname(__file__)
    tracking_path = os.path.join(cur_path, '..', args.tracking)

    pickle_path = os.path.join(cur_path, '..', args.camera_params)
    with open(pickle_path, 'rb') as handle:
        camera_parameters = pickle.load(handle)

    video_path = os.path.join(cur_path, '..', args.video_path)
    video_name = os.path.basename(video_path)
    speed_estimates_output = os.path.join(cur_path, '..', 'results', video_name + "_speed_estimates.csv")

    video_capture = cv2.VideoCapture(video_path)
    # Get first frame of input video
    success, frame = video_capture.read()
    # find_road_contours(frame, road_output)

    find_speed_estimates(tracking_path, speed_estimates_output, camera_parameters, video_capture.get(cv2.CAP_PROP_FPS))
