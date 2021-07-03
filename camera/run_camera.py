import argparse
import pickle
import os

# from automatic_calibration import *
from landmarks_calibration import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str)
    # Input should be CSV file (x, y, X, Y, Z)
    parser.add_argument("--landmarks_path", type=str)
    parser.add_argument("--frame_path", type=str)
    parser.add_argument("--mode", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cur_path = os.path.dirname(__file__)
    video_path = os.path.join(cur_path, '..', args.video_path)
    landmarks_path = os.path.join(cur_path, '..', args.landmarks_path)
    video_name = os.path.basename(video_path)
    pickle_path = os.path.join(cur_path, '..', 'results', video_name + "_camera_parameters.pickle")

    camera_parameters = landmarks_camera_calibration(video_path, landmarks_path)

    # if args.mode == "automatic-pose":
    #     frame_path = os.path.join(cur_path, '..', args.frame_path)
    #     output_dir = os.path.join(cur_path, '..', 'results')
    #     camera_parameters = automatic_camera_calibration(frame_path, output_dir, camera_parameters.camera_matrix,
    #                                                      camera_parameters.distortion_coefficients)

    with open(pickle_path, 'wb') as handle:
        pickle.dump(camera_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Camera Matrix:\n", camera_parameters.camera_matrix)
    print("Distortion Coefficients: ", camera_parameters.distortion_coefficients)
    print("Rotation Vectors:\n", camera_parameters.rotation_vectors)
    print("Translation Vectors:\n ", camera_parameters.translation_vectors)
