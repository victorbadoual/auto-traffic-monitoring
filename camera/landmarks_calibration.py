# Inspired from: https://github.com/AubreyC/trajectory-extractor/blob/0657dcaaf3c7270360dbbad5500aa628bc7787b1/traj_ext/camera_calib/calib_utils.py

import csv

import cv2
import numpy as np
import scipy.optimize as opt

from parameters import CameraParameters


def landmarks_camera_calibration(video_path, landmarks_path):
    """
    Perform camera calibration using known landmarks.

    returns:
     - intrinsic parameters: camera matrix K.
     - extrinsic parameters: rotation vectors R and translation vectors t.
    """
    video_capture = cv2.VideoCapture(video_path)

    # Get first frame of input video
    success, frame = video_capture.read()

    # Fetch landmarks with pixel coordinates to world coordinates matches from csv
    landmarks_file = open(landmarks_path, 'r')
    landmarks = csv.reader(landmarks_file, delimiter=",")

    image_points, world_points = [], []

    for row in landmarks:
        image_points.append(np.asarray([(row[0], row[1])], dtype='float64'))
        world_points.append(np.asarray([(row[2], row[3], row[4])], dtype='float64'))

    # Convert input to object points (required by opencv)
    image_object_points = np.zeros((1, len(image_points), 2), np.float32)
    image_object_points[0, :, :2] = np.asarray(image_points, dtype='float64').reshape(-1, 2)
    world_object_points = np.zeros((1, len(world_points), 3), np.float32)
    world_object_points[0, :, :3] = np.asarray(world_points, dtype='float64').reshape(-1, 3)

    # Add object points to list for each frame (here we only use one)
    image_object_points_list, world_object_points_list = [], []
    image_object_points_list.append(image_object_points)
    world_object_points_list.append(world_object_points)

    ret_val, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = \
        estimate_camera_parameters(frame.shape, image_object_points, world_object_points)

    print("Camera Calibration Error: ", ret_val)

    landmarks_file.close()

    return CameraParameters(camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors)


def cost_function(parameters, frame_size, image_points, world_points):
    ret_val, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = estimate_camera_extrinsic_parameters(
        frame_size, parameters[0], image_points, world_points)
    return ret_val


def estimate_camera_parameters(frame_size, image_points, world_points):
    # Run the optimization to find the optimal intrinsic parameters (focal length)
    optimal_parameters = opt.minimize(cost_function,
                                      np.asarray([frame_size[1]]),
                                      constraints=({'type': 'ineq', 'fun': lambda x: x[0]}),
                                      args=(frame_size, image_points, world_points))

    focal_length = optimal_parameters.x[0]

    return estimate_camera_extrinsic_parameters(frame_size, focal_length, image_points, world_points)


def estimate_camera_extrinsic_parameters(frame_size, focal_length, image_points, world_points):
    # Set camera intrinsic parameters
    center = (frame_size[1] / 2, frame_size[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")

    # Assuming no lens distortion
    distortion_coefficients = np.zeros((4, 1))

    # Compute extrinsic parameters from landmarks matches
    success, rotation_vectors, translation_vectors = cv2.solvePnP(world_points, image_points, camera_matrix,
                                                                  distortion_coefficients, flags=cv2.SOLVEPNP_ITERATIVE)

    # Project world points to image plane according to the new estimated extrinsic parameters
    image_points_projected, jacobian = cv2.projectPoints(world_points, rotation_vectors, translation_vectors,
                                                         camera_matrix,
                                                         distortion_coefficients)
    image_points_projection = image_points_projected[:, 0]

    # Compute projection error
    ret_val = np.linalg.norm(np.subtract(image_points_projection, image_points))

    return ret_val, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors

# ---- PREVIOUS LANDMARK CAMERA CALIBRATION ----
# Camera calibration from opencv only accepts landmarks on a same plane and is made for calibration patterns
# ret_val, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera(
#     world_object_points_list, pixel_object_points_list, gray_color.shape[::-1], None, None)
