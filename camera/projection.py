# Inspired from: https://github.com/AubreyC/trajectory-extractor/blob/0657dcaaf3c7270360dbbad5500aa628bc7787b1/traj_ext/camera_calib/calib_utils.py

import cv2
import numpy as np


def back_projection(image_point, camera_parameters):
    # Convert to homogeneous coordinates
    image_point = np.append(image_point, 1).reshape(3, 1)

    # Convert rotation vectors to rotation matrix
    rotation_matrix = cv2.Rodrigues(np.asarray(camera_parameters.rotation_vectors, dtype='float64'))[0]

    # From image plane to ground plane at camera center
    ground_plane_at_camera_center_over_s = np.linalg.inv(rotation_matrix).dot(
        np.linalg.inv(camera_parameters.camera_matrix).dot(image_point))
    ground_plane_at_camera_center_z = \
        rotation_matrix.T.dot(np.asarray(camera_parameters.translation_vectors, dtype='float64'))[2, 0]
    s = ground_plane_at_camera_center_z / ground_plane_at_camera_center_over_s[2, 0]
    ground_plane_at_camera_center = s * ground_plane_at_camera_center_over_s

    # From ground plane at camera center to ground plane at origin
    world_point = ground_plane_at_camera_center - rotation_matrix.T.dot(
        np.asarray(camera_parameters.translation_vectors, dtype='float64'))

    return world_point[0], world_point[1]
