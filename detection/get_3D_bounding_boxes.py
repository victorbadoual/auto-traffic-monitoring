# Inspired from https://github.com/AubreyC/trajectory-extractor/tree/master/traj_ext/box3D_fitting

import math
import sys

import cv2
import numpy as np
import scipy.optimize as opt
import os

from camera.projection import back_projection

sys.path.append('../camera')


def get_3D_bounding_box_mask(optimisation_parameters, fixed_parameters, image_size, camera_parameters):
    yaw = optimisation_parameters[0]
    x = optimisation_parameters[1]
    y = optimisation_parameters[2]
    z = fixed_parameters[0]
    length = fixed_parameters[1]
    width = fixed_parameters[2]
    height = fixed_parameters[3]

    # Get projected coordinates of bounding box corners
    corners = get_3d_bounding_box_corners(yaw, x, y, z, length, width, height)

    image_point_corners = []
    for corner in corners:
        # Convert arrays to object points
        corner_point = np.zeros((1, len(corner), 3), np.float32)
        corner_point[0, :, :3] = np.asarray(corner, dtype='float64').reshape(-1, 3)
        image_point_corner, jacobian = cv2.projectPoints(corner_point, camera_parameters.rotation_vectors,
                                                         camera_parameters.translation_vectors,
                                                         camera_parameters.camera_matrix,
                                                         camera_parameters.distortion_coefficients)
        image_point_corner = (int(image_point_corner[0][0][0]), int(image_point_corner[0][0][1]))
        image_point_corners.append(image_point_corner)

    # Convert object points to arrays
    image_points = np.array([], np.int32)
    image_points.shape = (0, 2)
    for image_point_corner in image_point_corners:
        daz = np.array([image_point_corner[0], image_point_corner[1]], np.int32)
        daz.shape = (1, 2)
        image_points = np.append(image_points, daz, axis=0)

    # Find Convex Hull from the rectangles points
    hull = cv2.convexHull(image_points)

    mask = np.zeros((image_size[0], image_size[1]), np.int8)
    cv2.fillConvexPoly(mask, hull, 1, lineType=8, shift=0)

    mask = mask.astype(np.bool)

    return mask


def euler_angles_to_rotation_matrix(euler_angles):
    # euler_angles = (roll, pitch, yaw)
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, math.cos(euler_angles[0]), math.sin(euler_angles[0])],
                                  [0, -math.sin(euler_angles[0]), math.cos(euler_angles[0])]])

    rotation_matrix_y = np.array([[math.cos(euler_angles[1]), 0, -math.sin(euler_angles[1])],
                                  [0, 1, 0],
                                  [math.sin(euler_angles[1]), 0, math.cos(euler_angles[1])]])

    rotation_matrix_z = np.array([[math.cos(euler_angles[2]), math.sin(euler_angles[2]), 0],
                                  [-math.sin(euler_angles[2]), math.cos(euler_angles[2]), 0],
                                  [0, 0, 1]])

    rotation_matrix = np.dot(rotation_matrix_x, np.dot(rotation_matrix_y, rotation_matrix_z))
    return rotation_matrix


def get_3d_bounding_box_corners(yaw, x, y, z, length, width, height):
    # Get the position of the center of the bottom side:
    bottom_center = np.array([x, y, z])
    bottom_center.shape = (3, 1)

    # euler_angles = (roll, pitch, Yaw)
    euler_angles = [0.0, 0.0, yaw]
    # Rotate from one frame to another
    rotation_matrix = euler_angles_to_rotation_matrix(euler_angles)
    rotation_matrix_transpose = rotation_matrix.transpose()

    # Find the corners:

    # Corner 1
    translation_vector = np.array([length / 2, -width / 2, 0])
    translation_vector.shape = (3, 1)
    corner_1 = bottom_center + rotation_matrix_transpose.dot(translation_vector)
    # Corner 2
    translation_vector = np.array([length / 2, width / 2, 0])
    translation_vector.shape = (3, 1)
    corner_2 = bottom_center + rotation_matrix_transpose.dot(translation_vector)
    # Corner 3
    translation_vector = np.array([-length / 2, -width / 2, 0])
    translation_vector.shape = (3, 1)
    corner_3 = bottom_center + rotation_matrix_transpose.dot(translation_vector)
    # Corner 4
    translation_vector = np.array([-length / 2, width / 2, 0])
    translation_vector.shape = (3, 1)
    corner_4 = bottom_center + rotation_matrix_transpose.dot(translation_vector)
    # Corner 5
    translation_vector = np.array([length / 2, -width / 2, height])
    translation_vector.shape = (3, 1)
    corner_5 = bottom_center + rotation_matrix_transpose.dot(translation_vector)
    # Corner 6
    translation_vector = np.array([length / 2, width / 2, height])
    translation_vector.shape = (3, 1)
    corner_6 = bottom_center + rotation_matrix_transpose.dot(translation_vector)
    # Corner 7
    translation_vector = np.array([-length / 2, -width / 2, height])
    translation_vector.shape = (3, 1)
    corner_7 = bottom_center + rotation_matrix_transpose.dot(translation_vector)
    # Corner 8
    translation_vector = np.array([-length / 2, width / 2, height])
    translation_vector.shape = (3, 1)
    corner_8 = bottom_center + rotation_matrix_transpose.dot(translation_vector)

    # Create list from corner points
    corners_list = [corner_1, corner_2, corner_3, corner_4, corner_5, corner_6, corner_7, corner_8]
    return corners_list


def get_masks_overlap(mask_1, mask_2):
    # Overlap mask
    mask_overlap = np.logical_and(mask_1 == 1, mask_2 == 1)

    # Mask of region: Mask_i \ Overlap
    mask_count_1 = (np.logical_and(mask_1 == 1, mask_overlap == 0))
    mask_count_2 = (np.logical_and(mask_2 == 1, mask_overlap == 0))

    # Count the 1
    count_1 = np.count_nonzero(mask_count_1)
    count_2 = np.count_nonzero(mask_count_2)

    # Weight more the regions of mask_1 going out of the region of mask_2 (e.g we want mask_1 to be inside mask_2)
    count = 4 * count_1 + count_2

    return count, mask_count_1, mask_count_2


def cost_function(optimisation_parameters, image_size, camera_parameters, mask, fixed_parameters):
    bounding_box_3d_mask = get_3D_bounding_box_mask(optimisation_parameters, fixed_parameters, image_size,
                                                    camera_parameters)

    overlap_score, mask_count_1, mask_count_2 = get_masks_overlap(mask, bounding_box_3d_mask)
    return overlap_score


def get_3D_bounding_box(bounding_box, mask, class_name, image_size, camera_parameters):
    x0, y0, w, h = bounding_box
    bounding_box_center = ((x0 + w / 2), int(y0 + h / 2))
    projected_point = back_projection(bounding_box_center, camera_parameters)

    projected_point_3d = (projected_point[0], projected_point[1], 0)

    length = float(5)
    width = float(2)
    height = float(-1.6)
    if class_name == 'car':
        length = float(5)
        width = float(2)
        height = float(-1.6)
    if class_name == 'bicycle' or class_name == 'motorbike':
        length = float(3)
        width = float(1)
        height = float(-1.6)
    if class_name == 'person':
        length = float(0.25)
        width = float(0.43)
        height = float(-1.67)
    if class_name == 'truck':
        length = float(13.6)
        width = float(2.45)
        height = float(-4)
    if class_name == 'bus':
        length = float(12)
        width = float(2.55)
        height = float(-5)
    if class_name == 'tree':
        length = float(0.5)
        width = float(0.5)
        height = float(-3)
    if class_name == 'traffic light':
        length = float(0.5)
        width = float(0.5)
        height = float(-3)
    if class_name == 'stop sign':
        length = float(0.5)
        width = float(0.5)
        height = float(-2.5)
    if class_name == 'parking meter':
        length = float(0.5)
        width = float(0.5)
        height = float(-1.5)
    if class_name == 'fire hydrant':
        length = float(0.5)
        width = float(0.5)
        height = float(-1)

    fixed_parameters = [projected_point_3d[2], length, width, height]

    optimal_parameters = None
    for yaw_degree in range(0, 180, 60):

        yaw_radian = np.deg2rad(yaw_degree)

        optimisation_parameters = [yaw_radian, projected_point_3d[0], projected_point_3d[1]]

        parameters = opt.minimize(cost_function, optimisation_parameters, method='Powell',
                                  args=(image_size, camera_parameters, mask, fixed_parameters),
                                  options={'maxfev': 1000, 'disp': True})
        if optimal_parameters is None:
            optimal_parameters = parameters

        if parameters.fun < optimal_parameters.fun:
            optimal_parameters = parameters

    yaw = round(optimal_parameters.x[0], 4)
    x = round(optimal_parameters.x[1][0], 4)
    y = round(optimal_parameters.x[2][0], 4)
    z = round(fixed_parameters[0], 4)
    length = round(fixed_parameters[1], 4)
    width = round(fixed_parameters[2], 4)
    height = round(fixed_parameters[3], 4)

    return yaw, x, y, z, length, width, height


def get_3D_bounding_box_corners_on_image(yaw, x, y, z, length, width, height, camera_parameters):
    corners = get_3d_bounding_box_corners(yaw, x, y, z, length, width, height)

    image_point_corners = []
    for corner in corners:
        image_point_corner, jacobian = cv2.projectPoints(corner, camera_parameters.rotation_vectors,
                                                         camera_parameters.translation_vectors,
                                                         camera_parameters.camera_matrix,
                                                         camera_parameters.distortion_coefficients)
        image_point_corner = (int(image_point_corner[0][0][0]), int(image_point_corner[0][0][1]))
        image_point_corners.append(image_point_corner)

    return image_point_corners
