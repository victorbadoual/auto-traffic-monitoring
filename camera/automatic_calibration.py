import math
import os

import cv2
import numpy as np
import pyransac3d as pyrsc
from numpy.linalg import svd

from detection.detectron2_panoptic_wrapper import Detectron2PanopticWrapper
from external.monodepth2 import test_simple
from parameters import CameraParameters


# Inspired from https://stackoverflow.com/a/18968498
def plane_fit(points):
    """
    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """
    points = np.reshape(points, (np.shape(points)[0], -1))  # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                   points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T)  # Could also use np.cov(x) here.
    return ctr, svd(M)[0][:, -1]


# Inspired from https://math.stackexchange.com/questions/1956699/getting-a-transformation-matrix-from-a-normal-vector
def project_to_plane(d, normal):
    unit_normal = normal.T / (math.sqrt(math.pow(normal[0], 2) + math.pow(normal[1], 2) + math.pow(normal[2], 2)))
    translation_vectors = (-d / (
        math.sqrt(math.pow(normal[0], 2) + math.pow(normal[1], 2) + math.pow(normal[2], 2)))) * unit_normal
    quotient = math.sqrt(math.pow(unit_normal[0], 2) + math.pow(unit_normal[1], 2))
    rotation_matrix = np.array([[unit_normal[1] / quotient, - unit_normal[0] / quotient, 0],
                                [unit_normal[0] * unit_normal[2] / quotient,
                                 unit_normal[1] * unit_normal[2] / quotient, - quotient],
                                [unit_normal[0], unit_normal[1], unit_normal[2]]])
    return translation_vectors, rotation_matrix


def automatic_camera_calibration(frame_path, output_dir, camera_matrix, distortion_coefficients,
                                 model="mono+stereo_640x192"):
    test_simple.test_simple(model, frame_path, output_dir)
    base_path = str(os.path.splitext(os.path.basename(frame_path))[0])
    depth_image_path = os.path.join(output_dir, base_path + '_disp.npy')
    depth_image = np.load(depth_image_path)

    frame = cv2.imread(frame_path)

    panoptic_detector = Detectron2PanopticWrapper()
    bbox_xcycwh, scores, classes, masks, panoptic_seg, segments_info = panoptic_detector.detect(frame)

    road_indexes = []
    for i in range(0, panoptic_seg.shape[0]):
        for j in range(0, panoptic_seg.shape[1]):
            class_id = segments_info[int(panoptic_seg[i, j].numpy()) - 1]['category_id']
            if Detectron2PanopticWrapper.metadata.stuff_classes[class_id] == 'road':
                road_indexes.append([i, j])

    depth_image = depth_image[0][0]
    road_points = []
    for i in range(0, len(road_indexes)):
        x = road_indexes[i][0]
        y = road_indexes[i][1]
        road_points.append([x, y, depth_image[road_indexes[i][0]][road_indexes[i][1]]])

    camera_points = []
    for point in road_points:
        camera_point = np.linalg.inv(camera_matrix) * np.asmatrix(point).T
        camera_point = [camera_point[0], -camera_point[1], camera_point[2]]
        camera_points.append(camera_point)

    camera_points = np.asarray(camera_points)
    camera_points = np.reshape(camera_points, (camera_points.shape[0], camera_points.shape[1]))

    # Ransac plane estimation
    plane = pyrsc.Plane()
    best_eq, best_inliers = plane.fit(camera_points)
    [a, b, c, d] = best_eq

    # SVD plane estimation
    # point_on_plane, n = plane_fit(camera_points)
    # [a, b, c] = n

    # Compute rotation matrix from u, v and w vectors of the road plane
    n = [a, b, c]
    u = [b, -a, 0]
    v = np.cross(n, u)
    w = np.cross(u, v)
    rotation_matrix = np.asarray([u, v, w])

    # Choose centroid for the translation vectors
    camera_points = camera_points[best_inliers]
    x = [p[0] for p in camera_points]
    y = [p[1] for p in camera_points]
    z = [p[2] for p in camera_points]
    translation_vectors = [[sum(x) / len(camera_points)], [sum(y) / len(camera_points)],
                           [- sum(z) / len(camera_points)]]

    translation_vectors = np.asarray(translation_vectors)
    rotation_matrix = np.asarray(rotation_matrix)

    rotation_vectors = cv2.Rodrigues(np.asmatrix(rotation_matrix, dtype='float64'))[0]

    return CameraParameters(camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors)
