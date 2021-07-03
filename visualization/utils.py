import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from camera.projection import back_projection
from detection.detectron2_panoptic_wrapper import Detectron2PanopticWrapper

sys.path.append('../detection')

json_file = '../detection/labels/instances_val2017.json'

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]


def display_grid(frame, pixel_grid_points):
    frame_with_grid = frame
    for i in range(0, len(pixel_grid_points), 2):
        pixel_grid_point = pixel_grid_points[i][0]
        next_pixel_grid_point = pixel_grid_points[i + 1][0]
        cv2.line(frame_with_grid, (int(pixel_grid_point[0]), int(pixel_grid_point[1])),
                 (int(next_pixel_grid_point[0]), int(next_pixel_grid_point[1])), (226, 43, 138), thickness=2)
    cv2.imshow("road plane grid %s" % (frame_with_grid.shape,), frame_with_grid)
    cv2.waitKey(0)


# def draw_bounding_boxes(frame, bounding_boxes, identities, class_names):
#     for idx, bounding_box in enumerate(bounding_boxes):
#         x1, y1, x2, y2 = [int(i) for i in bounding_box]
#         track_id = int(identities[idx]) if identities is not None else 0
#         class_name = int(class_names[idx]) if class_names is not None else 0
#         class_name = class_name + 1
#         label = "Unknown"
#         with open(json_file, 'r') as COCO:
#             js = json.loads(COCO.read())
#             for categ in js['categories']:
#                 # print(type(categ['id']))
#                 if categ['id'] == class_name:
#                     label = categ['name']
#         color = colors[int(track_id) % len(colors)]
#         color = [i * 255 for i in color]
#
#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#         cv2.putText(frame, label + "-" + str(track_id), (int(x1), int(y1 - 5)), 0, 0.5, (255, 255, 255), 1)
#     return frame


def find_road_contours(frame, road_output, camera_parameters):
    panoptic_detector = Detectron2PanopticWrapper()
    bbox_xcycwh, scores, classes, masks, panoptic_seg, segments_info = panoptic_detector.detect(frame)

    panoptic_seg_contour = np.zeros((panoptic_seg.shape[0], panoptic_seg.shape[1]), dtype="uint8")
    for i in range(0, panoptic_seg.shape[0]):
        for j in range(0, panoptic_seg.shape[1]):
            class_id = segments_info[int(panoptic_seg[i, j].numpy()) - 1]['category_id']
            if Detectron2PanopticWrapper.metadata.stuff_classes[class_id] == 'road':
                panoptic_seg_contour[i, j] = 1
            else:
                panoptic_seg_contour[i, j] = 0
    contours, hierarchy = cv2.findContours(panoptic_seg_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((panoptic_seg_contour.shape[0], panoptic_seg_contour.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    # Show in a window
    cv2.imshow('Contours', drawing)
    # cv2.waitKey(0)
    contours.sort(key=lambda c: len(c), reverse=True)
    if len(contours[1]) > len(contours[0] / 2):
        contours = np.concatenate((np.reshape(contours[0], (-1, 2)), np.reshape(contours[1], (-1, 2))))
    else:
        contours = np.reshape(contours[0], (-1, 2))
    road_points = []
    for contour in contours:
        road_points.append(back_projection(contour, camera_parameters))
    road_points = np.asarray(road_points)
    plt.scatter(road_points[:, 0], road_points[:, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("vehicle/pedestrian trajectories")
    plt.savefig(road_output)
