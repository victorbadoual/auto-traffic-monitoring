import cv2
import matplotlib.pyplot as plt
import numpy as np

from detection.get_3D_bounding_boxes import get_3D_bounding_box_corners_on_image

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]


def draw_2d_bounding_boxes(frame, bounding_boxes_tlbr, tracking_ids, class_names):
    for idx, (bounding_box, tracking_id, class_name) in enumerate(zip(bounding_boxes_tlbr, tracking_ids, class_names)):
        x1, y1, x2, y2 = [int(i) for i in bounding_box]
        color = colors[int(tracking_id) % len(colors)]
        color = [i * 255 for i in color]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, class_name + "-" + str(tracking_id), (int(x1), int(y1 - 5)), 0, 0.5, (255, 255, 255), 1)
    return frame


def draw_3d_bounding_boxes(frame, bounding_boxes, bounding_boxes_3d, tracking_ids, class_names, camera_parameters):
    for idx, (bounding_box, bounding_box_3d, tracking_id, class_name) in enumerate(
            zip(bounding_boxes, bounding_boxes_3d, tracking_ids, class_names)):
        (yaw, x, y, z, length, width, height) = bounding_box_3d
        corners_3D_bounding_box = get_3D_bounding_box_corners_on_image(yaw, x, y, z, length, width, height,
                                                                       camera_parameters)
        x1, y1, x2, y2 = [int(i) for i in bounding_box]

        color = colors[int(tracking_id) % len(colors)]
        color = [i * 255 for i in color]

        cv2.line(frame, corners_3D_bounding_box[0], corners_3D_bounding_box[1], color, 2)
        cv2.line(frame, corners_3D_bounding_box[0], corners_3D_bounding_box[2], color, 2)
        cv2.line(frame, corners_3D_bounding_box[1], corners_3D_bounding_box[3], color, 2)
        cv2.line(frame, corners_3D_bounding_box[2], corners_3D_bounding_box[3], color, 2)

        cv2.line(frame, corners_3D_bounding_box[4], corners_3D_bounding_box[5], color, 2)
        cv2.line(frame, corners_3D_bounding_box[4], corners_3D_bounding_box[6], color, 2)
        cv2.line(frame, corners_3D_bounding_box[5], corners_3D_bounding_box[7], color, 2)
        cv2.line(frame, corners_3D_bounding_box[6], corners_3D_bounding_box[7], color, 2)

        cv2.line(frame, corners_3D_bounding_box[0], corners_3D_bounding_box[4], color, 2)
        cv2.line(frame, corners_3D_bounding_box[1], corners_3D_bounding_box[5], color, 2)
        cv2.line(frame, corners_3D_bounding_box[2], corners_3D_bounding_box[6], color, 2)
        cv2.line(frame, corners_3D_bounding_box[3], corners_3D_bounding_box[7], color, 2)
        cv2.putText(frame, class_name + "-" + str(tracking_id), (int(x1), int(y1 - 5)), 0, 0.5, (255, 255, 255), 1)
    return frame
