import argparse
import os
import pickle

from detection.detectron2_wrapper import Detectron2Wrapper
from detection.get_3D_bounding_boxes import get_3D_bounding_box
from visualization.bounding_box_drawing import *
from visualization.utils import *

sys.path.append('../camera')


def tlwh_to_tlbr(bounding_box_tlwh):
    x, y, w, h = bounding_box_tlwh
    x1 = x
    x2 = int(x + w)
    y1 = y
    y2 = int(y + h)
    return x1, y1, x2, y2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--camera_params", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cur_path = os.path.dirname(__file__)
    video_path = os.path.join(cur_path, '../auto-traffic-monitoring', args.video_path)
    pickle_path = os.path.join(cur_path, '../auto-traffic-monitoring', args.camera_params)
    with open(pickle_path, 'rb') as handle:
        camera_parameters = pickle.load(handle)

    video_name = os.path.basename(video_path)
    detection_output_path = os.path.join(cur_path, '../auto-traffic-monitoring', 'results', video_name + "_detection.csv")
    detection_3d_bounding_boxes_output_path = os.path.join(cur_path, '../auto-traffic-monitoring', 'results',
                                                           video_name + "_detection_3d_bounding_boxes.csv")
    pickle_output_path = os.path.join(cur_path, '../auto-traffic-monitoring', 'results', video_name + "_detection.pickle")
    detection_output = open(detection_output_path, "w+")
    detection_3d_bounding_boxes_output = open(detection_3d_bounding_boxes_output_path, "w+")
    class_labels_path = os.path.join(cur_path, '../auto-traffic-monitoring', 'detection/labels', 'coco_labels.txt')
    class_labels = np.genfromtxt(class_labels_path, delimiter=',', dtype=str)

    video_capture = cv2.VideoCapture(video_path)
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_output_path = os.path.join(cur_path, '../auto-traffic-monitoring', 'results', video_name + "_detection.mp4")
    video_output = cv2.VideoWriter(video_output_path, fourcc, 20, (video_width, video_height))

    detector = Detectron2Wrapper()

    frame_count = 1
    detection_id = 0
    masks_dict = {}
    while True:
        success, frame = video_capture.read()
        if success is False:
            break
        bounding_boxes_tlwh, scores, class_ids, masks = detector.detect(frame)
        bounding_boxes_tlbr = []
        if bounding_boxes_tlwh is not None:
            bounding_boxes_tlwh[:, 3:] *= 1
            for idx, (bounding_box_tlwh, score, class_id, mask) in enumerate(
                    zip(bounding_boxes_tlwh, scores, class_ids, masks)):
                detection_output.write(
                    "%d,%d,%f,%f,%f,%f,%f,%f\n" % (
                        frame_count, class_id, bounding_box_tlwh[0], bounding_box_tlwh[1], bounding_box_tlwh[2],
                        bounding_box_tlwh[3], score,
                        detection_id))
                masks_dict[detection_id] = mask
                bounding_boxes_tlbr.append(tlwh_to_tlbr(bounding_box_tlwh))
                class_name = class_labels[class_id]
                yaw, x, y, z, length, width, height = get_3D_bounding_box(bounding_box_tlwh, mask, class_name,
                                                                          (video_height, video_width),
                                                                          camera_parameters)
                detection_3d_bounding_boxes_output.write(
                    "%d,%f,%f,%f,%f,%f,%f,%f\n" % (detection_id, yaw, x, y, z, length, width, height))
                corners_3D_bounding_box = get_3D_bounding_box_corners_on_image(yaw, x, y, z, length, width, height,
                                                                               camera_parameters)
                detection_id += 1
                # Draw 3d bounding boxes

                x1, y1, x2, y2 = [int(i) for i in bounding_box_tlwh]

                color = colors[int(class_id) % len(colors)]
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
                cv2.putText(frame, class_name + "-" + str(class_id), (int(x1), int(y1 - 5)), 0, 0.5, (255, 255, 255), 1)

            # frame = draw_bounding_boxes(frame, bbox_xyxy, classes)
            video_output.write(frame)
            cv2.imshow("detection", frame)
            cv2.waitKey(1)
        frame_count += 1

    with open(pickle_output_path, 'wb') as handle:
        pickle.dump(masks_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    detection_output.close()
    detection_3d_bounding_boxes_output.close()
    video_output.release()
