import numpy as np

from external.detectron2.detectron2.engine.defaults import DefaultPredictor
from external.detectron2.detectron2.config import get_cfg


class Detectron2Wrapper:

    def __init__(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file("../external/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        self.cfg.MODEL.DEVICE = "cpu"
        self.cfg.MODEL.WEIGHTS = "detectron2_model_weights/model_final_a3ec72.pkl"
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
        self.predictor = DefaultPredictor(self.cfg)

    def detect(self, im):
        predictions = self.predictor(im)
        scores = predictions["instances"].scores.cpu().numpy()
        pred_boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        pred_classes = predictions["instances"].pred_classes.cpu().numpy()
        pred_masks = predictions["instances"].pred_masks.cpu().numpy()

        bbox_xcycwh, cls_conf, cls_ids, cls_mask = [], [], [], []

        for (box, _class, score, mask) in zip(pred_boxes, pred_classes, scores, pred_masks):
            x0, y0, x1, y1 = box
            bbox_xcycwh.append([x0, y0, (x1 - x0), (y1 - y0)])
            cls_conf.append(score)
            cls_ids.append(_class)
            cls_mask.append(mask)

        return np.array(bbox_xcycwh, dtype=np.float64), np.array(cls_conf), np.array(cls_ids), np.array(cls_mask)
