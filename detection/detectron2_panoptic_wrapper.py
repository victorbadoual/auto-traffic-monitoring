import cv2
import numpy as np
from detectron2.data import MetadataCatalog
from external.detectron2.detectron2.config import CfgNode as CN
from external.detectron2.detectron2.config import get_cfg
from external.detectron2.detectron2.engine.defaults import DefaultPredictor
from external.detectron2.detectron2.utils.visualizer import Visualizer, ColorMode


def add_deeplab_config(cfg):
    """
    Add config for DeepLab.
    """
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Used for `poly` learning rate schedule.
    cfg.SOLVER.POLY_LR_POWER = 0.9
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING = 0.0
    # Loss type, choose from `cross_entropy`, `hard_pixel_mining`.
    cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE = "hard_pixel_mining"
    # DeepLab settings
    cfg.MODEL.SEM_SEG_HEAD.PROJECT_FEATURES = ["res2"]
    cfg.MODEL.SEM_SEG_HEAD.PROJECT_CHANNELS = [48]
    cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS = 256
    cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS = [6, 12, 18]
    cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT = 0.1
    cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV = False
    # Backbone new configs
    cfg.MODEL.RESNETS.RES4_DILATION = 1
    cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 2, 4]
    # ResNet stem type from: `basic`, `deeplab`
    cfg.MODEL.RESNETS.STEM_TYPE = "deeplab"


def add_panoptic_deeplab_config(cfg):
    """
    Add config for Panoptic-DeepLab.
    """
    # Reuse DeepLab config.
    add_deeplab_config(cfg)
    # Target generation parameters.
    cfg.INPUT.GAUSSIAN_SIGMA = 10
    cfg.INPUT.IGNORE_STUFF_IN_OFFSET = True
    cfg.INPUT.SMALL_INSTANCE_AREA = 4096
    cfg.INPUT.SMALL_INSTANCE_WEIGHT = 3
    cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC = False
    # Optimizer type.
    cfg.SOLVER.OPTIMIZER = "ADAM"
    # Panoptic-DeepLab semantic segmentation head.
    # We add an extra convolution before predictor.
    cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS = 256
    cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K = 0.2
    # Panoptic-DeepLab instance segmentation head.
    cfg.MODEL.INS_EMBED_HEAD = CN()
    cfg.MODEL.INS_EMBED_HEAD.NAME = "PanopticDeepLabInsEmbedHead"
    cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES = ["res2", "res3", "res5"]
    cfg.MODEL.INS_EMBED_HEAD.PROJECT_FEATURES = ["res2", "res3"]
    cfg.MODEL.INS_EMBED_HEAD.PROJECT_CHANNELS = [32, 64]
    cfg.MODEL.INS_EMBED_HEAD.ASPP_CHANNELS = 256
    cfg.MODEL.INS_EMBED_HEAD.ASPP_DILATIONS = [6, 12, 18]
    cfg.MODEL.INS_EMBED_HEAD.ASPP_DROPOUT = 0.1
    # We add an extra convolution before predictor.
    cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS = 32
    cfg.MODEL.INS_EMBED_HEAD.CONVS_DIM = 128
    cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.INS_EMBED_HEAD.NORM = "SyncBN"
    cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT = 200.0
    cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT = 0.01
    # Panoptic-DeepLab post-processing setting.
    cfg.MODEL.PANOPTIC_DEEPLAB = CN()
    # Stuff area limit, ignore stuff region below this number.
    cfg.MODEL.PANOPTIC_DEEPLAB.STUFF_AREA = 2048
    cfg.MODEL.PANOPTIC_DEEPLAB.CENTER_THRESHOLD = 0.1
    cfg.MODEL.PANOPTIC_DEEPLAB.NMS_KERNEL = 7
    cfg.MODEL.PANOPTIC_DEEPLAB.TOP_K_INSTANCE = 200
    # If set to False, Panoptic-DeepLab will not evaluate instance segmentation.
    cfg.MODEL.PANOPTIC_DEEPLAB.PREDICT_INSTANCES = True
    cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV = False
    # This is the padding parameter for images with various sizes. ASPP layers
    # requires input images to be divisible by the average pooling size and we
    # can use `MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY` to pad all images to
    # a fixed resolution (e.g. 640x640 for COCO) to avoid having a image size
    # that is not divisible by ASPP average pooling size.
    cfg.MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY = -1
    # Only evaluates network speed (ignores post-processing).
    cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED = False


class Detectron2PanopticWrapper:
    metadata = MetadataCatalog.get("coco_2017_val_panoptic_separated")

    def __init__(self):
        self.cfg = get_cfg()
        # add_panoptic_deeplab_config(self.cfg)
        self.cfg.merge_from_file("../external/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        self.cfg.MODEL.DEVICE = "cpu"
        self.cfg.MODEL.WEIGHTS = "../detection/detectron2_model_weights/model_final_cafdb1.pkl"
        self.predictor = DefaultPredictor(self.cfg)

    def detect(self, im):
        predictions = self.predictor(im)

        panoptic_seg, segments_info = predictions["panoptic_seg"]
        v = Visualizer(im[:, :, ::-1],
                       metadata=self.metadata,
                       scale=1.0,
                       instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                       )
        v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        cv2.imshow('/Users/victorbadoual/PycharmProjects/auto-traffic-monitoring/input/scripts/frames/frame100.jpg',
                   v.get_image()[:, :, ::-1])
        cv2.waitKey(5)

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

        return np.array(bbox_xcycwh, dtype=np.float64), np.array(cls_conf), np.array(cls_ids), np.array(cls_mask), \
               panoptic_seg.to("cpu"), segments_info
