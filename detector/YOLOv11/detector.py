import torch
import numpy as np
import cv2
from ultralytics import YOLO
from detector.YOLOv5.utils.general import xyxy2xywh
from detector.YOLOv5.utils.plots import Annotator, colors

class YOLOv11:
    def __init__(self, weight='yolov11.pt', conf_thres=0.25, iou_thres=0.45, device='cuda:0'):
        self.model = YOLO(weight)
        self.device = device
        self.model.fuse()
        self.names = self.model.names
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def __call__(self, im0, augment=False, save_result=False):
        results = self.model(im0, conf=self.conf_thres, iou=self.iou_thres, device=self.device)[0]
        pred = results.boxes.data.cpu().numpy()

        if len(pred) == 0:
            return (np.array([]), np.array([]), np.array([])) if not save_result else (np.array([]), np.array([]), np.array([]), im0)

        boxes = pred[:, :4]
        conf = pred[:, 4]
        cls = pred[:, 5]

        # Convert to xywh
        boxes_xywh = xyxy2xywh(torch.tensor(boxes)).numpy()

        if save_result:
            annotator = Annotator(im0, line_width=3, example=str(self.names))
            for xyxy_, c, cf in zip(boxes, cls.astype(int), conf):
                label = f"{self.names[c]} {cf:.2f}"
                annotator.box_label(xyxy_, label, color=colors(c, False))
            im0 = annotator.result()
            return (boxes_xywh, conf, cls, im0)

        return boxes_xywh, conf, cls
