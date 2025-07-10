import torch
import numpy as np
from ultralytics import YOLO
import cv2

class YOLOv11:
    def __init__(self, weight, conf_thres=0.25, iou_thres=0.45, max_det=100, device='cuda:0'):
        self.device = device
        self.model = YOLO(weight).to(device)
        self.model.fuse()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.class_names = self.model.names

    def __call__(self, im0, augment=False, save_result=False):
        im_rgb = im0 if im0.shape[2] == 3 else cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        results = self.model.predict(
            source=im_rgb,
            verbose=False,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            max_det=self.max_det,
            augment=augment
        )
        pred = results[0]
        boxes = pred.boxes.xywh.cpu().numpy() if pred.boxes is not None else np.array([])
        confs = pred.boxes.conf.cpu().numpy() if pred.boxes is not None else np.array([])
        cls_ids = pred.boxes.cls.cpu().numpy() if pred.boxes is not None else np.array([])

        return boxes, confs, cls_ids
