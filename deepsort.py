import os
import sys
import cv2
import time
import torch
import numpy as np
import json
import argparse
import warnings

# Extend sys.path for FastReID and ByteTrack
sys.path.append(os.path.join(os.path.dirname(__file__), 'thirdparty/fast-reid'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'thirdparty/ByteTrack'))

# Internal imports
from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results

# ByteTrack imports
from yolox.tracker.byte_tracker import BYTETracker

# Helper for IoU

def bbox_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0

class VideoTracker:
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in CPU mode, which may be very slow!", UserWarning)

        # Initialize detectors and trackers
        self.detector = build_detector(cfg, use_cuda=use_cuda, segment=args.segment)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        # ByteTrack init
        bt_args = type('BT', (), {})()
        bt_args.track_thresh = cfg.YOLOV11.SCORE_THRESH
        bt_args.track_buffer = 30
        bt_args.match_thresh = cfg.YOLOV11.NMS_THRESH
        self.byte_tracker = BYTETracker(bt_args, frame_rate=30)

        # GUI
        if args.display:
            try:
                cv2.namedWindow("test", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("test", args.display_width, args.display_height)
            except cv2.error:
                print("Display not supported, skipping GUI.")
                args.display = False

        # Video source
        self.vdo = cv2.VideoCapture(args.cam if args.cam != -1 else video_path)

    def __enter__(self):
        assert self.vdo.isOpened(), f"Cannot open {self.video_path}"
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output setup
        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))
            self.logger.info(f"Saving results to {self.args.save_path}")

        # Load class map
        with open(self.args.class_map, 'r') as f:
            self.idx_to_class = json.load(f)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            print(f"Exception: {exc_type}, {exc}")

    def run(self):
        results = []
        frame_idx = 0

        while self.vdo.grab():
            frame_idx += 1
            if frame_idx % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            if ori_im is None:
                break
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # 1) Detection
            if self.args.segment:
                bbox_xywh, confs, cls_ids, seg_masks = self.detector(im)
            else:
                bbox_xywh, confs, cls_ids = self.detector(im)

            if bbox_xywh is None or len(bbox_xywh) == 0:
                dets_xyxy = np.zeros((0,5))
            else:
                # convert xywh to xyxy for ByteTrack
                xyxy = np.zeros((len(bbox_xywh),4))
                xyxy[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2
                xyxy[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2
                xyxy[:,2] = bbox_xywh[:,0] + bbox_xywh[:,2]/2
                xyxy[:,3] = bbox_xywh[:,1] + bbox_xywh[:,3]/2
                dets_xyxy = np.hstack((xyxy, confs[:,None]))

            # 2) ByteTrack update
            bt_tracks = self.byte_tracker.update(dets_xyxy, (self.im_height, self.im_width), (self.im_height, self.im_width))
            bt_map = {t.track_id: t.tlbr for t in bt_tracks}

            # 3) DeepSORT update
            if bbox_xywh is None or len(bbox_xywh) == 0:
                ds_outputs = np.zeros((0,6))
            else:
                ds_outputs, _ = self.deepsort.update(bbox_xywh, confs, cls_ids, im)

            # 4) Fuse IDs
            final_outputs = []
            for det in ds_outputs:
                x1,y1,x2,y2,ds_id,cls_id = det.tolist()
                best_id, best_iou = ds_id, 0.3
                for bt_id, bt_box in bt_map.items():
                    iou = bbox_iou([x1,y1,x2,y2], bt_box)
                    if iou > best_iou:
                        best_iou, best_id = iou, bt_id
                final_outputs.append([x1,y1,x2,y2, best_id, cls_id])
            outputs = np.array(final_outputs) if final_outputs else np.zeros((0,6))

            # Draw
            if len(outputs) > 0:
                bbox_tlwh = []
                xyxy = outputs[:,:4]
                ids = outputs[:,4].astype(int)
                classes = outputs[:,5].astype(int)
                names = [self.idx_to_class.get(str(c), 'Unknown') for c in classes]
                ori_im = draw_boxes(ori_im, xyxy, names, ids, None)
                for box in xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(box))
                results.append((frame_idx-1, bbox_tlwh, ids, classes))

            # display/write
            if self.args.display:
                cv2.imshow('test', ori_im)
                if cv2.waitKey(1)==27: break
            if self.args.save_path:
                self.writer.write(ori_im)

            write_results(self.save_results_path, results, 'mot')
            self.logger.info(f"Frame {frame_idx} | dets {len(dets_xyxy)} | tracks {len(outputs)} | fps {1/(time.time()-start):.2f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--VIDEO_PATH', type=str, default='demo.avi')
    parser.add_argument('--config_yolov11', type=str, default='./configs/yolov11.yaml')
    parser.add_argument('--config_deepsort', type=str, default='./configs/deep_sort.yaml')
    parser.add_argument('--config_fastreid', type=str, default='./configs/fastreid.yaml')
    parser.add_argument('--class_map', type=str, default='football_classes.json')
    parser.add_argument('--segment', action='store_true')
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--frame_interval', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./output/')
    parser.add_argument('--cpu', dest='use_cuda', action='store_false', default=True)
    parser.add_argument('--camera', dest='cam', type=int, default=-1)
    parser.add_argument('--fastreid', action='store_true')
    parser.add_argument('--mmdet', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cfg = get_config()
    if not args.mmdet and not args.segment:
        cfg.merge_from_file(args.config_yolov11); cfg.USE_MMDET=False; cfg.USE_SEGMENT=False
    elif args.segment:
        cfg.merge_from_file(args.config_detection); cfg.USE_SEGMENT=True; cfg.USE_MMDET=False
    elif args.mmdet:
        cfg.merge_from_file(args.config_mmdetection); cfg.USE_MMDET=True; cfg.USE_SEGMENT=False
    cfg.merge_from_file(args.config_deepsort)
    if args.fastreid: cfg.merge_from_file(args.config_fastreid); cfg.USE_FASTREID=True
    else: cfg.USE_FASTREID=False
    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo:
        vdo.run()
