import os
import sys
import cv2
import time
import torch
import json
import argparse
import warnings

# Extend sys.path for FastReID
sys.path.append(os.path.join(os.path.dirname(__file__), 'thirdparty/fast-reid'))

# Internal imports
from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results


class VideoTracker:
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in CPU mode, which may be very slow!", UserWarning)

        # GUI display setup
        if args.display:
            try:
                cv2.namedWindow("test", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("test", args.display_width, args.display_height)
            except cv2.error:
                print("Display not supported in this environment. Skipping GUI.")
                args.display = False

        self.vdo = cv2.VideoCapture(args.cam if args.cam != -1 else video_path)
        self.detector = build_detector(cfg, use_cuda=use_cuda, segment=args.segment)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    def __enter__(self):
        assert self.vdo.isOpened(), f"Failed to open video source: {self.video_path}"

        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))
            self.logger.info(f"Saving results to {self.args.save_path}")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            print(f"Exception occurred: {exc_type}, {exc_value}")

    def run(self):
        results = []
        idx_frame = 0

        with open('football_classes.json', 'r') as f:
            idx_to_class = json.load(f)

        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start_time = time.time()
            _, ori_im = self.vdo.retrieve()
            if ori_im is None:
                break
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # Detection
            if self.args.segment:
                bbox_xywh, cls_conf, cls_ids, seg_masks = self.detector(im)
            else:
                bbox_xywh, cls_conf, cls_ids = self.detector(im)

            # Filter for 'person' class (COCO ID 2)
            mask = cls_ids == 2
            bbox_xywh = bbox_xywh[mask]
            cls_conf = cls_conf[mask]
            cls_ids = cls_ids[mask]

            if bbox_xywh is None or bbox_xywh.ndim != 2 or bbox_xywh.shape[0] == 0:
                outputs = torch.empty((0, 6))
                mask_outputs = None
            else:
                bbox_xywh[:, 2:] *= 1.2  # Slightly enlarge boxes

                if self.args.segment:
                    seg_masks = seg_masks[mask]
                    outputs, mask_outputs = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im, seg_masks)
                else:
                    outputs, _ = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im)
                    mask_outputs = None

            # Draw and save results
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                classes = outputs[:, -2]
                names = [idx_to_class.get(str(cls), "Unknown") for cls in classes]

                ori_im = draw_boxes(ori_im, bbox_xyxy, names, identities, mask_outputs if self.args.segment else None)

                for bb in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb))

                results.append((idx_frame - 1, bbox_tlwh, identities, classes))

            # Show or save output frame
            if self.args.display:
                cv2.imshow("test", ori_im)
                if cv2.waitKey(1) == 27:
                    break

            if self.args.save_path:
                self.writer.write(ori_im)

            # Save MOT-format results
            write_results(self.save_results_path, results, 'mot')

            end_time = time.time()
            self.logger.info(
                f"time: {end_time - start_time:.03f}s, fps: {1 / (end_time - start_time):.03f}, "
                f"detections: {bbox_xywh.shape[0] if bbox_xywh is not None else 0}, trackings: {len(outputs)}"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH", type=str, default='demo.avi')
    parser.add_argument("--config_mmdetection", type=str, default="./configs/mmdet.yaml")
    parser.add_argument("--config_detection", type=str, default="./configs/mask_rcnn.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--config_fastreid", type=str, default="./configs/fastreid.yaml")
    parser.add_argument("--config_yolov11", type=str, default="./configs/yolov11.yaml")
    parser.add_argument("--fastreid", action="store_true")
    parser.add_argument("--mmdet", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", dest="cam", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()

    # Configure detector based on flags
    if not args.mmdet and not args.segment:
        cfg.merge_from_file(args.config_yolov11)
        cfg.USE_MMDET = False
        cfg.USE_SEGMENT = False
    elif args.segment:
        cfg.merge_from_file(args.config_detection)
        cfg.USE_SEGMENT = True
        cfg.USE_MMDET = False
    elif args.mmdet:
        cfg.merge_from_file(args.config_mmdetection)
        cfg.USE_MMDET = True
        cfg.USE_SEGMENT = False

    # Always merge Deep SORT config
    cfg.merge_from_file(args.config_deepsort)

    # Optional: FastReID
    if args.fastreid:
        cfg.merge_from_file(args.config_fastreid)
        cfg.USE_FASTREID = True
    else:
        cfg.USE_FASTREID = False

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()
