"""Single YOLO loader used by every script in this repo.

This is the fix for the biggest issue found in the original submission:
each notebook pointed at a *different* uploaded YOLO checkpoint
(`.../khushalnikam/yolocheckp/...` for DINO, `.../nishant251110053/models/...`
for VAE, `runs/detect/train9/...` for the baseline). Since the baseline
row's detection metrics come from `model.val()` on one checkpoint, and the
VAE/DINO rows implicitly used detections from different checkpoints, the
three rows in the results table were never strictly comparable on the
detection side. Loading the detector once, here, from config.yaml makes
that mistake impossible to repeat.
"""
from ultralytics import YOLO


def load_detector(cfg):
    return YOLO(cfg["checkpoints"]["yolo"])


def detect_boxes(model, image_bgr, verbose=False):
    """Return list of (x1, y1, x2, y2, cls_id, conf) in absolute pixel coords."""
    results = model(image_bgr, verbose=verbose)
    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else [None] * len(xyxy)
        conf = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else [None] * len(xyxy)
        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, conf):
            boxes.append((int(x1), int(y1), int(x2), int(y2), int(c), float(s)))
    return boxes
