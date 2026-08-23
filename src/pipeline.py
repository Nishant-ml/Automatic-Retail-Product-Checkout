"""End-to-end pipeline: build a FAISS product database, then evaluate.

METHODOLOGY FIX (see docs/METRICS_NOTE.md for the full write-up):
The original report's Table II put "Precision / Recall / mAP" for all
three rows (YOLOv8, +VAE, +DINOv2) in the same columns, but they were
computed two different ways:
  - YOLOv8 row: real detection mAP from ultralytics `model.val()`.
  - +VAE / +DINOv2 rows: a hand-rolled AP that scores a detection as
    correct only if IoU passes AND the FAISS/kNN retrieval label
    matches, ranked by retrieval cosine similarity (not YOLO's own
    detection confidence).

Those aren't the same metric, so the rows were never directly
comparable, and it's why the DINOv2 row can look better on mAP@0.5 but
worse on mAP@0.5:0.95 than the plain YOLOv8 baseline.

This module reports them as two clearly separate metric groups instead:
  1. detection_metrics()   -> real ultralytics mAP/precision/recall,
                               computed ONCE since all pipelines share
                               the same YOLO checkpoint.
  2. retrieval_metrics()   -> top-1 / top-5 retrieval accuracy of the
                               embedder against ground-truth crop labels
                               (an embedder-quality metric, not a
                               detection metric).
  3. rpc_metrics()         -> ACD / mCCD / mCIoU, the RPC benchmark
                               metrics, computed identically for every
                               embedder (this part of the original code
                               was already apples-to-apples).
"""
import json
import os
from collections import Counter, defaultdict

import cv2
import faiss
import numpy as np
from tqdm import tqdm

from detect import detect_boxes
from metrics import calculate_iou, compute_rpc_metrics


def _load_coco(json_path):
    with open(json_path) as f:
        data = json.load(f)
    img_id_to_name = {img["id"]: img["file_name"] for img in data["images"]}
    anns_by_img = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)
    return img_id_to_name, anns_by_img


def build_database(embedder, yolo_model, cfg, index_path, labels_path):
    """Detect products in the training pool, match crops to GT boxes by
    IoU, embed matched crops, and write a FAISS index + label file."""
    img_id_to_name, anns_by_img = _load_coco(cfg["dataset"]["train_json"])
    img_dir = cfg["dataset"]["train_img_dir"]
    max_images = cfg["eval"]["max_images"]
    iou_thresh = cfg["eval"]["iou_match_threshold"]

    items = list(img_id_to_name.items())
    if max_images:
        items = items[:max_images]

    all_embeddings, all_labels = [], []

    for img_id, file_name in tqdm(items, desc=f"Building DB ({os.path.basename(index_path)})"):
        image = cv2.imread(os.path.join(img_dir, file_name))
        if image is None:
            continue

        gt_anns = anns_by_img[img_id]
        for (x1, y1, x2, y2, _cls, _conf) in detect_boxes(yolo_model, image):
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            best_iou, best_label = 0.0, -1
            for ann in gt_anns:
                gx, gy, gw, gh = ann["bbox"]
                iou = calculate_iou([x1, y1, x2, y2], [gx, gy, gx + gw, gy + gh])
                if iou > best_iou:
                    best_iou, best_label = iou, ann["category_id"]

            if best_iou > iou_thresh:
                all_embeddings.append(embedder.embed(crop))
                all_labels.append(int(best_label))

    if not all_embeddings:
        raise RuntimeError("No embeddings were extracted — check dataset paths in config.yaml")

    all_embeddings = np.array(all_embeddings).astype("float32")
    index = faiss.IndexFlatIP(all_embeddings.shape[1])
    index.add(all_embeddings)
    faiss.write_index(index, index_path)
    with open(labels_path, "w") as f:
        json.dump(all_labels, f)

    print(f"Database built: {len(all_labels)} embeddings -> {index_path}")


def detection_metrics(yolo_model, cfg):
    """Real detection mAP/precision/recall via ultralytics' own
    validation — computed once, shared by every embedder since they all
    use the same detector checkpoint."""
    metrics = yolo_model.val(
        data=cfg["dataset"].get("yolo_yaml", None) or _infer_yaml(cfg),
        conf=0.25,
        iou=0.6,
        verbose=False,
    )
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def _infer_yaml(cfg):
    # Fallback if the user hasn't set dataset.yolo_yaml explicitly.
    guess = os.path.join(os.path.dirname(cfg["dataset"]["root"]), "rpc.yaml")
    if os.path.exists(guess):
        return guess
    raise FileNotFoundError(
        "Set dataset.yolo_yaml in config.yaml to the YOLO data yaml used for training."
    )


def retrieval_and_rpc_metrics(embedder, yolo_model, cfg, index_path, labels_path):
    """Retrieval top-1/top-5 accuracy (embedder quality) + RPC ACD/mCCD/mCIoU
    (pipeline output quality), computed the same way regardless of embedder."""
    index = faiss.read_index(index_path)
    with open(labels_path) as f:
        db_labels = json.load(f)

    img_id_to_name, anns_by_img = _load_coco(cfg["dataset"]["test_json"])
    img_dir = cfg["dataset"]["test_img_dir"]
    max_images = cfg["eval"]["max_images"]
    iou_thresh = cfg["eval"]["iou_match_threshold"]
    k = cfg["embedding"]["knn_k"]

    items = list(img_id_to_name.items())
    if max_images:
        items = items[:max_images]

    top1_correct, top5_correct, total_matched = 0, 0, 0
    acd_list, mccd_list, mciou_list = [], [], []

    for img_id, file_name in tqdm(items, desc="Evaluating"):
        image = cv2.imread(os.path.join(img_dir, file_name))
        if image is None:
            continue

        gt_anns = anns_by_img[img_id]
        gt_counts = Counter(a["category_id"] for a in gt_anns)
        pred_counts = Counter()

        for (x1, y1, x2, y2, _cls, _conf) in detect_boxes(yolo_model, image):
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            emb = embedder.embed(crop).reshape(1, -1)
            _d, idx = index.search(emb, k)
            neighbor_labels = [db_labels[i] for i in idx[0] if i != -1]
            if not neighbor_labels:
                continue
            pred_label = Counter(neighbor_labels).most_common(1)[0][0]
            pred_counts[pred_label] += 1

            best_iou, gt_label = 0.0, None
            for ann in gt_anns:
                gx, gy, gw, gh = ann["bbox"]
                iou = calculate_iou([x1, y1, x2, y2], [gx, gy, gx + gw, gy + gh])
                if iou > best_iou:
                    best_iou, gt_label = iou, ann["category_id"]

            if best_iou > iou_thresh and gt_label is not None:
                total_matched += 1
                if neighbor_labels[0] == gt_label:
                    top1_correct += 1
                if gt_label in neighbor_labels[:5]:
                    top5_correct += 1

        acd, mccd, mciou = compute_rpc_metrics(gt_counts, pred_counts)
        acd_list.append(acd)
        mccd_list.append(mccd)
        mciou_list.append(mciou)

    return {
        "retrieval_top1_acc": top1_correct / total_matched if total_matched else 0.0,
        "retrieval_top5_acc": top5_correct / total_matched if total_matched else 0.0,
        "acd": float(np.mean(acd_list)) if acd_list else 0.0,
        "mccd": float(np.mean(mccd_list)) if mccd_list else 0.0,
        "mciou": float(np.mean(mciou_list)) if mciou_list else 0.0,
    }
