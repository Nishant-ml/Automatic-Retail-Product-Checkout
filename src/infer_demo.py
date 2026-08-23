"""Single-image end-to-end demo: image in -> annotated image out.

This is the piece the original submission was missing. Three notebooks
full of training/eval code prove the models work, but there was no script
that just takes a photo and shows what the system actually does. Run:

    python src/infer_demo.py --image path/to/shelf.jpg --embedder dino
"""
import argparse
import json
import os
from collections import Counter

import cv2
import faiss

from config import get_device, load_config
from detect import detect_boxes, load_detector
from embeddings import build_embedder


def annotate(image, detections, index, db_labels, embedder, k, category_names=None):
    for (x1, y1, x2, y2, _cls, conf) in detections:
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        emb = embedder.embed(crop).reshape(1, -1)
        scores, idx = index.search(emb, k)
        neighbor_labels = [db_labels[i] for i in idx[0] if i != -1]
        if not neighbor_labels:
            continue

        pred_id = Counter(neighbor_labels).most_common(1)[0][0]
        sim = float(scores[0][0])
        name = category_names.get(pred_id, str(pred_id)) if category_names else str(pred_id)
        label = f"{name} ({sim:.2f})"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, max(0, y1 - h - 8)), (x1 + w, y1), (0, 0, 255), -1)
        cv2.putText(image, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image


def main():
    parser = argparse.ArgumentParser(description="Run product detection + retrieval on one image.")
    parser.add_argument("--image", required=True, help="Path to an input shelf/checkout image.")
    parser.add_argument("--embedder", choices=["vae", "dino"], default="dino")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="outputs/demo_result.jpg")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg)

    yolo_model = load_detector(cfg)
    embedder = build_embedder(args.embedder, cfg, device)

    index_path = cfg["faiss"][f"{args.embedder}_index"]
    labels_path = cfg["faiss"][f"{args.embedder}_labels"]
    if not (os.path.exists(index_path) and os.path.exists(labels_path)):
        raise FileNotFoundError(
            f"No FAISS database found at {index_path}. Run `python src/run_pipeline.py "
            f"--embedder {args.embedder} --build-db` first."
        )

    index = faiss.read_index(index_path)
    with open(labels_path) as f:
        db_labels = json.load(f)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    detections = detect_boxes(yolo_model, image)
    print(f"Detected {len(detections)} products.")

    result = annotate(image, detections, index, db_labels, embedder, cfg["embedding"]["knn_k"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, result)
    print(f"Saved annotated image -> {args.out}")


if __name__ == "__main__":
    main()
