"""Single entry point for the whole build-database + evaluate flow.

Replaces running three separate Kaggle notebooks top to bottom with
different hardcoded paths. Usage:

    python src/run_pipeline.py --embedder vae  --build-db --evaluate
    python src/run_pipeline.py --embedder dino --evaluate   # reuse existing DB
"""
import argparse
import json

from config import get_device, load_config
from detect import load_detector
from embeddings import build_embedder
from pipeline import build_database, detection_metrics, retrieval_and_rpc_metrics


def main():
    parser = argparse.ArgumentParser(description="Build product DB and/or evaluate a pipeline.")
    parser.add_argument("--embedder", choices=["vae", "dino"], required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--build-db", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--with-detection-metrics", action="store_true",
                         help="Also run ultralytics model.val() (shared across embedders).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg)
    print(f"Device: {device}")

    yolo_model = load_detector(cfg)
    embedder = build_embedder(args.embedder, cfg, device)

    index_path = cfg["faiss"][f"{args.embedder}_index"]
    labels_path = cfg["faiss"][f"{args.embedder}_labels"]

    if args.build_db:
        build_database(embedder, yolo_model, cfg, index_path, labels_path)

    results = {}
    if args.with_detection_metrics:
        results["detection"] = detection_metrics(yolo_model, cfg)

    if args.evaluate:
        results[args.embedder] = retrieval_and_rpc_metrics(
            embedder, yolo_model, cfg, index_path, labels_path
        )

    if results:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
