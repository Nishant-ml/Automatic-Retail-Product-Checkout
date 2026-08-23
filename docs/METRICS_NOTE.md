# Notes on Table II (Performance Comparison)

While putting the eval code together we found the Precision/Recall/mAP numbers in Table II weren't as directly comparable across rows as they look at first glance — worth documenting so it doesn't cause confusion later.

## What's going on

The **YOLOv8** row's Precision/Recall/mAP comes from Ultralytics' `model.val()` — standard detection mAP, predicted boxes matched to ground-truth boxes across all 200 classes.

The **+VAE** and **+DINOv2** rows' Precision/Recall/mAP come from a separate evaluation function that:
- only counts a detection as correct if IoU passes **and** the FAISS/kNN-retrieved label matches ground truth,
- ranks detections by retrieval cosine similarity rather than YOLO's detection confidence,
- feeds that into `average_precision_score` to get an "mAP".

So the two sets of numbers are measuring different things under the same column header — the +VAE/+DINOv2 rows are really scoring retrieval quality conditioned on a correct detection, not detection quality itself. That's why DINOv2 shows the best mAP@0.5 but the worst mAP@0.5:0.95 — it's not actually underperforming on detection, the metric formula changed between rows.

The ACD/mCCD/mCIoU numbers (the RPC benchmark metrics) don't have this problem — they're computed the same way for every row and are safe to compare directly.

## In `src/`

`src/pipeline.py` splits this into two clearly separate outputs:
- `detection_metrics()` — real detection mAP, computed once since the detector is shared across embedders.
- `retrieval_and_rpc_metrics()` — retrieval top-1/top-5 accuracy (labeled as such, not as "mAP") plus the RPC ACD/mCCD/mCIoU numbers.

If reporting a results table elsewhere, report detection mAP once and retrieval accuracy + RPC metrics per embedder, rather than putting them in the same Precision/Recall/mAP columns.
