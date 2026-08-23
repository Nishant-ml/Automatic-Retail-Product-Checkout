# Automatic Retail Product Checkout — Group16

An automated checkout system that detects products on a cluttered shelf/checkout image using YOLOv8, then classifies each detected crop through embedding retrieval — either a VAE or DINOv2, backed by FAISS for similarity search. Built as a group project at IIT Kanpur, evaluated on the RPC (Retail Product Checkout) dataset.

Full report: `docs/Group16_Report.pdf`
Notes on the evaluation methodology: `docs/METRICS_NOTE.md`

## Approach

1. **YOLOv8** detects and crops individual products from the input image.
2. Each crop is turned into an embedding using one of two methods:
   - **VAE**: a conv encoder-decoder, using the latent mean as the embedding.
   - **DINOv2**: a fine-tuned ViT-S/14 student model (distillation + a product-similarity loss on top).
3. The embedding is matched against a FAISS database using k-NN (k=5) with majority voting to classify the product.

DINOv2 embeddings turn out more discriminative for visually similar products, at the cost of slower inference. The VAE pipeline is faster but weaker on fine-grained distinctions between similar-looking items — see `docs/METRICS_NOTE.md` for details on how we measured this and a couple of gotchas in reading the comparison table.

## Setup

```bash
pip install -r requirements.txt
```

Edit `config.yaml`:
- `dataset.root` → path to the RPC dataset (needs `val2019/`, `test2019/`, `instances_val2019.json`, `instances_test2019.json`)
- `dataset.yolo_yaml` → path to the YOLO data yaml used for training (only needed for `--with-detection-metrics`)
- `checkpoints.*` → paths to trained `yolo_best.pt`, `vae_best.pth`, `dino_student.pth` (train these using the notebooks in `notebooks/`)

## Usage

Build a product database and evaluate an embedder:

```bash
python src/run_pipeline.py --embedder dino --build-db --evaluate
python src/run_pipeline.py --embedder vae  --build-db --evaluate
```

Get detection mAP (shared across embedders, since the detector doesn't change):

```bash
python src/run_pipeline.py --embedder dino --with-detection-metrics
```

Run inference on a single image:

```bash
python src/infer_demo.py --image path/to/shelf.jpg --embedder dino
# -> outputs/demo_result.jpg
```

## Repo layout

```
config.yaml              paths and hyperparameters
requirements.txt
src/
  config.py              config loader
  models.py              VAE + DINOHead architectures
  embeddings.py          VAE/DINOv2 embedder interface
  detect.py              YOLO wrapper
  metrics.py             IoU + RPC benchmark metrics (ACD/mCCD/mCIoU)
  pipeline.py            database building + evaluation
  run_pipeline.py        CLI entry point
  infer_demo.py          single-image demo
notebooks/               training notebooks (YOLOv8, VAE, DINO)
docs/
  Group16_Report.pdf     full report
  METRICS_NOTE.md        notes on the evaluation methodology
outputs/                 FAISS indices, demo results (generated)
```

## Team

Sarthak Dumbre, Nishant, Vishal Junjare, Pranjal, Prafull Joshi, Esha Singh — Computer Science and Engineering, IIT Kanpur.
