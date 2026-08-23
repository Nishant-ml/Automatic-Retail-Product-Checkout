# Automatic Retail Product Checkout via Detection & Embedding-Based Retrieval

An automated checkout system designed to localize, identify, and count dense products in cluttered retail environments using the **Retail Product Checkout (RPC)** dataset (200 SKUs).

Unlike traditional fixed-classifier pipelines that require complete model retraining whenever new inventory arrives, this architecture decouples object localization from product identification through an **Embedding-Based Retrieval paradigm** powered by **FAISS** vector search.

---

## 📌 Key Highlights

* **Scalable Zero-Retraining Pipeline:** New SKUs are registered simply by indexing visual embeddings into the database, eliminating the need for periodic supervised model retraining.


* **Two-Stage Modular Architecture:** Utilizes **YOLOv8** for multi-object spatial localization and crops, paired with deep representation encoders for feature extraction.


* **Comparative Embedding Encoders:** Evaluates a convolutional **Variational Autoencoder (VAE)** against a self-supervised **DINOv2 (Vision Transformer)** student model.


* **High-Accuracy Retrieval:** The **YOLOv8 + DINOv2** pipeline achieves **0.921 mAP@0.5** and **81.5% precision**, significantly improving fine-grained discrimination between visually similar product packagings.



---

## 🏗️ System Architecture

```text
Input Cluttered Image
         │
         ▼
┌─────────────────────────────────┐
│   Object Detection (YOLOv8)     │ ──► Multi-Object Bounding Boxes & Crops
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│      Feature Representation     │
│   (DINOv2 ViT-S/14 vs. VAE)     │ ──► High-Dimensional Latent Embeddings
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Similarity Search & Ranking   │
│     (FAISS Cosine Similarity)   │ ──► k-NN (k=5) + Majority Voting
└─────────────────────────────────┘
         │
         ▼
Final Itemized Bill & Product Counts

```

---

## 🔬 Methodological Comparison

* **Baseline YOLOv8 (Direct Classification):** Single-stage direct bounding box and class prediction. Highly optimized for speed (28.57 FPS), but inflexible to inventory updates and prone to misclassifications on visually identical packaging.


* **YOLOv8 + VAE:** Encodes crops into a continuous latent space using mean/variance parameterization trained on reconstruction loss. Fast (4.65 FPS) and compact, but struggles with fine-grained packaging nuances.


* **YOLOv8 + DINOv2:** Leverages Vision Transformer self-attention to capture global contextual tokens and fine-grained visual cues. Delivers state-of-the-art discriminative retrieval under dense occlusions (0.921 mAP@0.5).



---

## 📊 Benchmark Results

Evaluated on the RPC dataset across standard object detection and RPC-specific counting metrics (Average Counting Distance $\text{ACD}\downarrow$, mean Category Counting Distance $\text{mCCD}\downarrow$, and mean Category Intersection over Union $\text{mCIoU}\uparrow$):

| Method | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | ACD (&darr;) | mCCD (&darr;) | mCIoU (&uarr;) | FPS (&uarr;) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8 Baseline** | 0.777 | 0.784 | 0.865 | 0.634 | 2.87 | 0.96 | 65.67 | **28.57** |
| **YOLOv8 + VAE** | 0.788 | 0.793 | 0.877 | 0.737 | **0.13** | **0.22** | **66.97** | 4.65 |
| **YOLOv8 + DINOv2** | **0.815** | **0.818** | **0.921** | **0.714** | 2.87 | 0.36 | 63.13 | 3.12 |

---

## 📁 Repository Structure

```text
├── config.yaml               # Paths and pipeline hyperparameters
├── requirements.txt          # Python dependencies
├── src/
│   ├── config.py             # Configuration loader
│   ├── models.py             # VAE and DINOHead architectures
│   ├── embeddings.py         # VAE / DINOv2 embedding extractors
│   ├── detect.py             # YOLOv8 detection wrapper
│   ├── metrics.py            # IoU and RPC metrics (ACD, mCCD, mCIoU)
│   ├── pipeline.py           # Database indexing and evaluation engine
│   ├── run_pipeline.py       # Main CLI entry point
│   └── infer_demo.py         # Single-image visual inference demo
├── notebooks/                # Training pipelines (YOLOv8, VAE, DINOv2)
├── docs/
│   ├── Group16_Report.pdf    # Full technical project report
│   └── METRICS_NOTE.md       # Notes on evaluation methodology
└── outputs/                  # Saved FAISS indices and demo predictions

```

---

## 🛠️ Installation & Setup

**1. Clone the repository and install dependencies:**

```bash
git clone https://github.com/Nishant-ml/Automatic-Retail-Product-Checkout.git
cd Automatic-Retail-Product-Checkout
pip install -r requirements.txt

```

**2. Configure dataset and checkpoints:**
Update `config.yaml` with your local data and model weights paths:

* `dataset.root`: Path to RPC dataset (`val2019/`, `test2019/`, `instances_*.json`)
* `checkpoints.*`: Paths to `yolo_best.pt`, `vae_best.pth`, and `dino_student.pth`

---

## 🚀 Usage

**Build Database & Run Evaluation:**

```bash
# Evaluate DINOv2 retrieval pipeline
python src/run_pipeline.py --embedder dino --build-db --evaluate

# Evaluate VAE retrieval pipeline
python src/run_pipeline.py --embedder vae --build-db --evaluate

```

**Compute Full Detection Metrics:**

```bash
python src/run_pipeline.py --embedder dino --with-detection-metrics

```

**Run Single-Image Demo Inference:**

```bash
python src/infer_demo.py --image path/to/shelf.jpg --embedder dino
# Annotated output saved to: outputs/demo_result.jpg

```

---

## 👥 Contributors (Group 16 — IIT Kanpur)

* **Sarthak Dumbre** — `sarthakd25@iitk.ac.in`

* **Nishant** — `nishantk25@iitk.ac.in`

* **Vishal Junjare** — `vrjunjare25@iitk.ac.in`

* **Pranjal** — `pranjal25@iitk.ac.in`

* **Prafull Joshi** — `prafullj25@iitk.ac.in`

* **Esha Singh** — `esingh25@iitk.ac.in`
