"""Unified embedder interface so pipeline.py can swap VAE <-> DINOv2
without duplicating the detect -> crop -> embed -> retrieve loop (the
original notebooks had this loop written out fresh, slightly differently,
in three separate places)."""
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from models import VAE, DINOHead


class BaseEmbedder:
    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


class VAEEmbedder(BaseEmbedder):
    def __init__(self, checkpoint_path, device, latent_dim=256, image_size=128):
        self.device = device
        self.model = VAE(latent_dim=latent_dim).to(device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.eval()
        self.transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
        self._dim = latent_dim

    @property
    def dim(self):
        return self._dim

    def embed(self, crop_bgr):
        img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, _ = self.model.encode(x)
        emb = mu.cpu().numpy()[0]
        return (emb / np.linalg.norm(emb)).astype("float32")


class DINOEmbedder(BaseEmbedder):
    def __init__(self, checkpoint_path, device, arch="dinov2_vits14", image_size=224):
        self.device = device
        self.student = torch.hub.load("facebookresearch/dinov2", arch).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.student.load_state_dict(checkpoint["student"])
        self.student.eval()
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._dim = 384

    @property
    def dim(self):
        return self._dim

    def embed(self, crop_bgr):
        img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.student(x).cpu().numpy()[0]
        return (emb / np.linalg.norm(emb)).astype("float32")


def build_embedder(name: str, cfg, device):
    emb_cfg = cfg["embedding"]
    if name == "vae":
        return VAEEmbedder(
            cfg["checkpoints"]["vae"], device,
            latent_dim=emb_cfg["vae_latent_dim"],
            image_size=emb_cfg["image_size_vae"],
        )
    if name == "dino":
        return DINOEmbedder(
            cfg["checkpoints"]["dino"], device,
            arch=emb_cfg["dino_arch"],
            image_size=emb_cfg["image_size_dino"],
        )
    raise ValueError(f"Unknown embedder '{name}', expected 'vae' or 'dino'")
