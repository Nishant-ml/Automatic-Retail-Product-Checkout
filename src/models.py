"""Model definitions, pulled out of the notebooks so they're defined once
instead of copy-pasted into both the VAE training notebook and the VAE
inference cell (they had drifted slightly out of sync between the two)."""
import torch
import torch.nn as nn


class VAE(nn.Module):
    """Conv encoder/decoder VAE used to embed product crops."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.001 * kl_loss


def contrastive_loss(embeddings, labels):
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    sim = embeddings @ embeddings.T
    labels = labels.unsqueeze(1)
    pos = (labels == labels.T).float()
    neg = (labels != labels.T).float()
    pos_loss = (1 - sim) * pos
    neg_loss = torch.nn.functional.relu(sim - 0.3) * neg
    return (pos_loss.sum() + neg_loss.sum()) / (pos.sum() + neg.sum())


class DINOHead(nn.Module):
    """Projection head on top of a DINOv2 backbone."""

    def __init__(self, in_dim: int = 384, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.GELU(), nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class DINOLoss(nn.Module):
    def __init__(self, t_temp: float = 0.04, s_temp: float = 0.1):
        super().__init__()
        self.t_temp = t_temp
        self.s_temp = s_temp

    def forward(self, student_out, teacher_out):
        loss, n = 0, 0
        for t in teacher_out:
            t = torch.softmax(t.detach() / self.t_temp, dim=-1)
            for s in student_out:
                s = torch.log_softmax(s / self.s_temp, dim=-1)
                loss += torch.mean(torch.sum(-t * s, dim=-1))
                n += 1
        return loss / n
