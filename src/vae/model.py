# vae.py
import torch
import torch.nn.functional as F
from torch import nn

class VAE(nn.Module):
    def __init__(self, seq_len, nb_state, emb=5, latent_dim=16, hidden=128):
        super().__init__()
        self.seq_len, self.nb_state, self.emb = seq_len, nb_state, emb
        self.embeddings = nn.Parameter(torch.randn(seq_len, nb_state, emb) * 0.02)
        in_dim, out_dim = seq_len*emb, seq_len*nb_state
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out_dim))

    def _embed(self, x):
        B, L = x.shape
        E = self.embeddings.view(self.seq_len*self.nb_state, self.emb)
        offsets = (torch.arange(self.seq_len, device=x.device)*self.nb_state)[None, :]
        idx = (x + offsets).view(B*L)
        return E[idx].view(B, L*self.emb)

    def encode(self, x_idx):
        h = self.enc(self._embed(x_idx))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5*logvar)

    def decode(self, z):
        return self.dec(z).view(-1, self.seq_len, self.nb_state)

    def forward(self, input_seq):
        mu, logvar = self.encode(input_seq)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return {"logits": logits, "z": z, "mu": mu, "logvar": logvar}

    @staticmethod
    def kl_normal(mu, logvar):
        return 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1.0 - logvar)

    @staticmethod
    def kl_per_sample(mu, logvar):
        # per-sample KL divergence to N(0,I)
        return 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar).mean(dim=1)

    def loss(self, input_seq, out, stability_weights=None, beta=1.0, lambda_s=1.0, reduction="mean"):
        B, L = input_seq.shape
        logits = out["logits"].view(B*L, self.nb_state)
        targets = input_seq.view(B*L)
        ce = F.cross_entropy(logits, targets, reduction=reduction)

        kl_i = self.kl_per_sample(out["mu"], out["logvar"])  # [B]
        if stability_weights is None:
            kl = beta * kl_i.mean()
        else:
            w = stability_weights.view(-1)
            kl = ((beta + lambda_s * w) * kl_i).mean()

        total = ce + kl
        return {"total": total, "ce": ce, "kl": kl, "kl_mean_unweighted": kl_i.mean()}
