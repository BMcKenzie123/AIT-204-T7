"""
train_gan.py — PyTorch Vanilla GAN on MNIST

Trains a dense GAN to generate 28×28 handwritten digits from 100-dim Gaussian
noise. Designed to be run end-to-end on a single GPU (e.g. Google Colab T4).

Outputs (written to ./gan_outputs/):
    - generator_epochXXX.pt       — generator weights at milestone epochs
    - discriminator_epochXXX.pt   — discriminator weights at milestone epochs
    - samples_epoch_XXX.png       — 10×10 sample grid at each milestone
    - samples_epoch_000_untrained.png  — untrained generator output
    - loss_curves.png             — D/G loss + D accuracy across training
    - disc_confidence.png         — post-training discriminator score histogram
    - history.csv                 — per-epoch metrics
    - eval_stats.json             — summary statistics for the report

Usage:
    python train_gan.py
    # or in a notebook cell:
    # %run train_gan.py
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from tqdm.auto import tqdm


# ======================================================================
# Configuration
# ======================================================================
SEED            = 42
NOISE_DIM       = 100
IMG_DIM         = 28 * 28                    # 784
BATCH_SIZE      = 128
EPOCHS          = 400
LR              = 2e-4                       # Adam learning rate
BETA_1, BETA_2  = 0.5, 0.999                 # Adam momentum parameters
SAMPLE_EPOCHS   = [1, 30, 100, 400]           # checkpoint milestones
OUT_DIR         = Path("./gan_outputs")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)


# ======================================================================
# Models
# ======================================================================
class Generator(nn.Module):
    """
    Maps z ∈ R^100 (Gaussian noise) to a 784-dim image vector in [-1, 1].

    Three progressively wider Dense blocks (256 → 512 → 1024), each with
    LeakyReLU(0.2) + BatchNorm1d. The final linear layer projects to 784
    and is squashed by tanh so outputs match the real-data preprocessing
    range of [-1, 1].
    """
    def __init__(self, noise_dim: int = NOISE_DIM, img_dim: int = IMG_DIM):
        super().__init__()
        # Keras BN momentum=0.8 ⇔ PyTorch BN momentum=0.2 (the convention
        # is flipped between the two frameworks).
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256, momentum=0.2),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512, momentum=0.2),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024, momentum=0.2),

            nn.Linear(1024, img_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """
    Maps a 784-dim image vector to a scalar in [0, 1] interpreted as
    P(image is real).

    Three Dense blocks (1024 → 512 → 256) with LeakyReLU(0.2).
    Batch-norm is deliberately omitted — it tends to memorise batch
    statistics in D and destabilise GAN training.
    """
    def __init__(self, img_dim: int = IMG_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================================
# Helpers
# ======================================================================
def save_sample_grid(generator: Generator, path: Path, epoch, fixed_noise: torch.Tensor):
    """
    Generate 100 samples from `fixed_noise` and save a 10×10 greyscale grid.
    Keeps `generator` in its original train/eval mode and restores it after.
    """
    was_training = generator.training
    generator.eval()
    with torch.no_grad():
        imgs = generator(fixed_noise).cpu().numpy()
    imgs = imgs.reshape(-1, 28, 28)
    imgs = (imgs + 1.0) / 2.0  # [-1, 1] -> [0, 1]

    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    fig.suptitle(f"Generator output — epoch {epoch}", fontsize=14)
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    if was_training:
        generator.train()


def load_mnist_tensor(device: torch.device) -> torch.Tensor:
    """
    Download MNIST and return the entire training set as a single tensor on
    `device`, shape (60000, 784), pixel values normalised to [-1, 1].

    Loading the whole dataset to GPU once and indexing into it per-batch is
    significantly faster than going through a DataLoader for a tiny dataset
    like MNIST, and avoids the multiprocessing-worker cleanup errors that
    `num_workers > 0` can trigger inside notebook kernels.
    """
    ds = datasets.MNIST(root="./data", train=True, download=True)
    # ds.data is a (60000, 28, 28) uint8 tensor
    X = ds.data.float().view(-1, 784)
    X = (X - 127.5) / 127.5                 # -> [-1, 1]
    return X.to(device)


# ======================================================================
# Main
# ======================================================================
def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"Device: {DEVICE}")
    print(f"Training {EPOCHS} epochs, batch size {BATCH_SIZE}")

    # Data — load once, keep on GPU, sample batches by random index.
    X = load_mnist_tensor(DEVICE)
    STEPS_PER_EPOCH = X.size(0) // BATCH_SIZE   # ~468 with full MNIST
    print(f"MNIST: {X.size(0):,} images loaded to {DEVICE}, "
          f"{STEPS_PER_EPOCH} steps/epoch")

    # Models ----------------------------------------------------------
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    print(f"Generator params    : {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}")

    # Separate optimizers — each updates only its own network's parameters.
    # This is the PyTorch way to freeze D during G updates: we simply call
    # d_opt.step() when we want D to learn, and g_opt.step() otherwise.
    # No need for Keras's fragile `trainable = False` flag.
    g_opt = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA_1, BETA_2))
    d_opt = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA_1, BETA_2))

    bce = nn.BCELoss()

    # Fixed noise so we can compare milestone grids apples-to-apples:
    # the same latent inputs evolve into different outputs as G learns.
    fixed_noise = torch.randn(100, NOISE_DIM, device=DEVICE)

    # Epoch-0 (untrained) snapshot
    save_sample_grid(G, OUT_DIR / "samples_epoch_000_untrained.png",
                     epoch="0 (untrained)", fixed_noise=fixed_noise)

    # Training --------------------------------------------------------
    history = {"epoch": [], "d_loss": [], "d_acc": [], "g_loss": []}
    t0 = time.time()

    real_labels_full = torch.full((BATCH_SIZE, 1), 1.0, device=DEVICE)
    fake_labels_full = torch.full((BATCH_SIZE, 1), 0.0, device=DEVICE)

    for epoch in range(1, EPOCHS + 1):
        G.train(); D.train()
        d_losses, d_accs, g_losses = [], [], []

        show_bar = (epoch in SAMPLE_EPOCHS) or (epoch % 25 == 0) or (epoch == 1)
        iterator = range(STEPS_PER_EPOCH)
        if show_bar:
            iterator = tqdm(iterator, desc=f"epoch {epoch:3d}", leave=False)

        for _ in iterator:
            # Sample real batch via random index into the GPU tensor
            idx = torch.randint(0, X.size(0), (BATCH_SIZE,), device=DEVICE)
            real_imgs = X[idx]

            # --------------------------------------------------------
            # (1) Train D on real + fake
            # --------------------------------------------------------
            d_opt.zero_grad()
            d_real = D(real_imgs)
            loss_d_real = bce(d_real, real_labels_full)

            noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=DEVICE)
            fake_imgs = G(noise)
            d_fake = D(fake_imgs.detach())      # detach = no gradient into G
            loss_d_fake = bce(d_fake, fake_labels_full)

            d_loss = loss_d_real + loss_d_fake
            d_loss.backward()
            d_opt.step()

            with torch.no_grad():
                acc_real = (d_real > 0.5).float().mean().item()
                acc_fake = (d_fake < 0.5).float().mean().item()
                d_acc = 0.5 * (acc_real + acc_fake)

            # --------------------------------------------------------
            # (2) Train G — try to make D say "real" on fake images
            # --------------------------------------------------------
            g_opt.zero_grad()
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=DEVICE)
            fake_imgs = G(noise)
            d_on_fake = D(fake_imgs)            # no detach — gradient flows into G
            g_loss = bce(d_on_fake, real_labels_full)
            g_loss.backward()
            g_opt.step()

            d_losses.append(d_loss.item() / 2)
            d_accs.append(d_acc)
            g_losses.append(g_loss.item())

        history["epoch"].append(epoch)
        history["d_loss"].append(float(np.mean(d_losses)))
        history["d_acc"].append(float(np.mean(d_accs)))
        history["g_loss"].append(float(np.mean(g_losses)))

        # Milestone snapshots
        if epoch in SAMPLE_EPOCHS:
            path = OUT_DIR / f"samples_epoch_{epoch:03d}.png"
            save_sample_grid(G, path, epoch=epoch, fixed_noise=fixed_noise)
            torch.save(G.state_dict(), OUT_DIR / f"generator_epoch{epoch:03d}.pt")
            torch.save(D.state_dict(), OUT_DIR / f"discriminator_epoch{epoch:03d}.pt")
            elapsed = (time.time() - t0) / 60.0
            print(
                f"[epoch {epoch:3d}] "
                f"d_loss={history['d_loss'][-1]:.4f}  "
                f"d_acc={history['d_acc'][-1]:.3f}  "
                f"g_loss={history['g_loss'][-1]:.4f}  "
                f"elapsed={elapsed:.1f} min"
            )
        elif epoch % 25 == 0:
            elapsed = (time.time() - t0) / 60.0
            print(
                f"  epoch {epoch:3d}  "
                f"d_loss={history['d_loss'][-1]:.4f}  "
                f"d_acc={history['d_acc'][-1]:.3f}  "
                f"g_loss={history['g_loss'][-1]:.4f}  "
                f"({elapsed:.1f} min)"
            )

    total = (time.time() - t0) / 60.0
    print(f"\nFinished in {total:.1f} min")

    # Final checkpoint used by the Streamlit app
    torch.save(G.state_dict(), OUT_DIR / "generator_final.pt")
    torch.save(D.state_dict(), OUT_DIR / "discriminator_final.pt")

    # ----------------------------------------------------------------
    # History CSV + loss curves plot
    # ----------------------------------------------------------------
    with open(OUT_DIR / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "d_loss", "d_acc", "g_loss"])
        for e, dl, da, gl in zip(history["epoch"], history["d_loss"],
                                 history["d_acc"], history["g_loss"]):
            w.writerow([e, dl, da, gl])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].plot(history["epoch"], history["d_loss"], color="#B85042",
                 linewidth=1.2, label="Discriminator loss")
    axes[0].plot(history["epoch"], history["g_loss"], color="#065A82",
                 linewidth=1.2, label="Generator loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("GAN training losses")
    axes[0].grid(alpha=0.3); axes[0].legend()

    axes[1].plot(history["epoch"], history["d_acc"], color="#2C5F2D",
                 linewidth=1.2, label="Discriminator accuracy")
    axes[1].axhline(0.5, color="grey", linestyle="--", alpha=0.7,
                    label="Ideal equilibrium (0.5)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Discriminator accuracy")
    axes[1].grid(alpha=0.3); axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_curves.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------------------------------------------
    # Quantitative evaluation: discriminator confidence on real vs fake
    # ----------------------------------------------------------------
    print("\nRunning post-training evaluation…")
    G.eval(); D.eval()
    N_EVAL = 5000

    with torch.no_grad():
        idx = torch.randperm(X.size(0), device=DEVICE)[:N_EVAL]
        real_sample = X[idx]
        noise_eval  = torch.randn(N_EVAL, NOISE_DIM, device=DEVICE)
        fake_sample = G(noise_eval)

        d_real = D(real_sample).cpu().numpy().flatten()
        d_fake = D(fake_sample).cpu().numpy().flatten()

    stats = {
        "device": str(DEVICE),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "total_minutes": total,
        "final_d_loss": history["d_loss"][-1],
        "final_g_loss": history["g_loss"][-1],
        "final_d_acc":  history["d_acc"][-1],
        "mean_disc_score_on_real":  float(d_real.mean()),
        "mean_disc_score_on_fake":  float(d_fake.mean()),
        "pct_fake_classified_real": float((d_fake > 0.5).mean()),
        "pct_real_classified_real": float((d_real > 0.5).mean()),
    }
    with open(OUT_DIR / "eval_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Histogram plot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(d_real, bins=40, alpha=0.7, color="#2C5F2D", label="Real MNIST")
    ax.hist(d_fake, bins=40, alpha=0.7, color="#B85042", label="Generator output")
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5,
               label="Decision threshold")
    ax.set_xlabel("Discriminator score  —  P(image is real)")
    ax.set_ylabel(f"Count (of {N_EVAL})")
    ax.set_title("Discriminator confidence on real vs. generated images")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "disc_confidence.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    print("Quantitative results:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nAll artifacts saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
