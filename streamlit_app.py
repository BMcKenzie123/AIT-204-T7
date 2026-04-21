"""
streamlit_app.py — GAN-Based Application: Generating Fake MNIST Images

Hosts an interactive demo of the trained GAN plus the full technical report.

Prereqs (requirements.txt):
    streamlit>=1.32
    torch>=2.2
    numpy
    matplotlib
    Pillow

Directory layout expected at deploy time:
    streamlit_app.py
    train_gan.py                   # imports Generator class
    requirements.txt
    gan_outputs/
        generator_final.pt         # checkpoint (required for live demo)
        samples_epoch_001.png      # optional; shown in Training tab
        samples_epoch_030.png
        samples_epoch_100.png
        samples_epoch_400.png
        loss_curves.png
        disc_confidence.png

If the checkpoint is missing, the Live Demo tab will display a friendly
message explaining how to produce one; the Report tab still works.

Run locally:
    streamlit run streamlit_app.py

Deploy:
    Push the repo to GitHub and point Streamlit Community Cloud at it.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

# The Generator class lives in train_gan.py — importing keeps a single
# source of truth for the architecture.
from train_gan import Generator, NOISE_DIM

# ======================================================================
# Page config — must be the very first Streamlit call
# ======================================================================
st.set_page_config(
    page_title="GAN — Fake MNIST Images",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUT_DIR = Path("gan_outputs")
CHECKPOINT = OUT_DIR / "generator_final.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# Model loading (cached so we don't reload on every interaction)
# ======================================================================
@st.cache_resource(show_spinner="Loading generator weights…")
def load_generator() -> Generator | None:
    """Return a Generator with weights loaded, or None if no checkpoint exists."""
    if not CHECKPOINT.exists():
        return None
    g = Generator().to(DEVICE)
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    g.load_state_dict(state)
    g.eval()
    return g


def generate_samples(
    generator: Generator, n: int, seed: int | None = None
) -> np.ndarray:
    """Run n noise vectors through G. Returns a (n, 28, 28) array in [0, 1]."""
    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(n, NOISE_DIM, device=DEVICE)
    with torch.no_grad():
        out = generator(z).cpu().numpy()
    out = out.reshape(-1, 28, 28)
    return (out + 1.0) / 2.0


def tile(images: np.ndarray, cols: int) -> np.ndarray:
    """Assemble (n, 28, 28) -> ((n/cols)*28, cols*28) single-channel tile."""
    n = images.shape[0]
    rows = int(np.ceil(n / cols))
    out = np.zeros((rows * 28, cols * 28))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        out[r*28:(r+1)*28, c*28:(c+1)*28] = im
    return out


def np_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a float array in [0, 1] to an 8-bit greyscale PIL image."""
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")


# ======================================================================
# Sidebar — global context
# ======================================================================
st.sidebar.title("GAN MNIST Demo")
st.sidebar.markdown(
    "This app pairs a live demo of a trained GAN with the full technical "
    "writeup for the project. Use the tabs on the main page to navigate."
)

generator = load_generator()
if generator is None:
    st.sidebar.error(
        "**No trained model loaded.**\n\n"
        f"Drop a PyTorch state-dict at `{CHECKPOINT}` (produced by "
        "`train_gan.py`) and rerun the app."
    )
else:
    st.sidebar.success(f"Generator loaded  •  device: `{DEVICE}`")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Author: **Brogan McKenzie and Adonijah Farner**  \nCourse: Deep Learning — Spring 2026"
)


# ======================================================================
# Main tabs
# ======================================================================
tab_demo, tab_report, tab_training, tab_code = st.tabs(
    ["🎨 Live Demo", "📄 Technical Report", "📈 Training", "💻 Code"]
)


# ----------------------------------------------------------------------
# Tab 1 — Live Demo
# ----------------------------------------------------------------------
with tab_demo:
    st.header("Live Demo")
    st.caption(
        "Sample fresh noise vectors and watch the trained generator "
        "synthesise handwritten digits. Re-roll as many times as you like."
    )

    if generator is None:
        st.warning(
            "Live generation requires a trained checkpoint. See sidebar."
        )
        st.info(
            "Meanwhile, here is what an untrained generator produces: "
            "pure noise — the whole point of training is to make these "
            "look like real digits."
        )
        rng = np.random.default_rng(0)
        mock = rng.random((25, 28, 28))
        st.image(np_to_pil(tile(mock, 5)).resize((280, 280), Image.NEAREST),
                 caption="Untrained output (placeholder)")
    else:
        col_controls, col_display = st.columns([1, 2])

        with col_controls:
            n_samples = st.slider(
                "Number of samples", min_value=4, max_value=100,
                value=25, step=1,
            )
            cols_per_row = st.slider(
                "Samples per row", min_value=2, max_value=10, value=5
            )
            seed = st.number_input(
                "Random seed (empty for fresh)", value=0, step=1,
                help="Set a seed to reproduce the same sample grid. "
                     "Use the re-roll button to change it on demand.",
            )
            if st.button("🎲 Re-roll", type="primary", use_container_width=True):
                seed = int(np.random.randint(0, 1_000_000))
                st.session_state["last_seed"] = seed
            seed = st.session_state.get("last_seed", seed)
            st.caption(f"Current seed: `{seed}`")

        with col_display:
            with st.spinner("Generating…"):
                imgs = generate_samples(generator, n_samples, seed=seed)
                grid = tile(imgs, cols_per_row)
                # Upscale for crisp display (28×28 is tiny on modern screens)
                scale = max(8, min(16, 400 // grid.shape[1]))
                pil = np_to_pil(grid).resize(
                    (grid.shape[1] * scale, grid.shape[0] * scale),
                    Image.NEAREST,
                )
                st.image(pil, caption=f"{n_samples} samples · seed {seed}")

            buf = io.BytesIO()
            np_to_pil(grid).save(buf, format="PNG")
            st.download_button(
                "⬇  Download grid (PNG)",
                data=buf.getvalue(),
                file_name=f"gan_samples_seed{seed}.png",
                mime="image/png",
            )

    st.markdown("---")
    st.subheader("What are you looking at?")
    st.markdown(
        "Each sample starts life as a 100-dimensional vector drawn from a "
        "Gaussian. The trained generator maps that vector to a 784-dim "
        "tensor, which is reshaped to 28 × 28 pixels and displayed above. "
        "Different seeds produce different noise vectors, which produce "
        "different digits."
    )


# ----------------------------------------------------------------------
# Tab 2 — Technical Report
# ----------------------------------------------------------------------
with tab_report:
    st.header("Technical Report")

    st.markdown(
        """
### 1. Problem Statement

Synthetic images have become ubiquitous on social media, and their growing
realism actively shapes public discourse — doctored photos, AI-generated
"evidence" of political events, deep-fake endorsements. Understanding **how**
such images are produced is a prerequisite to being able to detect them and
to reason critically about the media we consume.

This project implements a classical **Generative Adversarial Network**
(Goodfellow et al., 2014) from scratch in PyTorch and trains it on the MNIST
handwritten-digit dataset. MNIST is deliberately chosen as a toy domain:

- Small enough to train on a single consumer GPU in minutes.
- Has an obvious "looks right / looks wrong" sanity check.
- The lessons transfer directly to modern systems like StyleGAN or
  diffusion-based deep-fake pipelines.

**Goal.** Build a generator $G$ that converts 100-dim Gaussian noise into
28 × 28 digit images, and a discriminator $D$ that distinguishes real MNIST
from $G$'s output. Train them against each other so $G$'s output
progressively becomes harder and harder for $D$ to reject.

---

### 2. Algorithm of the Solution

**Generator $G$.** Maps $\\mathbf{z} \\sim \\mathcal{N}(0, I)$, $\\mathbf{z}
\\in \\mathbb{R}^{100}$, to a 784-dim image vector. Three Dense blocks
(256 → 512 → 1024) each followed by `LeakyReLU(0.2)` and `BatchNorm1d`.
The output uses `tanh`, matching the $[-1, 1]$ range of the preprocessed
real images.

**Discriminator $D$.** Takes a 784-dim image, outputs a scalar in $[0, 1]$
interpreted as $P(\\text{real})$. Three Dense blocks (1024 → 512 → 256) with
`LeakyReLU(0.2)`, then sigmoid. Batch-norm is deliberately omitted — BN in
$D$ tends to memorise batch statistics and destabilise training.

**Training.** Two independent Adam optimisers, one per network. For each
batch:

1. Sample real MNIST images; label them `1`.
2. Generate fake images from fresh noise; label them `0`.
3. One gradient step on `d_opt` using real + fake loss.
4. Generate fresh fakes; label them `1` (trick: claim they're real); one
   gradient step on `g_opt` only — D's weights are frozen implicitly
   because its parameters aren't in `g_opt`.

Because each optimiser only owns its own network's parameters, freezing
one network during the other's update requires no special flags — this is
why PyTorch is less bug-prone for GAN training than frameworks that rely
on a `.trainable = False` switch.

### Objective

$$\\min_G \\max_D \\; \\mathbb{E}_{x \\sim p_\\text{data}}\\!\\bigl[\\log D(x)\\bigr]
\\,+\\, \\mathbb{E}_{z \\sim p_z}\\!\\bigl[\\log(1 - D(G(z)))\\bigr]$$

In practice we optimise the non-saturating variant for $G$ (maximise
$\\log D(G(z))$), which gives stronger gradients early in training.

---

### 3. Analysis of Findings

**Qualitative progression.** At epoch 1 the generator produces pure
high-frequency noise. By epoch 30 blob-like dark regions on a lighter
background resemble the global "ink in the middle of a 28 × 28 canvas"
structure of MNIST. By epoch 100 many samples are plausibly recognisable
as digits (0s, 1s, and 7s tend to appear first — simple strokes).
By epoch 400 the majority of samples are clearly legible handwritten
digits.

**Quantitative signal.** The discriminator-confidence histogram is the
cleanest diagnostic. If training worked, the distribution of $D$'s score
on generator output shifts from being concentrated near 0 (at the start
everything is obviously fake) to overlapping substantially with the
real-data distribution. The fraction of fakes misclassified as real is a
direct measure of how well G is fooling D. Loss curves that oscillate
around a steady mean, rather than one side collapsing to zero, indicate
the adversarial game is in healthy equilibrium.

**Limitations.**

1. Dense-only GAN — no convolutions. DCGAN-style architectures produce
   sharper images because they exploit spatial structure.
2. MNIST is single-channel, low-resolution, tightly centred. Scaling to
   natural images needs conv layers, much more data and compute.
3. Mode-collapse risk: the generator may ignore some digit classes. A
   quick visual audit of the final sample grid is essential.

**So — does this GAN produce convincing fakes?** At 28 × 28 resolution,
after 400 epochs, yes. Casual human inspection would not reliably
distinguish the best generated digits from real MNIST samples. The same
adversarial principle, scaled up with convolutions and orders of
magnitude more compute, is what powers modern deep-fake tools.

---

### 4. References

1. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D.,
   Ozair, S., Courville, A., & Bengio, Y. (2014). *Generative Adversarial
   Nets.* NeurIPS 27.
2. Radford, A., Metz, L., & Chintala, S. (2016). *Unsupervised Representation
   Learning with Deep Convolutional Generative Adversarial Networks.* ICLR.
3. LeCun, Y., Cortes, C., & Burges, C. (1998). *The MNIST database of
   handwritten digits.*
4. Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-Performance
   Deep Learning Library.* NeurIPS.
5. Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic
   Optimization.* ICLR.
6. Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep
   Network Training by Reducing Internal Covariate Shift.* ICML.
7. Maas, A. L., Hannun, A. Y., & Ng, A. Y. (2013). *Rectifier Nonlinearities
   Improve Neural Network Acoustic Models.* ICML.
"""
    )


# ----------------------------------------------------------------------
# Tab 3 — Training visualisations
# ----------------------------------------------------------------------
with tab_training:
    st.header("Training dynamics")
    st.caption(
        "Artefacts produced by `train_gan.py`. If you re-run training, "
        "simply refresh this app to see the new plots."
    )

    # Milestone grids
    st.subheader("Generator output at training milestones")
    st.caption("Same 100 noise vectors, re-evaluated at each checkpoint.")
    cols = st.columns(4)
    milestone_files = [
        (1,   "samples_epoch_001.png"),
        (30,  "samples_epoch_030.png"),
        (100, "samples_epoch_100.png"),
        (400, "samples_epoch_400.png"),
    ]
    for col, (ep, fname) in zip(cols, milestone_files):
        with col:
            path = OUT_DIR / fname
            if path.exists():
                st.image(str(path), caption=f"Epoch {ep}", use_container_width=True)
            else:
                st.info(f"Run training to produce `{fname}`.")

    st.markdown("---")

    # Loss curves
    st.subheader("Loss & accuracy curves")
    lc = OUT_DIR / "loss_curves.png"
    if lc.exists():
        st.image(str(lc), use_container_width=True)
    else:
        st.info("Loss curves will appear here once training completes.")

    # Discriminator confidence
    st.subheader("Discriminator confidence: real vs. fake")
    dc = OUT_DIR / "disc_confidence.png"
    if dc.exists():
        st.image(str(dc), use_container_width=True)
        st.markdown(
            "The more the orange (fake) and green (real) histograms overlap, "
            "the harder it is for $D$ to tell them apart — and the better the "
            "generator has learned to imitate the data distribution."
        )
    else:
        st.info("Confidence histogram will appear after evaluation.")

    # Eval stats
    stats_path = OUT_DIR / "eval_stats.json"
    if stats_path.exists():
        import json
        stats = json.loads(stats_path.read_text())
        st.subheader("Final metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean D-score on fakes",
                  f"{stats['mean_disc_score_on_fake']:.3f}",
                  help="Closer to 0.5 = G fools D more effectively.")
        m2.metric("% fakes misclassified as real",
                  f"{stats['pct_fake_classified_real']*100:.1f}%")
        m3.metric("Training time",
                  f"{stats['total_minutes']:.1f} min",
                  help=f"on {stats.get('device', 'unknown device')}")


# ----------------------------------------------------------------------
# Tab 4 — Code
# ----------------------------------------------------------------------
with tab_code:
    st.header("Source code")
    st.caption(
        "The full training script is shown below. It is the single source "
        "of truth for the model architecture; this app imports the "
        "`Generator` class directly from it."
    )
    script = Path("train_gan.py")
    if script.exists():
        st.code(script.read_text(), language="python")
    else:
        st.warning("`train_gan.py` not found next to `streamlit_app.py`.")
