# KLA SemiCon AI Hackathon — Image Restoration: Complete Technical Guide

> **Problem Statement:** AI-Based Restoration of Degraded Semiconductor Inspection Images  
> **Core Task:** Given paired (degraded → clean) training images, train a model that removes speckle noise, Gaussian noise, and upscales low-resolution images — simultaneously — and generalises to unseen chip structures.

---

## Table of Contents

1. [Problem Statement — Technical Breakdown](#1-problem-statement--technical-breakdown)
2. [Input / Output Specification](#2-input--output-specification)
3. [Degradation Types — Deep Dive](#3-degradation-types--deep-dive)
4. [Architecture Candidates & Reasoning](#4-architecture-candidates--reasoning)
5. [Recommended Architecture: NAFNet + SPADE-SR Hybrid](#5-recommended-architecture-nafnet--spade-sr-hybrid)
6. [Loss Function Design](#6-loss-function-design)
7. [Data Pipeline for Large Datasets](#7-data-pipeline-for-large-datasets)
8. [Training Strategy — Session-Resumable](#8-training-strategy--session-resumable)
9. [Hyperparameter Tuning](#9-hyperparameter-tuning)
10. [Logging & Experiment Tracking](#10-logging--experiment-tracking)
11. [Compute Requirements](#11-compute-requirements)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Inference Script (Submission-Ready)](#13-inference-script-submission-ready)
14. [GitHub Repository Structure](#14-github-repository-structure)
15. [Learning Resources](#15-learning-resources)

---

## 1. Problem Statement — Technical Breakdown

### What the model must do

You are training a **blind image restoration** model — "blind" because at inference time you do not know which degradation(s) are present, at what severity, or at which scale factor. The model must:

1. **Denoise** — remove multiplicative speckle noise and additive Gaussian noise without blurring real edge detail.
2. **Super-resolve** — upscale from `256×256 → 512×512` or `128×128 → 256×256` (i.e., ×2 SR in both cases).
3. **Handle simultaneity** — a single image may carry both noise types AND be downsampled.
4. **Generalise OOD** — the test set contains semiconductor structures never seen during training.

### Why this is non-trivial

| Challenge | Why it matters |
|---|---|
| Pixel values exceed [0,1] in degraded images | Standard sigmoid normalisation will clip real signal; the model must handle super-gaussian intensity distributions |
| Speckle is *multiplicative* | Most denoisers assume additive noise; speckle requires log-domain or specialised handling |
| Super-resolution ×2 on top of noise | Naive bicubic upscaling amplifies noise; joint denoising+SR is a coupled problem |
| OOD generalisation | Architectural inductive biases (e.g. local vs. global attention) determine how well you generalise |
| Speed constraint | Must be fast enough to be practical; large transformer stacks are disqualifying |

---

## 2. Input / Output Specification

```
INPUT
─────────────────────────────────────────────
Type        : Grayscale (1-channel) TIFF/PNG
Resolution  : 256×256 OR 128×128 pixels
Value range : May EXCEED [0, 1] due to speckle
              (e.g. values in [-0.3, 1.4] are valid)
Degradation : Speckle noise + Gaussian noise
              + 2× downsampling (any combination)

OUTPUT
─────────────────────────────────────────────
Type        : Grayscale (1-channel) TIFF/PNG
Resolution  : 512×512 OR 256×256 pixels (always 2× input)
Value range : Strictly [0, 1] (clamp after inference)
Target      : Must match ground truth SSIM/PSNR/LPIPS
```

### Data preprocessing contract

```python
# Do NOT clip input to [0,1] before the model sees it.
# The out-of-range values carry signal about speckle severity.
# Only the OUTPUT should be clamped.

degraded = load_image(path)          # raw float32, may be in [-ε, 1+ε]
restored = model(degraded)
restored = torch.clamp(restored, 0.0, 1.0)   # clamp only at output
```

---

## 3. Degradation Types — Deep Dive

### 3.1 Speckle Noise

**Mathematical model:**

```
I_observed = I_true × η + ε
where η ~ Gamma(k, θ),  ε ~ N(0, σ²)
```

Speckle is *multiplicative* — bright pixels get noisier proportionally. This is the dominant noise in coherent imaging systems (SEM, SAR). Key properties:
- Signal-dependent variance (heteroscedastic)
- Pixel values pushed outside [0,1]
- Log-transform converts it to near-additive: `log(I_obs) ≈ log(I_true) + log(η)`

**Implication for architecture:** The network must learn to handle heteroscedastic noise. Architectures with adaptive normalisation (like SPADE) or channel attention (like SENet blocks) handle this better than plain CNNs.

### 3.2 Gaussian Noise

**Mathematical model:**

```
I_observed = I_true + ε,   ε ~ N(0, σ²)
```

Additive, signal-independent. Softens edges. The classic BM3D / DnCNN target. Gaussian noise in [0,1]-normalised images is well-understood; the hard part here is *simultaneous* handling with speckle.

### 3.3 Spatial Resolution Reduction (×2 Super-Resolution)

**Mathematical model:**

```
I_LR = (I_HR ⊛ k) ↓s
where k = blur kernel, ↓s = s-fold subsampling
```

The task is ×2 SR (256→512 or 128→256). This is classical SISR (Single Image Super-Resolution). Key challenges:
- Ill-posed inverse problem (many HR images map to same LR)
- Fine semiconductor features (lines ~1-3 pixels wide) must be reconstructed
- Noise amplification during upsampling must be avoided

---

## 4. Architecture Candidates & Reasoning

Below are four viable architectures, analysed for this specific problem.

### 4.1 U-Net (Baseline)

**How it works:** Encoder-decoder with skip connections. Encoder extracts multi-scale features; decoder reconstructs at full resolution.

```
Input (noisy LR) → [Conv↓] × 4 → Bottleneck → [ConvTranspose↑] × 4 → Output (clean HR)
                         ↕ skip connections ↕
```

**Pros:** Fast, well-understood, easy to train  
**Cons:** No global context (pure local convolution), doesn't handle OOD well, no explicit super-resolution pathway  
**Verdict:** Good baseline; NOT optimal. Use for sanity checks.

---

### 4.2 RRDB-Net (Enhanced SRGAN Generator) — ESRGAN Family

**How it works:** Stacked Residual-in-Residual Dense Blocks (RRDB). Each block has dense connections within, and residual scaling prevents training collapse.

```
LR → Conv → [RRDB × N] → Conv → PixelShuffle(×2) → Output
```

**Why it's relevant:**
- PixelShuffle (sub-pixel convolution) is the state-of-art SR upsampling operator
- RRDB blocks learn rich multi-scale features
- Designed explicitly for ×2/×4 SR

**Pros:** Excellent SR quality, fast inference, well-documented  
**Cons:** Designed for natural images, may not handle the speckle heteroscedasticity well without modification  
**Verdict:** Strong SR backbone. Best used as the upsampling module inside a larger pipeline.

---

### 4.3 NAFNet (Nonlinear Activation Free Network)

**Paper:** *Simple Baselines for Image Restoration* (Chen et al., ECCV 2022)

**How it works:** Replaces all nonlinearities with *SimpleGate* (elementwise product of two feature groups — equivalent to a learned gating). Uses *Simplified Channel Attention* instead of full SE blocks. No GELU, no ReLU — pure linear + gate operations.

```
Input → [NAFBlock × N] → Output
NAFBlock = LayerNorm → Conv1×1 → SimpleGate → Conv3×3 → SCA → residual
```

**Why it wins for this problem:**
- State-of-art on SIDD, DND (denoising) and GoPro (deblurring) benchmarks
- 10-100× faster than transformer-based methods (Restormer, SwinIR)
- SimpleGate is smooth — critical for heteroscedastic noise (no hard-cutoff like ReLU)
- Works without BatchNorm (which breaks for small batch sizes on huge datasets)

**Pros:** Speed + accuracy Pareto-optimal, simple to train  
**Cons:** Purely local receptive field unless width is large  
**Verdict:** ✅ Best denoising backbone.

---

### 4.4 SwinIR / Restormer (Transformer-based)

**How they work:** Replace CNN blocks with window-based self-attention (SwinIR) or axial attention (Restormer). Global context at the cost of O(N²) or O(N log N) attention.

**Why NOT recommended as primary:**
- Inference time on 512×512: ~2-8 seconds on A100; slower on smaller GPUs
- With a test set that may be large, this violates the speed constraint
- SwinIR has 11M+ parameters — hard to deploy efficiently
- Out-of-distribution performance is not dramatically better than NAFNet with augmentation

**When to use:** If you have an H100 and need marginal SSIM gains, Restormer is a valid experiment. Otherwise, stick with NAFNet.

**Verdict:** ❌ Too slow for real deployment; ⚠️ viable if compute budget allows.

---

## 5. Recommended Architecture: NAFNet + RRDB Hybrid

**Design Philosophy:** Use NAFNet for joint denoising + feature extraction, then use a lightweight RRDB + PixelShuffle head for super-resolution upsampling. This separates concerns while keeping the pipeline end-to-end trainable.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT (noisy LR)                         │
│                   [B, 1, H/2, W/2],  float ∈ [-ε, 1+ε]        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Stem Conv 3×3  │  → [B, C, H/2, W/2]
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │     NAFNet Encoder           │
              │  [NAFBlock×4] C channels     │  → scale 1
              │  [NAFBlock×4] 2C channels    │  → scale 1/2 (stride)
              │  [NAFBlock×4] 4C channels    │  → scale 1/4 (stride)
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │     Bottleneck              │
              │  [NAFBlock×4] 4C channels   │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │     NAFNet Decoder           │
              │  [NAFBlock×4] 4C → 2C        │  + skip conn.
              │  [NAFBlock×4] 2C → C         │  + skip conn.
              │  [NAFBlock×4] C → C          │  + skip conn.
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │     SR Head                 │
              │  RRDB ×4 (residual dense)   │
              │  Conv 3×3                   │
              │  PixelShuffle(r=2)          │  ×2 upscale
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  Output Conv 1×1            │
              │  → [B, 1, H, W]            │
              │  clamp(0, 1) at inference   │
              └─────────────────────────────┘

C = 32 or 64 depending on GPU VRAM
```

### Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Upsampling method | PixelShuffle (sub-pixel conv) | No checkerboard artifacts; learnable; fast |
| Normalisation | LayerNorm per channel | Works with batch size 1; no BatchNorm statistics drift |
| Activation | SimpleGate (NAFNet) | Smooth, handles heteroscedastic data; no dead neurons |
| Skip connections | U-Net style | Preserves fine-detail gradients; critical for edge recovery |
| Channel count C | 32 (fast) or 64 (quality) | 32 for prototyping; 64 for final submission |

### PyTorch Implementation Skeleton

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.Unflatten(1, (channels, 1, 1))
        )
    def forward(self, x):
        return x * self.attn(x)

class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_c = c * dw_expand
        ffn_c = c * ffn_expand
        self.norm1 = nn.LayerNorm(c)
        self.conv1 = nn.Conv2d(c, dw_c, 1)
        self.conv2 = nn.Conv2d(dw_c // 2, dw_c // 2, 3, padding=1, groups=dw_c // 2)
        self.sg = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_c // 2)
        self.conv3 = nn.Conv2d(dw_c // 2, c, 1)
        self.norm2 = nn.LayerNorm(c)
        self.conv4 = nn.Conv2d(c, ffn_c, 1)
        self.conv5 = nn.Conv2d(ffn_c // 2, c, 1)
        self.beta = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-2)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-2)

    def forward(self, x):
        B, C, H, W = x.shape
        # Spatial mixing
        inp = x
        x = self.norm1(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.conv1(x)
        x = self.sg(x)
        x = self.conv2(x)
        x = self.sca(x)
        x = self.conv3(x)
        x = inp + x * self.beta
        # Channel mixing
        inp = x
        x = self.norm2(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return inp + x * self.gamma


class NAFNetSR(nn.Module):
    """Joint denoising + ×2 super-resolution model."""
    def __init__(self, in_ch=1, width=64, enc_blocks=[2,2,4,8], dec_blocks=[2,2,2,2]):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, width, 3, padding=1)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2))
            ch *= 2

        # Bottleneck
        self.bottleneck = nn.Sequential(*[NAFBlock(ch) for _ in range(4)])

        # Decoder
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 1),
                nn.PixelShuffle(2)
            ))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch * 2) for _ in range(n)]))

        # SR head with PixelShuffle ×2
        self.sr_head = nn.Sequential(
            nn.Conv2d(ch * 2, width, 3, padding=1),
            nn.Conv2d(width, in_ch * 4, 3, padding=1),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        return self.sr_head(x)
```

---

## 6. Loss Function Design

Using a single loss (e.g., L1) leads to blurry results. The optimal strategy combines multiple complementary losses:

### 6.1 Reconstruction Loss (L1 + L2 mix)

```python
# Charbonnier loss — smooth L1, more robust to outliers than MSE
def charbonnier_loss(pred, target, eps=1e-3):
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()
```

**Why Charbonnier over L2:** L2 (MSE) is dominated by large errors and produces over-smoothed outputs. Charbonnier (smooth L1) treats large and small errors more equally, preserving high-frequency detail — critical for semiconductor edge features.

### 6.2 Perceptual Loss (VGG Feature Matching)

```python
from torchvision.models import vgg16
import torch.nn as nn

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        # Adapt for single-channel: repeat grayscale → RGB
        self.vgg = vgg
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        # Images are grayscale; VGG expects 3-channel
        pred_rgb = pred.repeat(1, 3, 1, 1)
        tgt_rgb = target.repeat(1, 3, 1, 1)
        return self.loss(self.vgg(pred_rgb), self.vgg(tgt_rgb))
```

**Why perceptual loss:** Perceptual loss matches high-level feature distributions, not pixel values. It rewards structural similarity over exact pixel matches — critical for OOD generalisation and avoiding mode-averaging blur.

### 6.3 Frequency Loss (FFT-based)

```python
def frequency_loss(pred, target):
    pred_fft = torch.fft.rfft2(pred, norm='ortho')
    tgt_fft = torch.fft.rfft2(target, norm='ortho')
    return F.l1_loss(torch.abs(pred_fft), torch.abs(tgt_fft))
```

**Why frequency loss:** Semiconductor images have structured high-frequency content (sharp edges, periodic gratings). L1/L2 in pixel space over-smooths these; matching in frequency space directly penalises missing high-frequency detail.

### 6.4 SSIM Loss

```python
# Use pytorch-msssim library
from pytorch_msssim import ssim

def ssim_loss(pred, target):
    return 1.0 - ssim(pred, target, data_range=1.0, size_average=True)
```

**Why SSIM loss:** SSIM correlates with human perception of sharpness and structure. Using it as a training signal directly optimises for the primary leaderboard metric.

### 6.5 Combined Loss (Final)

```python
class CombinedLoss(nn.Module):
    def __init__(self, λ_perc=0.1, λ_freq=0.05, λ_ssim=0.2):
        super().__init__()
        self.perc = PerceptualLoss()
        self.λ_perc = λ_perc
        self.λ_freq = λ_freq
        self.λ_ssim = λ_ssim

    def forward(self, pred, target):
        l_char  = charbonnier_loss(pred, target)
        l_perc  = self.perc(pred, target)
        l_freq  = frequency_loss(pred, target)
        l_ssim  = ssim_loss(pred, target)

        total = l_char + self.λ_perc * l_perc + self.λ_freq * l_freq + self.λ_ssim * l_ssim
        return total, {
            'charbonnier': l_char.item(),
            'perceptual': l_perc.item(),
            'frequency': l_freq.item(),
            'ssim': l_ssim.item(),
        }
```

**Weight rationale:**

| Loss term | Weight | Reason |
|---|---|---|
| Charbonnier | 1.0 | Primary pixel fidelity; anchor term |
| Perceptual | 0.1 | Structural similarity; too high → hallucination |
| Frequency | 0.05 | High-freq recovery; too high → ringing artifacts |
| SSIM | 0.2 | Direct metric optimisation |

---

## 7. Data Pipeline for Large Datasets

With a large dataset of high-resolution images, naive loading will bottleneck training. Here is a production-grade pipeline.

### 7.1 Dataset Class with Patch-Based Training

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SemiconDataset(Dataset):
    """
    Paired (degraded LR, clean HR) dataset with on-the-fly patch extraction.
    Supports HUGE datasets via lazy loading (no preloading to RAM).
    """
    def __init__(self, root_dir, patch_size=128, split='train', patches_per_image=16):
        self.root = Path(root_dir)
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.lr_dir = self.root / 'degraded'
        self.hr_dir = self.root / 'clean'

        self.lr_files = sorted(self.lr_dir.glob('*.png')) + \
                        sorted(self.lr_dir.glob('*.tiff'))
        assert len(self.lr_files) > 0, "No images found in degraded directory"

        self.augment = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ], additional_targets={'hr': 'image'})

    def __len__(self):
        return len(self.lr_files) * self.patches_per_image

    def __getitem__(self, idx):
        file_idx = idx // self.patches_per_image
        lr_path = self.lr_files[file_idx]
        hr_path = self.hr_dir / lr_path.name

        # Load as float32; do NOT clip — preserve out-of-range speckle values
        lr = cv2.imread(str(lr_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        hr = cv2.imread(str(hr_path), cv2.IMREAD_UNCHANGED).astype(np.float32)

        # Normalise HR to [0,1] if not already (GT is always [0,1] per spec)
        if hr.max() > 1.0:
            hr = hr / 65535.0  # 16-bit TIFF
        # LR: do NOT normalise — preserve speckle out-of-range signal

        # Patch extraction: HR patch_size × lr patch_size (2× ratio)
        H, W = lr.shape[:2]
        ps = self.patch_size
        x = np.random.randint(0, W - ps + 1)
        y = np.random.randint(0, H - ps + 1)

        lr_patch = lr[y:y+ps, x:x+ps]
        hr_patch = hr[y*2:(y+ps)*2, x*2:(x+ps)*2]

        # Augmentation (same transform for both)
        augmented = self.augment(image=lr_patch, hr=hr_patch)
        lr_patch = augmented['image']
        hr_patch = augmented['hr']

        # Add channel dim [H, W] → [1, H, W]
        lr_t = torch.from_numpy(lr_patch).unsqueeze(0)
        hr_t = torch.from_numpy(hr_patch).unsqueeze(0)

        return lr_t, hr_t


def get_dataloader(root_dir, batch_size=8, num_workers=8, patch_size=128):
    dataset = SemiconDataset(root_dir, patch_size=patch_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,          # faster GPU transfer
        persistent_workers=True,  # avoid worker respawn overhead
        prefetch_factor=4,        # pipeline loading ahead of training
    )
```

### 7.2 Data Augmentation Strategy

| Augmentation | Rationale |
|---|---|
| Random 90° rotation | Semiconductor images have 4-fold symmetry |
| Horizontal/vertical flip | Same as above; OOD robustness |
| Random crop (patch training) | Increases effective dataset size × 16; reduces memory |
| MixUp (light) | Improves generalisation; blends feature distributions |
| NO: brightness/contrast jitter | GT is precisely calibrated; perturbing would corrupt training signal |
| NO: elastic deform | Physical geometry must be preserved |

**Critical note on speckle:** Do NOT add synthetic speckle augmentation on top of existing data — the dataset already contains real speckle. Adding more would change the noise distribution and confuse the model.

---

## 8. Training Strategy — Session-Resumable

This is the most important section for the hackathon. Your training must survive session interruptions.

### 8.1 Session-Resumable Training Loop

```python
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import wandb
import json
from pathlib import Path

def train(config):
    # ── Setup ──────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NAFNetSR(in_ch=1, width=config['width']).to(device)
    criterion = CombinedLoss().to(device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=config['lr'],
                            betas=(0.9, 0.999),
                            weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6
    )
    scaler = GradScaler()  # mixed precision (FP16/BF16 on Ampere+)

    train_loader = get_dataloader(config['data_dir'], config['batch_size'])

    # ── Resume from checkpoint ─────────────────────────────────────────
    ckpt_dir = Path(config['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_ssim = 0.0

    latest_ckpt = ckpt_dir / 'latest.pth'
    if latest_ckpt.exists():
        print(f"[Resume] Loading checkpoint from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_ssim = ckpt.get('best_ssim', 0.0)
        print(f"[Resume] Continuing from epoch {start_epoch}, best SSIM={best_ssim:.4f}")

    # ── Logging ────────────────────────────────────────────────────────
    wandb.init(
        project='kla-semicon-restoration',
        config=config,
        resume='allow',  # resumes the same run if the run_id matches
        id=config.get('wandb_run_id'),
    )

    # ── Training Loop ──────────────────────────────────────────────────
    for epoch in range(start_epoch, config['epochs']):
        model.train()
        epoch_losses = {}

        for step, (lr_img, hr_img) in enumerate(train_loader):
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()

            with autocast(dtype=torch.bfloat16):    # BF16 on H100; FP16 on V100
                pred = model(lr_img)
                loss, loss_dict = criterion(pred, hr_img)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # Accumulate loss components
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v

            if step % config['log_every'] == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': scheduler.get_last_lr()[0],
                    'step': epoch * len(train_loader) + step,
                    **{f'train/{k}': v for k, v in loss_dict.items()}
                })

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────
        if epoch % config['val_every'] == 0:
            val_ssim, val_psnr = validate(model, config, device)
            wandb.log({'val/ssim': val_ssim, 'val/psnr': val_psnr, 'epoch': epoch})

            if val_ssim > best_ssim:
                best_ssim = val_ssim
                torch.save(model.state_dict(), ckpt_dir / 'best_model.pth')
                print(f"[Epoch {epoch}] New best SSIM: {best_ssim:.4f} — model saved")

        # ── Save checkpoint (always) ───────────────────────────────────
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_ssim': best_ssim,
        }, latest_ckpt)

    wandb.finish()
    return model
```

### 8.2 Mixed Precision Training

| GPU | Recommended dtype | Why |
|---|---|---|
| NVIDIA H100 | `torch.bfloat16` | Native BF16 tensor cores; no overflow risk (wider exponent than FP16) |
| NVIDIA A100 | `torch.bfloat16` | Same as H100 |
| NVIDIA V100 | `torch.float16` | BF16 not natively supported |
| NVIDIA RTX 30xx/40xx | `torch.bfloat16` | Ampere+ architecture |

Mixed precision gives **~2× throughput** and **~50% memory reduction** with negligible accuracy loss.

### 8.3 Gradient Accumulation (for effective large batch)

```python
# When GPU memory limits batch_size to 4, simulate batch_size=32:
accum_steps = 8

optimizer.zero_grad(set_to_none=True)
for step, (lr_img, hr_img) in enumerate(train_loader):
    with autocast():
        pred = model(lr_img)
        loss, _ = criterion(pred, hr_img)
        loss = loss / accum_steps  # scale loss

    scaler.scale(loss).backward()

    if (step + 1) % accum_steps == 0:
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

### 8.4 Training Schedule (Recommended)

```
Phase 1: Warm-up (epochs 1-5)
  └─ LR: 1e-5 → 2e-4 (linear warm-up)
  └─ Loss: Charbonnier only (stable start)

Phase 2: Main training (epochs 6-80)
  └─ LR: Cosine anneal from 2e-4 → 1e-6
  └─ Loss: Full combined loss

Phase 3: Fine-tuning (epochs 81-100)
  └─ LR: 1e-6 (constant, very low)
  └─ Loss: Full combined loss + higher SSIM weight (0.3)
  └─ Larger patch size (256 instead of 128)
```

---

## 9. Hyperparameter Tuning

### 9.1 Key Hyperparameters

```yaml
# config.yaml — base configuration
model:
  width: 64           # NAFNet channel count [32, 64, 96] — tradeoff: speed vs quality
  enc_blocks: [2,2,4,8]
  dec_blocks: [2,2,2,2]

training:
  lr: 2.0e-4          # AdamW learning rate
  weight_decay: 1.0e-4
  batch_size: 8       # per-GPU; scale with GPU count
  patch_size: 128     # training crop; 256 for fine-tuning
  epochs: 100
  log_every: 50
  val_every: 5

loss:
  lambda_perc: 0.1
  lambda_freq: 0.05
  lambda_ssim: 0.2

data:
  num_workers: 8
  patches_per_image: 16
```

### 9.2 Tuning Strategy (Without Ray Tune)

For a hackathon, grid search over the most impactful parameters:

```python
# Most impactful parameters to sweep (in order):
# 1. width (32 vs 64 vs 96) — biggest quality/speed tradeoff
# 2. lr (1e-4 vs 2e-4 vs 5e-4) — training stability
# 3. lambda_ssim (0.1 vs 0.2 vs 0.3) — perceptual quality
# 4. patch_size (128 vs 256) — receptive field vs memory

# Simple manual sweep:
sweep_configs = [
    {'width': 32, 'lr': 2e-4, 'lambda_ssim': 0.2},  # fast baseline
    {'width': 64, 'lr': 2e-4, 'lambda_ssim': 0.2},  # main experiment
    {'width': 64, 'lr': 1e-4, 'lambda_ssim': 0.3},  # quality push
]
```

### 9.3 Optuna (Automated Tuning)

```python
import optuna

def objective(trial):
    config = {
        'width': trial.suggest_categorical('width', [32, 64]),
        'lr': trial.suggest_loguniform('lr', 1e-5, 5e-4),
        'lambda_perc': trial.suggest_float('lambda_perc', 0.05, 0.2),
        'lambda_ssim': trial.suggest_float('lambda_ssim', 0.1, 0.4),
        'epochs': 20,  # short runs for tuning
        'data_dir': 'data/',
        'batch_size': 8,
    }
    model = train(config)
    return validate(model, config, device)[0]  # return SSIM

study = optuna.create_study(
    direction='maximize',
    study_name='nafnet_sr_tuning',
    storage='sqlite:///optuna_study.db',  # persistent across sessions!
    load_if_exists=True,
)
study.optimize(objective, n_trials=20, timeout=3600*12)
print(study.best_params)
```

**Note:** Use `load_if_exists=True` so Optuna resumes previous trials if the session restarts.

---

## 10. Logging & Experiment Tracking

### 10.1 Weights & Biases (Primary)

```bash
pip install wandb
wandb login  # one-time setup
```

```python
# Log images for visual inspection
def log_val_images(model, val_loader, device, n=4):
    model.eval()
    with torch.no_grad():
        lr, hr = next(iter(val_loader))
        lr, hr = lr[:n].to(device), hr[:n].to(device)
        pred = torch.clamp(model(lr), 0, 1)

    images = []
    for i in range(n):
        images.append(wandb.Image(lr[i].cpu(), caption=f"Degraded {i}"))
        images.append(wandb.Image(pred[i].cpu(), caption=f"Restored {i}"))
        images.append(wandb.Image(hr[i].cpu(), caption=f"Ground Truth {i}"))

    wandb.log({'validation_images': images})
```

### 10.2 CSV Logging (Backup — no internet needed)

```python
import csv
from datetime import datetime

class CSVLogger:
    def __init__(self, path='training_log.csv'):
        self.path = path
        self.fields = None

    def log(self, metrics: dict):
        metrics['timestamp'] = datetime.now().isoformat()
        if self.fields is None:
            self.fields = list(metrics.keys())
            with open(self.path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()
        with open(self.path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(metrics)
```

### 10.3 TensorBoard (Local)

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=f'runs/experiment_{config["width"]}')
writer.add_scalar('Loss/train', loss.item(), global_step)
writer.add_images('Images/restored', pred, global_step)
```

```bash
tensorboard --logdir=runs/ --port=6006
```

---

## 11. Compute Requirements

### 11.1 Minimum Requirements

| Component | Minimum | Recommended | Competition H100 |
|---|---|---|---|
| GPU VRAM | 8 GB (RTX 3070) | 24 GB (RTX 3090/4090) | 80 GB (H100) |
| RAM | 32 GB | 64 GB | 512 GB |
| Storage | 500 GB SSD | 2 TB NVMe | — |
| CUDA version | 11.8 | 12.1 | 12.x |

### 11.2 Batch Size vs GPU Memory (NAFNet width=64)

| GPU | Patch Size | Batch Size | VRAM Used |
|---|---|---|---|
| RTX 3070 (8GB) | 128×128 | 4 | ~7.5 GB |
| RTX 3090 (24GB) | 128×128 | 16 | ~18 GB |
| A100 (40GB) | 256×256 | 8 | ~32 GB |
| H100 (80GB) | 256×256 | 32 | ~60 GB |

### 11.3 Training Time Estimates

| Setup | Epochs | Time/epoch | Total |
|---|---|---|---|
| RTX 3090, width=64, 10k images | 100 | ~8 min | ~13 hrs |
| A100, width=64, 10k images | 100 | ~3 min | ~5 hrs |
| H100, width=64, 10k images | 100 | ~1.5 min | ~2.5 hrs |

### 11.4 Inference Time Estimates (512×512 image, full model)

| GPU | Width=32 | Width=64 |
|---|---|---|
| RTX 3090 | ~0.04s | ~0.12s |
| A100 | ~0.02s | ~0.06s |
| H100 | ~0.01s | ~0.04s |

Well within practical limits.

---

## 12. Evaluation Metrics

### 12.1 SSIM (Structural Similarity Index)

```python
from pytorch_msssim import ssim

def compute_ssim(pred, target):
    """pred, target: [B, 1, H, W] tensors in [0, 1]"""
    return ssim(pred, target, data_range=1.0, size_average=True).item()
```

SSIM ∈ [0, 1]; higher is better. Measures luminance, contrast, and structure simultaneously. **Primary metric.**

### 12.2 PSNR (Peak Signal-to-Noise Ratio)

```python
import math

def compute_psnr(pred, target):
    """Returns PSNR in dB."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10 * math.log10(1.0 / mse)
```

PSNR in dB; higher is better. 30+ dB = good quality; 40+ dB = excellent.

### 12.3 LPIPS (Learned Perceptual Image Patch Similarity)

```python
import lpips

loss_fn = lpips.LPIPS(net='vgg').cuda()

def compute_lpips(pred, target):
    """pred, target: [B, 1, H, W] in [0, 1] → converts to [-1, 1] for LPIPS"""
    pred_norm = pred * 2 - 1
    tgt_norm = target * 2 - 1
    # LPIPS expects 3-channel; repeat grayscale
    pred_rgb = pred_norm.repeat(1, 3, 1, 1)
    tgt_rgb = tgt_norm.repeat(1, 3, 1, 1)
    return loss_fn(pred_rgb, tgt_rgb).mean().item()
```

LPIPS ∈ [0, 1]; **lower is better**. Perceptual similarity — correlates best with human judgement.

### 12.4 Full Validation Function

```python
from pytorch_msssim import ssim
import lpips

lpips_fn = lpips.LPIPS(net='vgg')

def validate(model, config, device):
    model.eval()
    val_loader = get_dataloader(config['val_dir'], batch_size=1, num_workers=4)
    all_ssim, all_psnr, all_lpips = [], [], []
    lpips_fn = lpips_fn.to(device)

    with torch.no_grad():
        for lr_img, hr_img in val_loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            pred = torch.clamp(model(lr_img), 0.0, 1.0)

            all_ssim.append(compute_ssim(pred, hr_img))
            all_psnr.append(compute_psnr(pred, hr_img))
            all_lpips.append(compute_lpips(pred, hr_img))

    metrics = {
        'SSIM':  sum(all_ssim)  / len(all_ssim),
        'PSNR':  sum(all_psnr)  / len(all_psnr),
        'LPIPS': sum(all_lpips) / len(all_lpips),
    }
    print(f"SSIM: {metrics['SSIM']:.4f} | PSNR: {metrics['PSNR']:.2f} dB | LPIPS: {metrics['LPIPS']:.4f}")
    return metrics['SSIM'], metrics['PSNR']
```

---

## 13. Inference Script (Submission-Ready)

This is the most critical file. It must run AS-IS on KLA's H100 without modification.

```python
#!/usr/bin/env python3
"""
evaluate.py — KLA SemiCon Hackathon Evaluation Script
Usage:
    python evaluate.py --input /path/to/test/images --output /path/to/output/dir

Loads trained NAFNetSR model, runs inference on all images in input dir,
writes restored images to output dir.
"""

import argparse
import time
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# ── Import model (same file or as module) ─────────────────────────────────────
# Assumes NAFNetSR class is importable from model.py in same directory
from model import NAFNetSR

def parse_args():
    p = argparse.ArgumentParser(description='Image restoration inference')
    p.add_argument('--input',  required=True, help='Path to directory of test images')
    p.add_argument('--output', required=True, help='Path to directory for output images')
    p.add_argument('--weights', default='checkpoints/best_model.pth',
                   help='Path to trained model weights')
    p.add_argument('--width', type=int, default=64,
                   help='NAFNet channel width (must match training config)')
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    return p.parse_args()

def load_model(weights_path, width, device):
    model = NAFNetSR(in_ch=1, width=width)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_image(path):
    """Load grayscale image as float32. Preserves out-of-range speckle values."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = img.astype(np.float32)
    # Normalise 8-bit or 16-bit to [0,1] scale (but don't clip speckle exceedances)
    if img.max() > 256:
        img = img / 65535.0
    elif img.max() > 1.0:
        img = img / 255.0
    return img

def save_image(img_np, path):
    """Save float32 [0,1] array as 16-bit PNG."""
    img_16 = (np.clip(img_np, 0, 1) * 65535).astype(np.uint16)
    cv2.imwrite(str(path), img_16)

@torch.no_grad()
def restore_image(model, img_np, device):
    """Run model on a single image. Returns restored float32 numpy array."""
    tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    output = model(tensor)
    output = torch.clamp(output, 0.0, 1.0)
    return output.squeeze().cpu().numpy()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading model from {args.weights}")
    model = load_model(args.weights, args.width, device)

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        list(input_dir.glob('*.png')) +
        list(input_dir.glob('*.tiff')) +
        list(input_dir.glob('*.tif')) +
        list(input_dir.glob('*.jpg'))
    )
    print(f"[INFO] Found {len(image_files)} images to process")

    total_time = 0.0
    for img_path in tqdm(image_files, desc='Restoring'):
        img_np = load_image(img_path)

        t0 = time.perf_counter()
        restored = restore_image(model, img_np, device)
        total_time += time.perf_counter() - t0

        out_path = output_dir / img_path.name
        save_image(restored, out_path)

    avg_ms = (total_time / len(image_files)) * 1000
    print(f"\n[DONE] Processed {len(image_files)} images")
    print(f"[TIME] Average inference time: {avg_ms:.1f} ms/image")
    print(f"[OUT]  Restored images saved to: {output_dir}")

if __name__ == '__main__':
    main()
```

---

## 14. GitHub Repository Structure

```
TeamName_KLA_PS01/
├── README.md                    ← Setup + inference instructions (clear, complete)
├── requirements.txt             ← pip freeze output
├── config.yaml                  ← All hyperparameters (no magic numbers in code)
│
├── model.py                     ← NAFNetSR architecture
├── loss.py                      ← CombinedLoss, PerceptualLoss, etc.
├── dataset.py                   ← SemiconDataset, get_dataloader
├── metrics.py                   ← compute_ssim, compute_psnr, compute_lpips
│
├── train.py                     ← Session-resumable training script
├── evaluate.py                  ← [CRITICAL] Standalone inference script
├── validate.py                  ← Run metrics on a val set
│
├── checkpoints/
│   ├── best_model.pth           ← Best validation checkpoint (Git LFS or Drive link)
│   └── latest.pth               ← Most recent checkpoint
│
├── outputs/
│   └── test_restored/           ← Your model's output on the test set
│       ├── image_001.png
│       └── ...
│
├── notebooks/
│   └── exploration.ipynb        ← EDA, visualisations (not for inference)
│
└── docs/
    └── architecture.png         ← Architecture diagram
```

### README.md template

```markdown
# KLA Image Restoration — [Team Name]

## Setup
```bash
git clone https://github.com/your-org/TeamName_KLA_PS01
cd TeamName_KLA_PS01
pip install -r requirements.txt
```

## Download weights
```bash
# Option A: Git LFS
git lfs pull

# Option B: Drive
wget -O checkpoints/best_model.pth "YOUR_GOOGLE_DRIVE_LINK"
```

## Run inference
```bash
python evaluate.py \
  --input /path/to/test/images \
  --output /path/to/output \
  --weights checkpoints/best_model.pth
```

## Train from scratch
```bash
python train.py --config config.yaml --data_dir /path/to/dataset
```
```

---

## 15. Learning Resources

### Core Papers (Read in This Order)

| Topic | Paper | Link |
|---|---|---|
| **NAFNet** (primary backbone) | *Simple Baselines for Image Restoration* (Chen et al., ECCV 2022) | https://arxiv.org/abs/2204.04676 |
| **ESRGAN/RRDB** (SR module) | *ESRGAN: Enhanced Super-Resolution GANs* (Wang et al., 2018) | https://arxiv.org/abs/1809.00219 |
| **SwinIR** (if you try transformers) | *SwinIR: Image Restoration Using Swin Transformer* (Liang et al., 2021) | https://arxiv.org/abs/2108.10257 |
| **Restormer** (SOTA transformer) | *Restormer: Efficient Transformer for High-Resolution Image Restoration* (Zamir et al., 2022) | https://arxiv.org/abs/2111.09881 |
| **Perceptual Loss** | *Perceptual Losses for Real-Time Style Transfer* (Johnson et al., 2016) | https://arxiv.org/abs/1603.08155 |
| **SSIM** | *Image Quality Assessment: From Error Visibility to Structural Similarity* (Wang et al., 2004) | https://doi.org/10.1109/TIP.2003.819861 |
| **Frequency Loss** | *Focal Frequency Loss for Image Reconstruction* (Jiang et al., 2021) | https://arxiv.org/abs/2012.12821 |
| **Speckle noise model** | *Speckle in Optical Coherence Tomography* (Schmitt et al., 1999) | IEEE Journal of Selected Topics in QE |
| **Blind Image Restoration Survey** | Zhai et al., IEEE Access 2023 (cited in hackathon slides) | https://doi.org/10.1109/ACCESS.2023.3234456 |
| **Algorithm Unrolling** | Monga et al., IEEE Signal Proc. Mag. 2021 (cited in slides) | https://arxiv.org/abs/1912.10557 |
| **Data Augmentation Survey** | T. Kumar et al., IEEE Access 2024 (cited in slides) | https://doi.org/10.1109/ACCESS.2024.3354761 |
| **Loss Functions Survey** | Terven et al., Artif Intell Rev 2025 (cited in slides) | https://doi.org/10.1007/s10462-024-10958-9 |

### Code References

| Resource | Link |
|---|---|
| Official NAFNet implementation | https://github.com/megvii-research/NAFNet |
| BasicSR framework (includes ESRGAN, SwinIR) | https://github.com/XPixelGroup/BasicSR |
| pytorch-msssim | https://github.com/VainF/pytorch-msssim |
| LPIPS library | https://github.com/richzhang/PerceptualSimilarity |
| Albumentations (augmentation) | https://github.com/albumentations-team/albumentations |
| Optuna (hyperparameter tuning) | https://optuna.org |
| Weights & Biases | https://wandb.ai |

### Background Reading

| Topic | Resource |
|---|---|
| Speckle noise fundamentals | https://en.wikipedia.org/wiki/Speckle_pattern |
| PixelShuffle explained | https://paperswithcode.com/method/pixelshuffle |
| Mixed precision training | https://pytorch.org/docs/stable/amp.html |
| DataLoader best practices | https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html |
| Image restoration benchmarks | https://paperswithcode.com/task/image-restoration |

---

## Quick Start — Recommended Day-by-Day Plan

```
Day 1: Setup
  □ Clone NAFNet repo; run on sample data; confirm GPU works
  □ Build SemiconDataset; visualise 10 sample pairs; check LR out-of-range values
  □ Set up wandb project; confirm logging works

Day 2: Baseline
  □ Train width=32 NAFNet for 20 epochs (denoising only, no SR head)
  □ Establish baseline SSIM/PSNR on val split
  □ Add SR head; verify output resolution is 2× input

Day 3-4: Full Model
  □ Add all loss terms one by one; observe effect on val metrics
  □ Train width=64 model for 50 epochs; compare to baseline
  □ Tune λ_ssim and λ_perc via 3-run sweep

Day 5: Polish
  □ Train best config to 100 epochs
  □ Visual inspection of before/after on diverse samples
  □ Test evaluate.py on fresh machine; time per image
  □ Finalise GitHub repo; upload weights; record demo video
```

---

*Generated for KLA SemiCon AI Hackathon — Problem Statement PS01: AI-Based Restoration of Degraded Semiconductor Inspection Images.*
