# Night Surveillance Enhancement System

**Deep Learning-Based Low-Light Image & Video Enhancement using Zero-DCE**

---

## What This Project Does

This system automatically brightens and restores dark, low-quality surveillance footage and images using a trained neural network. It works on:

- **Photos** — upload a dark image, get a clear enhanced version instantly
- **Videos** — process surveillance footage frame by frame with zero flickering
- **Mixed-exposure images** — scenes with both very dark and very bright areas at once

The model runs **entirely on your computer** — no internet required after setup, no GPU needed.

---

## One-Click Setup

```bash
python run.py
```

That single command automatically:
1. Installs all required Python packages
2. Downloads the training dataset (LOL dataset, ~1 GB)
3. Trains the model (~30–60 min on CPU)
4. Opens the web interface at **http://localhost:8501**

---

## Manual Step-by-Step Setup

Prefer to run each step yourself? Do it in order:

```bash
# Step 1 — Install all packages
pip install -r requirements.txt

# Step 2 — Download the LOL training dataset
python download_dataset.py

# Step 3 — Train the model (saves to pretrained/best_model.pth)
python run_train.py

# Step 4 — Launch the web app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> **Already have a trained model?** Skip Steps 2–3 and go straight to Step 4.

---

## Using the Web Interface

The app has four tabs at the top:

### Tab 1 — Image Enhancement
| Step | Action |
|---|---|
| 1 | Click **Browse files** → upload a dark/night image (PNG, JPG, BMP) |
| 2 | Check **Post-Processing** for extra denoising and sharpening |
| 3 | Click **Enhance Image** |
| 4 | View the before/after comparison side by side |
| 5 | Click **Download Enhanced Image** to save the result |

### Tab 2 — Video Enhancement
| Step | Action |
|---|---|
| 1 | Click **Browse files** → upload a surveillance video (MP4, AVI, MOV) |
| 2 | Adjust the **Temporal Smoothing** slider (higher = smoother, less flicker) |
| 3 | Click **Enhance Video** |
| 4 | The enhanced video plays directly in the browser |
| 5 | Click **Download Enhanced Video** to save |

### Tab 3 — Multi-Exposure Fusion
Best for images where some areas are **too dark** and others are **overexposed**.
Upload the image, choose a fusion method, and click **Apply Fusion**.

### Tab 4 — Object Detection
| Step | Action |
|---|---|
| 1 | Upload a dark surveillance image |
| 2 | Choose a YOLO model (yolov8x = best accuracy, yolov8n = fastest) |
| 3 | Adjust the **Confidence Threshold** slider |
| 4 | Check **Enhance before detection** (recommended for dark images) |
| 5 | Click **Detect Objects** — see original, enhanced, and annotated side-by-side |

The pipeline first brightens the image with Zero-DCE, then runs YOLOv8 detection on the enhanced result. This dramatically improves detection accuracy in dark scenes.

### Tab 5 — About
Technical overview of the system and model architecture.

---

## Command-Line Usage

You can enhance files directly from the terminal without the web app:

```bash
# Enhance a single image
python enhance_image.py --input dark_photo.jpg

# Enhance an entire folder of images
python enhance_image.py --input my_photos/ --output results/enhanced/

# Enhance a video
python enhance_video.py --input surveillance.mp4

# Enhance video with side-by-side comparison output
python enhance_video.py --input surveillance.mp4 --comparison

# Detect objects in a dark image (enhance + YOLOv8x)
python detect.py --input dark_photo.jpg

# Evaluate model accuracy on the test set
python test.py --save_visuals

# Resume/fine-tune from a saved checkpoint
python fast_train.py

# Full training with custom settings
python train.py --epochs 200 --batch_size 8 --image_size 512
```

---

## Project Structure

```
NightSurveillance_Project/
│
├── run.py                    ← ONE-CLICK LAUNCHER (start here)
├── app.py                    ← Streamlit web interface
├── config.py                 ← All settings in one place
├── requirements.txt          ← Python dependencies
│
├── models/
│   └── zero_dce_net.py       ← Neural network architecture (Zero-DCE)
│
├── utils/
│   ├── dataloader.py         ← Dataset loading & augmentation
│   ├── losses.py             ← 6 training loss functions
│   └── metrics.py            ← PSNR / SSIM / MAE evaluation
│
├── enhance_image.py          ← Image enhancement pipeline
├── enhance_video.py          ← Video enhancement pipeline (H.264 output)
├── multi_exposure_fusion.py  ← Fusion for mixed-exposure images
├── detect.py                 ← YOLOv8x object detection + Zero-DCE pipeline
│
├── run_train.py              ← Main training script (CPU-optimized)
├── fast_train.py             ← Resume/fine-tune from checkpoint
├── train.py                  ← Full training with all CLI options
├── train_video.py            ← Fine-tune on surveillance video frames
├── test.py                   ← Model evaluation with visualizations
├── download_dataset.py       ← Download the LOL dataset
├── main.py                   ← Unified CLI entry point
│
├── pretrained/
│   └── best_model.pth        ← Saved model weights (created after training)
│
├── datasets/
│   └── LOL/                  ← Training data (created after download)
│       ├── train/low/        ← 485 dark training images
│       ├── train/high/       ← 485 matching normal-light images
│       ├── test/low/         ← 15 dark test images
│       └── test/high/        ← 15 matching normal-light images
│
└── results/                  ← Enhanced outputs saved here
    ├── images/               ← Enhanced photos
    └── evaluation/           ← Test metrics and comparison grids
```

---

## How It Works

### The Enhancement Formula

The model does not simply increase brightness — that would amplify noise and destroy detail. Instead, it learns a custom **Light Enhancement curve** that is applied per pixel, per channel, 6 times:

```
LE(x) = x + α × x × (1 - x)
```

- `x` = pixel intensity (0 to 1)
- `α` = learned parameter (different for every pixel and every iteration)

This means dark areas get lifted gently, bright areas stay protected, and the curve adapts to the specific content of the image.

### Network Architecture (Zero-DCE)

A compact 7-layer CNN with only **~79,000 parameters** — lightweight enough to run on any CPU in real time:

```
Input Image  [B × 3 × H × W]
     │
     ├─ Conv1 → ReLU  (3 → 32 channels)   ─────────────────────┐
     │                                                           │
     ├─ Conv2 → ReLU  (32 → 32)  ──────────────────────┐        │
     │                                                  │        │
     ├─ Conv3 → ReLU  (32 → 32)  ─────────────┐        │        │
     │                                         │        │        │
     └─ Conv4 → ReLU  (32 → 32)  ← bottleneck │        │        │
          │                                    │        │        │
     Conv5 → ReLU  (64 → 32)  ◄───────────────┘        │        │
          │                                             │        │
     Conv6 → ReLU  (64 → 32)  ◄────────────────────────┘        │
          │                                                       │
     Conv7 → Tanh  (64 → 18)  ◄─────────────────────────────────┘
          │
          │  18 channels = 3 RGB channels × 6 curve iterations
          │
     Apply LE curves 6× to input image
          │
     Enhanced Image  [B × 3 × H × W]
```

Skip connections feed earlier feature maps into the decoder so the network retains fine texture details that would otherwise be lost.

### Training Loss Functions

The model is trained with six combined losses simultaneously:

| Loss | Type | Purpose |
|---|---|---|
| Spatial Consistency | Self-supervised | Keeps spatial structure — edges stay sharp |
| Exposure Control | Self-supervised | Targets average brightness of ~0.6 (natural) |
| Color Constancy | Self-supervised | Balances RGB channels to avoid color cast |
| Illumination Smoothness | Self-supervised | Prevents sudden jumps in the curve maps |
| L1 Reconstruction | Supervised | Pixel-exact match to ground truth normal images |
| SSIM | Supervised | Preserves structural similarity to ground truth |

### Video — Temporal Consistency

Processing video frame by frame without any coordination causes visible **flickering** (each frame gets slightly different curve parameters). This is solved with **Exponential Moving Average (EMA)**:

```
curves_t = α × curves_{t-1}  +  (1 − α) × new_curves_t
```

The `α` slider in the app (default **0.85**) controls smoothing:
- **Higher (0.9+)** → smoother transitions, less flicker, slightly slower to react
- **Lower (0.5)** → more responsive to sudden lighting changes, may flicker slightly

---

## Model Comparison

| Model | Architecture | PSNR (dB) | SSIM | Multi-Scale | Attention | GPU Required |
|---|---|---|---|---|---|---|
| Zero-DCE *(this project)* | Lightweight CNN | ~18–22 | ~0.65 | No | No | No |
| DnCNN | Plain CNN | 28.61 | 0.83 | No | No | Recommended |
| U-Net | Encoder-Decoder | 30.14 | 0.86 | No | Partial | Recommended |
| RetinexNet | Retinex + CNN | 16.77 | 0.56 | No | No | Recommended |
| EnlightenGAN | GAN-based | 17.48 | 0.65 | No | No | Required |
| KinD | Decomposition Net | 20.87 | 0.80 | No | Partial | Recommended |
| SNR-Aware | Transformer | 21.48 | 0.85 | No | Yes | Required |
| **MIRNet** | **Multi-Scale Residual** | **24.14** | **0.83** | **Yes (3 scales)** | **Yes** | **Required** |

**Why Zero-DCE here?**
Zero-DCE was chosen because it is the only model that runs on CPU in real time with no GPU requirement. It processes any image resolution without resizing, and achieves solid enhancement quality with under 79K parameters.

**For highest quality** (research/production): **MIRNet** is state-of-the-art, using multi-scale processing + dual attention (channel + spatial) + selective kernel fusion — but requires a dedicated GPU.

---

## Configuration Reference

All settings live in `config.py`. Key values you may want to change:

```python
# ── Model ──────────────────────────────────────────────────────
NUM_CURVES      = 6      # Curve iterations (more = stronger effect, slower)
HIDDEN_CHANNELS = 32     # Network width (more = better quality, more RAM)

# ── Training ────────────────────────────────────────────────────
EPOCHS          = 100    # Training epochs
BATCH_SIZE      = 2      # 2 for CPU, 8 for GPU
IMAGE_SIZE      = 128    # Crop size: 128 for CPU, 512 for GPU
LEARNING_RATE   = 1e-4

# ── Loss Weights ────────────────────────────────────────────────
W_SPATIAL       = 1.0    # Spatial consistency
W_EXPOSURE      = 10.0   # Exposure control
W_COLOR         = 5.0    # Color constancy
W_ILLUMINATION  = 200.0  # Illumination smoothness
W_RECONSTRUCTION= 8.0    # L1 pixel accuracy
W_SSIM          = 1.0    # Structural similarity
W_PERCEPTUAL    = 0.0    # VGG perceptual (set >0 only with GPU)

# ── Video ────────────────────────────────────────────────────────
TEMPORAL_WEIGHT = 0.85   # EMA smoothing (0 = none, 1 = fully frozen)

# ── Multi-Exposure Fusion ────────────────────────────────────────
MEF_EXPOSURES   = [0.5, 1.2, 2.0]   # Gamma values for exposure generation
```

---

## Training Reference

| Setting | CPU Mode | GPU Mode |
|---|---|---|
| Script | `python run_train.py` | `python train.py --epochs 200 --batch_size 8 --image_size 512` |
| Batch size | 2 | 8 |
| Image crop | 128×128 | 512×512 |
| Epochs | 100 | 200 |
| Time | ~30–60 min | ~5–10 min |
| Perceptual loss | Disabled | Enable (`W_PERCEPTUAL = 0.04`) |

**Resume / fine-tune** a trained model:
```bash
python fast_train.py            # Quick 35-epoch fine-tune on 64px crops
python train.py --resume pretrained/best_model.pth --epochs 200
```

---

## Expected Results

After 100 epochs of training on the LOL dataset:

| Metric | Expected Range | What It Means |
|---|---|---|
| PSNR | 16–22 dB | Higher = more pixel-accurate vs. ground truth |
| SSIM | 0.55–0.80 | Higher = more structurally similar (1.0 = perfect) |
| MAE | 0.05–0.15 | Lower = less average pixel error |

View results after evaluation:
```
results/evaluation/
├── metrics.txt         ← Per-image PSNR, SSIM, MAE table
├── results_grid.png    ← Visual grid: Input | Enhanced | Ground Truth
└── eval_*.png          ← Individual comparison images
```

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 or 3.13 |
| RAM | 4 GB | 8 GB+ |
| GPU | Not required | NVIDIA CUDA (for faster training) |
| Storage | 2 GB | 5 GB |
| OS | Windows / Linux / macOS | Any |

### Python Packages

| Package | Version | Purpose |
|---|---|---|
| torch | ≥ 2.0.0 | Neural network training & inference |
| torchvision | ≥ 0.15.0 | Image transforms |
| opencv-python | ≥ 4.8.0 | Video processing, image filters |
| streamlit | ≥ 1.28.0 | Web interface |
| Pillow | ≥ 10.0.0 | Image I/O |
| scikit-image | ≥ 0.21.0 | SSIM metric |
| numpy | ≥ 1.24.0 | Array operations |
| matplotlib | ≥ 3.7.0 | Result plots |
| gdown | ≥ 4.7.0 | Dataset download |
| tqdm | ≥ 4.65.0 | Progress bars |
| ultralytics | ≥ 8.0.0 | YOLOv8 object detection |

### Optional — ffmpeg (for Video)

Install **ffmpeg** to guarantee H.264 video encoding (required for playback in all browsers):

1. Download from **https://ffmpeg.org/download.html**
2. Extract and add the `bin/` folder to your system **PATH**
3. Verify: `ffmpeg -version`

Without ffmpeg, the app falls back to OpenCV codecs which may not play in all browsers.

---

## Troubleshooting

**"No trained model found" warning in the sidebar**
→ Run `python run_train.py` to train the model. It saves to `pretrained/best_model.pth`.

**Video doesn't play in browser after enhancement**
→ Install ffmpeg (see above). Without it, the mp4v codec is used, which Safari and some Chrome versions block.

**Dataset download fails**
→ Download manually from https://daooshee.github.io/BMVC2018website/
→ Extract to `datasets/LOL/` with subfolders `train/low/`, `train/high/`, `test/low/`, `test/high/`

**Out of memory during training**
→ In `config.py`, reduce `IMAGE_SIZE = 64` and `BATCH_SIZE = 1`

**Training too slow**
→ Use `python fast_train.py` for rapid 35-epoch fine-tuning (64px crops, much faster)

**Streamlit app not starting**
→ Run `pip install streamlit --upgrade`, then `python -m streamlit run app.py`

**Poor enhancement quality**
→ Train for more epochs (`--epochs 200`) or fine-tune with `python fast_train.py`
→ Make sure the dataset is correct (paired low/high images of the same scene)

---

## Dataset: LOL (Low-Light)

- **Paper**: Wei, C., et al. "Deep Retinex Decomposition for Low-Light Enhancement." BMVC 2018
- **Size**: 485 training pairs + 15 test pairs
- **Format**: Paired images — same scene, different exposure (dark vs. normal)
- **Resolution**: ~400×600 pixels each image

---

## References

1. **Zero-DCE**: Guo, C., et al. "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement." CVPR 2020.
2. **LOL Dataset**: Wei, C., et al. "Deep Retinex Decomposition for Low-Light Enhancement." BMVC 2018.
3. **MIRNet**: Zamir, S., et al. "Learning Enriched Features for Real Image Restoration and Enhancement." ECCV 2020.
4. **Exposure Fusion**: Mertens, T., et al. "Exposure Fusion." Pacific Graphics 2007.
