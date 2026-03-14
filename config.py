"""
Configuration parameters for the Night Surveillance Enhancement System.
"""
import os
import torch

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "LOL")
TRAIN_LOW_DIR = os.path.join(DATASET_DIR, "train", "low")
TRAIN_HIGH_DIR = os.path.join(DATASET_DIR, "train", "high")
TEST_LOW_DIR = os.path.join(DATASET_DIR, "test", "low")
TEST_HIGH_DIR = os.path.join(DATASET_DIR, "test", "high")
PRETRAINED_DIR = os.path.join(BASE_DIR, "pretrained")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model ───────────────────────────────────────────────────────────────────
NUM_CURVES = 6          # Number of curve iterations (6 gives ~95% quality of 8)
INPUT_CHANNELS = 3
HIDDEN_CHANNELS = 32

# ─── Training ────────────────────────────────────────────────────────────────
EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 128        # Training image crop size (128 for CPU, 512 for GPU)
NUM_WORKERS = 0         # 0 for Windows compatibility

# ─── Loss Weights ────────────────────────────────────────────────────────────
W_SPATIAL = 1.0         # Spatial consistency loss
W_EXPOSURE = 10.0       # Exposure control loss
W_COLOR = 5.0           # Color constancy loss
W_ILLUMINATION = 200.0  # Illumination smoothness loss
W_RECONSTRUCTION = 8.0  # L1 reconstruction loss (increased to compensate no VGG)
W_PERCEPTUAL = 0.0      # Perceptual (VGG) loss — disabled for CPU (set >0 for GPU)
W_SSIM = 1.0            # SSIM loss

EXPOSURE_MEAN = 0.6     # Target mean exposure level

# ─── Video Enhancement ───────────────────────────────────────────────────────
TEMPORAL_WEIGHT = 0.85   # Exponential moving average weight for temporal consistency
VIDEO_BATCH_SIZE = 1     # Frames to process at a time (1 for CPU)

# ─── Multi-Exposure Fusion ───────────────────────────────────────────────────
MEF_EXPOSURES = [0.5, 1.2, 2.0]  # Gamma values for generating exposures

# ─── Video Training ──────────────────────────────────────────────────────────
VIDEO_DIR = os.path.join(BASE_DIR, "datasets", "videos")  # Put dark videos here
VIDEO_FRAME_INTERVAL = 10   # Sample every Nth frame (reduces frame redundancy)
