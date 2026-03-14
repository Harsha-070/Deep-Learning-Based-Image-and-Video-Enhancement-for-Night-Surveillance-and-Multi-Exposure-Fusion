"""
One-click launcher for the Night Surveillance Enhancement System.

Just run:
    python run.py

This script will:
    1. Install dependencies (if needed)
    2. Download the LOL dataset (if needed)
    3. Train the model (if no pretrained model exists)
    4. Launch the Streamlit web app
"""

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

PYTHON = sys.executable
MODEL_PATH = os.path.join(BASE_DIR, "pretrained", "best_model.pth")
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "LOL", "train", "low")


def run_cmd(cmd, desc):
    """Run a command with a description."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"\n  ERROR: {desc} failed.")
        print(f"  You can try running it manually: {' '.join(cmd)}")
        return False
    return True


def check_deps():
    """Check if key dependencies are installed."""
    try:
        import torch
        import streamlit
        import cv2
        return True
    except ImportError:
        return False


def main():
    print()
    print("=" * 60)
    print("  Night Surveillance Enhancement - One-Click Setup")
    print("=" * 60)

    # Step 1: Install dependencies
    if not check_deps():
        print("\n  [1/4] Installing dependencies...")
        if not run_cmd([PYTHON, "-m", "pip", "install", "-r", "requirements.txt"],
                       "Installing Python dependencies"):
            return
    else:
        print("\n  [1/4] Dependencies already installed. OK")

    # Step 2: Download dataset
    if os.path.exists(DATASET_DIR) and len(os.listdir(DATASET_DIR)) > 0:
        print(f"\n  [2/4] Dataset already downloaded. OK")
    else:
        print("\n  [2/4] Downloading LOL dataset...")
        if not run_cmd([PYTHON, "download_dataset.py"],
                       "Downloading LOL Dataset"):
            print("  You can continue without the dataset if you already have a trained model.")

    # Step 3: Train model (if needed)
    if os.path.exists(MODEL_PATH):
        print(f"\n  [3/4] Trained model found. OK")
    else:
        if os.path.exists(DATASET_DIR) and len(os.listdir(DATASET_DIR)) > 0:
            print("\n  [3/4] Training model (this takes a few minutes)...")
            if not run_cmd([PYTHON, "run_train.py"],
                           "Training Zero-DCE Model"):
                return
        else:
            print("\n  [3/4] No dataset and no model found. Cannot proceed.")
            print("  Please download the dataset first: python download_dataset.py")
            return

    # Step 4: Launch app
    print("\n  [4/4] Launching web app...")
    print("\n" + "=" * 60)
    print("  Open your browser to: http://localhost:8501")
    print("=" * 60 + "\n")

    subprocess.run([PYTHON, "-m", "streamlit", "run", "app.py",
                    "--server.port", "8501",
                    "--server.headless", "true"])


if __name__ == "__main__":
    main()
