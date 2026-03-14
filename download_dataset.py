"""
Download the LOL (Low-Light) dataset for training and evaluation.

LOL Dataset (Chen Wei et al., BMVC 2018):
    - 485 training pairs + 15 test pairs
    - Low-light and normal-light paired images
    - Widely used benchmark for low-light image enhancement

Usage:
    python download_dataset.py
"""

import os
import zipfile
import gdown

# Google Drive file ID for LOL dataset
# LOL dataset v1
LOL_GDRIVE_ID = "157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB"
LOL_URL = f"https://drive.google.com/uc?id={LOL_GDRIVE_ID}"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
ZIP_PATH = os.path.join(DATASET_DIR, "LOL.zip")
LOL_DIR = os.path.join(DATASET_DIR, "LOL")


def download_lol_dataset():
    """Download and extract the LOL dataset."""
    os.makedirs(DATASET_DIR, exist_ok=True)

    # Check if already downloaded
    train_low = os.path.join(LOL_DIR, "train", "low")
    if os.path.exists(train_low) and len(os.listdir(train_low)) > 0:
        print(f"LOL dataset already exists at {LOL_DIR}")
        print(f"  Training pairs: {len(os.listdir(train_low))}")
        return

    print("=" * 60)
    print("Downloading LOL (Low-Light) Dataset")
    print("=" * 60)
    print(f"Source: Google Drive (ID: {LOL_GDRIVE_ID})")
    print(f"Destination: {LOL_DIR}")
    print()

    # Download from Google Drive
    print("Downloading... (this may take a few minutes)")
    try:
        gdown.download(LOL_URL, ZIP_PATH, quiet=False)
    except Exception as e:
        print(f"\nAutomatic download failed: {e}")
        print("\n" + "=" * 60)
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print("=" * 60)
        print(f"1. Go to: https://drive.google.com/uc?id={LOL_GDRIVE_ID}")
        print(f"2. Download the ZIP file")
        print(f"3. Extract to: {LOL_DIR}")
        print(f"   The structure should be:")
        print(f"   {LOL_DIR}/")
        print(f"   ├── train/")
        print(f"   │   ├── low/   (485 images)")
        print(f"   │   └── high/  (485 images)")
        print(f"   └── test/")
        print(f"       ├── low/   (15 images)")
        print(f"       └── high/  (15 images)")
        print()
        print("Alternative: Download from https://daooshee.github.io/BMVC2018website/")
        return

    # Extract
    if os.path.exists(ZIP_PATH):
        print("\nExtracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)
        print("Extraction complete!")

        # Reorganize if needed (handle different zip structures)
        _reorganize_dataset()

        # Cleanup
        os.remove(ZIP_PATH)
        print(f"Cleaned up ZIP file.")

    # Verify
    _verify_dataset()


def _reorganize_dataset():
    """Handle different possible zip structures and reorganize to expected format."""
    expected_dirs = [
        os.path.join(LOL_DIR, "train", "low"),
        os.path.join(LOL_DIR, "train", "high"),
        os.path.join(LOL_DIR, "test", "low"),
        os.path.join(LOL_DIR, "test", "high"),
    ]

    # Check if structure is already correct
    if all(os.path.exists(d) for d in expected_dirs):
        return

    # Look for alternative structures
    for root, dirs, files in os.walk(DATASET_DIR):
        if "our485" in dirs or "eval15" in dirs:
            # Common structure: our485/low, our485/high, eval15/low, eval15/high
            our485 = os.path.join(root, "our485")
            eval15 = os.path.join(root, "eval15")

            for d in expected_dirs:
                os.makedirs(d, exist_ok=True)

            if os.path.exists(our485):
                _move_files(os.path.join(our485, "low"), os.path.join(LOL_DIR, "train", "low"))
                _move_files(os.path.join(our485, "high"), os.path.join(LOL_DIR, "train", "high"))
            if os.path.exists(eval15):
                _move_files(os.path.join(eval15, "low"), os.path.join(LOL_DIR, "test", "low"))
                _move_files(os.path.join(eval15, "high"), os.path.join(LOL_DIR, "test", "high"))
            return


def _move_files(src, dst):
    """Move image files from src to dst directory."""
    import shutil
    if not os.path.exists(src):
        return
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            shutil.move(os.path.join(src, f), os.path.join(dst, f))


def _verify_dataset():
    """Verify dataset integrity."""
    print("\n" + "=" * 60)
    print("Dataset Verification")
    print("=" * 60)

    dirs = {
        "Train Low": os.path.join(LOL_DIR, "train", "low"),
        "Train High": os.path.join(LOL_DIR, "train", "high"),
        "Test Low": os.path.join(LOL_DIR, "test", "low"),
        "Test High": os.path.join(LOL_DIR, "test", "high"),
    }

    all_ok = True
    for name, path in dirs.items():
        if os.path.exists(path):
            count = len([f for f in os.listdir(path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            status = "OK" if count > 0 else "EMPTY"
            print(f"  {name:12s}: {count:4d} images  [{status}]")
            if count == 0:
                all_ok = False
        else:
            print(f"  {name:12s}: MISSING")
            all_ok = False

    if all_ok:
        print("\nDataset is ready for training!")
    else:
        print("\nWARNING: Some directories are missing or empty.")
        print("Please check the dataset structure manually.")


if __name__ == "__main__":
    download_lol_dataset()
