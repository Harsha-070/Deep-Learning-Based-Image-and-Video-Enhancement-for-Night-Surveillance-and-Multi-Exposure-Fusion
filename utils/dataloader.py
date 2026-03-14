"""
Dataset loading utilities for the LOL (Low-Light) dataset.

LOL Dataset:
    - 485 training image pairs (low-light / normal-light)
    - 15 test image pairs
    - Source: https://daooshee.github.io/BMVC2018website/

Supports data augmentation for training:
    - Random cropping
    - Random horizontal/vertical flipping
    - Random rotation
"""

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop


class LOLDataset(Dataset):
    """
    LOL (Low-Light) paired dataset for training and evaluation.

    Each sample consists of a low-light image and its corresponding
    normal-light ground truth.
    """

    def __init__(self, low_dir, high_dir, image_size=512, augment=True):
        """
        Args:
            low_dir: Path to low-light images directory.
            high_dir: Path to normal-light images directory.
            image_size: Size to crop/resize images for training.
            augment: Whether to apply data augmentation.
        """
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.image_size = image_size
        self.augment = augment

        # Get sorted list of image filenames
        self.low_images = sorted([
            f for f in os.listdir(low_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        self.high_images = sorted([
            f for f in os.listdir(high_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

        assert len(self.low_images) == len(self.high_images), \
            f"Mismatch: {len(self.low_images)} low vs {len(self.high_images)} high images"

        print(f"  Loaded {len(self.low_images)} image pairs from {low_dir}")

    def __len__(self):
        return len(self.low_images)

    def _apply_augmentation(self, low_img, high_img):
        """Apply synchronized augmentation to both images."""
        # Random crop
        if min(low_img.size) >= self.image_size:
            i, j, h, w = RandomCrop.get_params(
                low_img,
                output_size=(self.image_size, self.image_size)
            )
        else:
            i, j, h, w = 0, 0, low_img.size[1], low_img.size[0]

        if min(low_img.size) >= self.image_size:
            low_img = TF.crop(low_img, i, j, h, w)
            high_img = TF.crop(high_img, i, j, h, w)

        # Random horizontal flip
        if random.random() > 0.5:
            low_img = TF.hflip(low_img)
            high_img = TF.hflip(high_img)

        # Random vertical flip
        if random.random() > 0.5:
            low_img = TF.vflip(low_img)
            high_img = TF.vflip(high_img)

        # Random rotation (0, 90, 180, 270 degrees)
        angle = random.choice([0, 90, 180, 270])
        if angle > 0:
            low_img = TF.rotate(low_img, angle)
            high_img = TF.rotate(high_img, angle)

        return low_img, high_img

    def __getitem__(self, idx):
        """
        Returns:
            low_tensor: Low-light image tensor [3, H, W] in [0, 1].
            high_tensor: Normal-light image tensor [3, H, W] in [0, 1].
            filename: Image filename.
        """
        low_path = os.path.join(self.low_dir, self.low_images[idx])
        high_path = os.path.join(self.high_dir, self.high_images[idx])

        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')

        # Resize to consistent size if needed
        if low_img.size != high_img.size:
            min_w = min(low_img.size[0], high_img.size[0])
            min_h = min(low_img.size[1], high_img.size[1])
            low_img = low_img.resize((min_w, min_h), Image.LANCZOS)
            high_img = high_img.resize((min_w, min_h), Image.LANCZOS)

        if self.augment:
            low_img, high_img = self._apply_augmentation(low_img, high_img)

        # Resize to target size
        low_img = low_img.resize((self.image_size, self.image_size), Image.LANCZOS)
        high_img = high_img.resize((self.image_size, self.image_size), Image.LANCZOS)

        # Convert to tensors [0, 1]
        low_tensor = TF.to_tensor(low_img)
        high_tensor = TF.to_tensor(high_img)

        return low_tensor, high_tensor, self.low_images[idx]


class UnpairedLowLightDataset(Dataset):
    """
    Unpaired dataset for inference or zero-reference training.
    Only requires low-light images (no ground truth).
    """

    def __init__(self, image_dir, image_size=None):
        self.image_dir = image_dir
        self.image_size = image_size
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        print(f"  Loaded {len(self.images)} images from {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(path).convert('RGB')

        if self.image_size is not None:
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)

        return TF.to_tensor(img), self.images[idx]


class VideoFrameDataset(Dataset):
    """
    Dataset that extracts frames from video files for self-supervised training.

    Scans a directory for video files, samples frames at a fixed interval,
    and returns them as tensors. No ground-truth labels needed — Zero-DCE's
    self-supervised losses handle training without paired data.

    Directory layout (put your dark videos here):
        datasets/videos/
            surveillance1.mp4
            nightcam.avi
            ...
    """

    def __init__(self, video_dir, image_size=128, frame_interval=10, augment=True):
        """
        Args:
            video_dir: Directory containing video files.
            image_size: Crop/resize size for training patches.
            frame_interval: Extract one frame every N frames (reduces redundancy).
            augment: Apply random flip/rotation augmentation.
        """
        import cv2 as _cv2

        self.image_size = image_size
        self.augment = augment
        self.frames = []  # list of numpy arrays [H, W, 3] RGB

        video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
        video_files = sorted([
            os.path.join(video_dir, f) for f in os.listdir(video_dir)
            if f.lower().endswith(video_exts)
        ])

        if not video_files:
            raise ValueError(f"No video files found in {video_dir}")

        print(f"  Extracting frames from {len(video_files)} video(s)...")
        for vpath in video_files:
            cap = _cv2.VideoCapture(vpath)
            total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            count = 0
            extracted = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if count % frame_interval == 0:
                    # Convert BGR → RGB
                    self.frames.append(_cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB))
                    extracted += 1
                count += 1
            cap.release()
            print(f"    {os.path.basename(vpath)}: {total} frames → {extracted} sampled")

        print(f"  Total frames for training: {len(self.frames)}")

    def __len__(self):
        return len(self.frames)

    def _augment(self, img):
        """Apply random flip and rotation."""
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
        if random.random() > 0.5:
            img = np.flipud(img).copy()
        angle = random.choice([0, 90, 180, 270])
        if angle == 90:
            img = np.rot90(img, 1).copy()
        elif angle == 180:
            img = np.rot90(img, 2).copy()
        elif angle == 270:
            img = np.rot90(img, 3).copy()
        return img

    def __getitem__(self, idx):
        """
        Returns:
            frame_tensor: Frame tensor [3, H, W] in [0, 1].
            filename: Placeholder string for compatibility.
        """
        frame = self.frames[idx]
        h, w = frame.shape[:2]

        # Random crop
        if h >= self.image_size and w >= self.image_size:
            top  = random.randint(0, h - self.image_size)
            left = random.randint(0, w - self.image_size)
            frame = frame[top:top + self.image_size, left:left + self.image_size]
        else:
            from PIL import Image as _PIL
            frame = np.array(
                _PIL.fromarray(frame).resize((self.image_size, self.image_size), _PIL.LANCZOS)
            )

        if self.augment:
            frame = self._augment(frame)

        return TF.to_tensor(Image.fromarray(frame)), f"frame_{idx}"


def get_video_dataloader(video_dir, config):
    """
    Create a DataLoader from a directory of video files.

    Args:
        video_dir: Path to directory with video files.
        config: Configuration module.

    Returns:
        DataLoader for video frames.
    """
    dataset = VideoFrameDataset(
        video_dir=video_dir,
        image_size=config.IMAGE_SIZE,
        frame_interval=getattr(config, 'VIDEO_FRAME_INTERVAL', 10),
        augment=True,
    )
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )


def get_dataloaders(config):
    """
    Create training and testing data loaders.

    Args:
        config: Configuration module with dataset paths and parameters.

    Returns:
        train_loader: DataLoader for training.
        test_loader: DataLoader for testing.
    """
    print("Loading datasets...")
    train_dataset = LOLDataset(
        low_dir=config.TRAIN_LOW_DIR,
        high_dir=config.TRAIN_HIGH_DIR,
        image_size=config.IMAGE_SIZE,
        augment=True,
    )

    test_dataset = LOLDataset(
        low_dir=config.TEST_LOW_DIR,
        high_dir=config.TEST_HIGH_DIR,
        image_size=config.IMAGE_SIZE,
        augment=False,
    )

    import sys
    use_workers = config.NUM_WORKERS if sys.platform != 'win32' else 0
    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=use_workers,
        pin_memory=use_pin,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=use_workers,
        pin_memory=use_pin,
    )

    return train_loader, test_loader
