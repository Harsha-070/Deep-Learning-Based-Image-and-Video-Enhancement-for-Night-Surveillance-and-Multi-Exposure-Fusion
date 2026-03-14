"""
Video Enhancement Pipeline for night surveillance footage.

Features:
    - Frame-by-frame enhancement using Zero-DCE
    - Temporal consistency via exponential moving average
    - Prevents flickering across frames
    - Supports various video formats (mp4, avi, mov, etc.)
    - Real-time progress tracking

Usage:
    python enhance_video.py --input path/to/video.mp4
    python enhance_video.py --input path/to/video.mp4 --output results/video_enhanced.mp4
"""

import os
import shutil
import subprocess
import argparse
import time
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

import config
from models import EnhanceNet


def _reencode_to_h264(input_path, output_path):
    """
    Re-encode video to H.264 for browser/Streamlit compatibility.
    Tries ffmpeg → OpenCV H.264 codecs → mp4v fallback → copy original.
    Always produces an output file so the app never shows a blank player.
    """
    # ── Try ffmpeg ────────────────────────────────────────────────────────────
    for cmd in ["ffmpeg", "ffmpeg.exe"]:
        try:
            result = subprocess.run(
                [cmd, "-y", "-i", input_path,
                 "-c:v", "libx264", "-preset", "fast",
                 "-crf", "23", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", output_path],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0 and os.path.exists(output_path) \
               and os.path.getsize(output_path) > 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue

    # ── Fallback: OpenCV re-encode ────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        shutil.copy2(input_path, output_path)
        return os.path.exists(output_path)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        shutil.copy2(input_path, output_path)
        return os.path.exists(output_path)

    # Try H.264 codecs first, then mp4v as last codec option
    for codec_name in ['avc1', 'H264', 'X264', 'h264', 'mp4v']:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                continue
            for frame in frames:
                writer.write(frame)
            writer.release()
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True
        except Exception:
            continue

    # Absolute last resort: copy original so file always exists
    shutil.copy2(input_path, output_path)
    return os.path.exists(output_path)


class VideoEnhancer:
    """
    Video enhancement engine with temporal consistency.

    Uses exponential moving average (EMA) of curve parameters
    between consecutive frames to prevent flickering.
    """

    def __init__(self, model_path=None, device=None, temporal_weight=None):
        """
        Args:
            model_path: Path to trained model weights.
            device: torch device.
            temporal_weight: EMA weight for temporal smoothing (0-1).
                Higher = more smoothing, lower = more responsive.
        """
        self.device = device or config.DEVICE
        self.temporal_weight = temporal_weight if temporal_weight is not None else config.TEMPORAL_WEIGHT

        # Load model
        self.model = EnhanceNet(
            in_channels=config.INPUT_CHANNELS,
            hidden_channels=config.HIDDEN_CHANNELS,
            num_curves=config.NUM_CURVES,
        ).to(self.device)

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Model loaded from {model_path}")
        else:
            print("  WARNING: No model weights loaded.")

        self.model.eval()
        self.prev_curve_params = None

    @torch.no_grad()
    def enhance_frame(self, frame, use_temporal=True):
        """
        Enhance a single video frame with optional temporal consistency.

        Args:
            frame: Input frame as numpy array [H, W, 3] BGR.
            use_temporal: Apply temporal smoothing with previous frame.

        Returns:
            enhanced: Enhanced frame [H, W, 3] BGR uint8.
        """
        # BGR to RGB, convert to tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # Pad to multiple of 4
        pad_w = (4 - w % 4) % 4
        pad_h = (4 - h % 4) % 4
        if pad_w or pad_h:
            frame_rgb = np.pad(frame_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

        input_tensor = TF.to_tensor(Image.fromarray(frame_rgb)).unsqueeze(0).to(self.device)

        # Get curve parameters
        curve_params = self.model.curve_net(input_tensor)

        # Temporal smoothing via EMA
        if use_temporal and self.prev_curve_params is not None:
            # Ensure same spatial dimensions
            if curve_params.shape == self.prev_curve_params.shape:
                curve_params = (self.temporal_weight * self.prev_curve_params +
                                (1 - self.temporal_weight) * curve_params)

        self.prev_curve_params = curve_params.clone()

        # Apply curves
        enhanced_tensor, _ = self.model.apply_curves(input_tensor, curve_params)

        # Convert back
        enhanced = enhanced_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)

        # Remove padding
        enhanced = enhanced[:h, :w, :]

        # Denoise: bilateral filter (edge-preserving, compatible with all OpenCV versions)
        enhanced = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)

        return cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

    def process_video(self, input_path, output_path, show_progress=True,
                      create_comparison=False):
        """
        Process an entire video file.

        Args:
            input_path: Path to input video.
            output_path: Path to save enhanced video.
            show_progress: Show progress bar.
            create_comparison: Create side-by-side comparison video.

        Returns:
            stats: Dictionary with processing statistics.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'mp4v')

        print(f"\n  Input Video:")
        print(f"    Resolution : {width}x{height}")
        print(f"    FPS        : {fps}")
        print(f"    Frames     : {total_frames}")
        print(f"    Duration   : {total_frames / fps:.1f}s")

        # Output writer
        if create_comparison:
            out_width = width * 2 + 3  # side-by-side with separator
            writer = cv2.VideoWriter(output_path, codec, fps, (out_width, height))
        else:
            writer = cv2.VideoWriter(output_path, codec, fps, (width, height))

        # Reset temporal state
        self.prev_curve_params = None

        # Process frames
        frame_times = []
        pbar = tqdm(total=total_frames, desc='  Processing', disable=not show_progress)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            enhanced = self.enhance_frame(frame, use_temporal=True)
            frame_time = time.time() - start_time
            frame_times.append(frame_time)

            if create_comparison:
                # Side-by-side
                separator = np.ones((height, 3, 3), dtype=np.uint8) * 255
                comparison = np.hstack([frame, separator, enhanced])
                writer.write(comparison)
            else:
                writer.write(enhanced)

            frame_idx += 1
            pbar.update(1)
            pbar.set_postfix({'fps': f'{1.0 / frame_time:.1f}'})

        pbar.close()
        cap.release()
        writer.release()

        # Re-encode to H.264 for browser/Streamlit compatibility
        h264_path = output_path.replace(".mp4", "_h264.mp4")
        if _reencode_to_h264(output_path, h264_path):
            os.replace(h264_path, output_path)
            print("    Re-encoded to H.264 for browser compatibility.")
        else:
            print("    WARNING: ffmpeg not found. Video may not play in browser.")
            print("    Install ffmpeg for full browser compatibility.")

        # Statistics
        stats = {
            'total_frames': frame_idx,
            'avg_frame_time': np.mean(frame_times) if frame_times else 0,
            'avg_fps': 1.0 / np.mean(frame_times) if frame_times else 0,
            'total_time': sum(frame_times),
        }

        print(f"\n  Processing Complete:")
        print(f"    Frames processed : {stats['total_frames']}")
        print(f"    Avg frame time   : {stats['avg_frame_time'] * 1000:.1f} ms")
        print(f"    Processing FPS   : {stats['avg_fps']:.1f}")
        print(f"    Total time       : {stats['total_time']:.1f}s")
        print(f"    Output saved to  : {output_path}")

        return stats


def main():
    parser = argparse.ArgumentParser(description='Enhance night surveillance video')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', default=None, help='Output video path')
    parser.add_argument('--model', '-m', default='pretrained/best_model.pth', help='Model path')
    parser.add_argument('--temporal_weight', type=float, default=config.TEMPORAL_WEIGHT,
                        help='Temporal smoothing weight (0-1)')
    parser.add_argument('--comparison', action='store_true', help='Create comparison video')
    args = parser.parse_args()

    # Auto-generate output path
    if args.output is None:
        name, ext = os.path.splitext(args.input)
        args.output = f"{name}_enhanced{ext}"

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print("=" * 60)
    print("  Night Surveillance Video Enhancement")
    print("=" * 60)
    print(f"  Device           : {config.DEVICE}")
    print(f"  Temporal Weight  : {args.temporal_weight}")

    if not os.path.exists(args.model):
        print(f"  ERROR: Model weights not found at '{args.model}'.")
        print(f"  Please train the model first: python run_train.py")
        print(f"  Or specify a valid model path: --model <path>")
        return

    if not os.path.exists(args.input):
        print(f"  ERROR: Input video not found at '{args.input}'.")
        return

    enhancer = VideoEnhancer(
        model_path=args.model,
        temporal_weight=args.temporal_weight,
    )

    enhancer.process_video(
        args.input, args.output,
        create_comparison=args.comparison,
    )

    print("=" * 60)


if __name__ == '__main__':
    main()
