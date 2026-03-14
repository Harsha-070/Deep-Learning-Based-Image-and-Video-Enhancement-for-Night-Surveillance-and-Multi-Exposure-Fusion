"""
Night Surveillance Object Detection using YOLOv8x + Zero-DCE Enhancement.

Pipeline:
    1. Enhance low-light image with Zero-DCE (so YOLO can actually see objects)
    2. Run YOLOv8x detection on the brightened result
    3. Return annotated image with bounding boxes, labels, and confidence scores

Usage:
    python detect.py --input dark_image.jpg
    python detect.py --input dark_video.mp4
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

import config
from models import EnhanceNet

# Surveillance-relevant COCO class IDs to highlight
SURVEILLANCE_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    15: 'cat',
    16: 'dog',
    24: 'backpack',
    26: 'handbag',
    28: 'suitcase',
    39: 'bottle',
    56: 'chair',
    57: 'couch',
    67: 'cell phone',
    73: 'laptop',
    77: 'knife',
}

# Color map: person=red, vehicles=blue, others=green
CLASS_COLORS = {
    0:  (0, 0, 255),       # person — red (BGR)
    1:  (255, 100, 0),     # bicycle — blue
    2:  (255, 100, 0),     # car
    3:  (255, 100, 0),     # motorcycle
    5:  (255, 100, 0),     # bus
    7:  (255, 100, 0),     # truck
}
DEFAULT_COLOR = (0, 200, 0)  # green for everything else


def load_enhancer(model_path):
    """Load Zero-DCE enhancement model."""
    device = config.DEVICE
    model = EnhanceNet(
        in_channels=config.INPUT_CHANNELS,
        hidden_channels=config.HIDDEN_CHANNELS,
        num_curves=config.NUM_CURVES,
    ).to(device)

    if model_path and os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    model.eval()
    return model, device


@torch.no_grad()
def enhance_frame(model, device, frame_bgr):
    """Enhance a single BGR frame with Zero-DCE."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    pad_w = (4 - w % 4) % 4
    pad_h = (4 - h % 4) % 4
    if pad_w or pad_h:
        frame_rgb = np.pad(frame_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    tensor = TF.to_tensor(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)
    out, _, _ = model(tensor)

    enhanced = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
    enhanced = enhanced[:h, :w, :]

    return cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)


def draw_detections(image, results, conf_threshold=0.25):
    """
    Draw YOLO detection boxes on image.

    Returns annotated image and list of detection dicts.
    """
    annotated = image.copy()
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            cls_id = int(box.cls[0])
            cls_name = result.names.get(cls_id, f'class_{cls_id}')
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label background
            label = f'{cls_name} {conf:.0%}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            detections.append({
                'class': cls_name,
                'confidence': round(conf, 3),
                'bbox': [x1, y1, x2, y2],
            })

    return annotated, detections


class NightDetector:
    """
    Night surveillance detector: Zero-DCE enhancement + YOLOv8x detection.
    """

    def __init__(self, model_path='pretrained/best_model.pth',
                 yolo_model='yolov8x.pt', conf=0.25):
        """
        Args:
            model_path: Path to Zero-DCE weights.
            yolo_model: YOLO model name ('yolov8n.pt', 'yolov8x.pt', etc.)
                        Auto-downloaded on first use.
            conf: Confidence threshold for detections.
        """
        from ultralytics import YOLO

        self.conf = conf
        self.enhance_model, self.device = load_enhancer(model_path)
        self.yolo = YOLO(yolo_model)
        print(f"  YOLO model: {yolo_model}")
        print(f"  Enhancement: Zero-DCE loaded")

    def detect(self, image_bgr, enhance_first=True):
        """
        Detect objects in a BGR image.

        Args:
            image_bgr: Input image as numpy array [H, W, 3] BGR.
            enhance_first: Enhance with Zero-DCE before detection.

        Returns:
            enhanced: Enhanced BGR image (or original if enhance_first=False).
            annotated: Annotated image with bounding boxes.
            detections: List of detection dicts.
        """
        if enhance_first:
            enhanced = enhance_frame(self.enhance_model, self.device, image_bgr)
        else:
            enhanced = image_bgr.copy()

        results = self.yolo(enhanced, conf=self.conf, verbose=False)
        annotated, detections = draw_detections(enhanced, results, self.conf)

        return enhanced, annotated, detections

    def detect_video(self, input_path, output_path, enhance_first=True,
                     show_progress=True):
        """
        Process a video file with enhancement + detection.

        Returns:
            stats: Processing statistics.
        """
        import time
        from tqdm import tqdm
        from enhance_video import _reencode_to_h264

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        codec = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, codec, fps, (width, height))

        frame_times = []
        all_detections = []
        pbar = tqdm(total=total, desc='  Detecting', disable=not show_progress)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            _, annotated, dets = self.detect(frame, enhance_first=enhance_first)
            frame_times.append(time.time() - t0)
            all_detections.extend(dets)
            writer.write(annotated)
            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()

        # Re-encode to H.264
        h264_path = output_path.replace('.mp4', '_h264.mp4')
        if _reencode_to_h264(output_path, h264_path):
            os.replace(h264_path, output_path)

        return {
            'total_frames': len(frame_times),
            'avg_fps': 1.0 / np.mean(frame_times) if frame_times else 0,
            'total_time': sum(frame_times),
            'total_detections': len(all_detections),
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Night surveillance detection')
    parser.add_argument('--input', '-i', required=True, help='Image or video path')
    parser.add_argument('--output', '-o', default='results/detection', help='Output directory')
    parser.add_argument('--model', '-m', default='pretrained/best_model.pth')
    parser.add_argument('--yolo', default='yolov8x.pt',
                        help='YOLO model: yolov8n/s/m/l/x.pt')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--no_enhance', action='store_true', help='Skip Zero-DCE enhancement')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  Night Surveillance — Enhance + Detect")
    print("=" * 60)

    detector = NightDetector(
        model_path=args.model,
        yolo_model=args.yolo,
        conf=args.conf,
    )

    ext = os.path.splitext(args.input)[1].lower()
    name = os.path.splitext(os.path.basename(args.input))[0]

    if ext in ('.mp4', '.avi', '.mov', '.mkv'):
        output_path = os.path.join(args.output, f'{name}_detected.mp4')
        stats = detector.detect_video(args.input, output_path,
                                      enhance_first=not args.no_enhance)
        print(f"\n  Frames     : {stats['total_frames']}")
        print(f"  FPS        : {stats['avg_fps']:.1f}")
        print(f"  Detections : {stats['total_detections']}")
        print(f"  Output     : {output_path}")
    else:
        image = cv2.imread(args.input)
        if image is None:
            print(f"  ERROR: Cannot read {args.input}")
            return

        enhanced, annotated, detections = detector.detect(
            image, enhance_first=not args.no_enhance
        )

        cv2.imwrite(os.path.join(args.output, f'{name}_enhanced.jpg'), enhanced)
        cv2.imwrite(os.path.join(args.output, f'{name}_detected.jpg'), annotated)

        print(f"\n  Detected {len(detections)} objects:")
        for d in detections:
            print(f"    {d['class']:15s} {d['confidence']:.0%}  {d['bbox']}")

        print(f"\n  Results saved to {args.output}/")

    print("=" * 60)


if __name__ == '__main__':
    main()
