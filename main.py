"""
Main Entry Point - Night Surveillance Enhancement System

Deep Learning-Based Image and Video Enhancement for
Night Surveillance and Multi-Exposure Fusion.

This script provides a unified command-line interface for all
system operations: download, train, test, enhance, and deploy.

Usage:
    python main.py download          # Download LOL dataset
    python main.py train             # Train/fine-tune the model
    python main.py test              # Evaluate on test set
    python main.py enhance_image     # Enhance a single image
    python main.py enhance_video     # Enhance a video
    python main.py fusion            # Multi-exposure fusion
    python main.py app               # Launch web interface
    python main.py demo              # Run complete demo pipeline
"""

import os
import sys
import argparse


def print_banner():
    print()
    print("=" * 65)
    print("  Deep Learning-Based Image and Video Enhancement")
    print("  for Night Surveillance and Multi-Exposure Fusion")
    print("=" * 65)
    print(f"  Python  : {sys.version.split()[0]}")
    try:
        import torch
        print(f"  PyTorch : {torch.__version__}")
        print(f"  CUDA    : {'Available (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'Not available (using CPU)'}")
    except ImportError:
        print("  PyTorch : NOT INSTALLED")
    print("=" * 65)
    print()


def cmd_download(args):
    """Download the LOL dataset."""
    from download_dataset import download_lol_dataset
    download_lol_dataset()


def cmd_train(args):
    """Train the enhancement model."""
    sys.argv = ['train.py']
    if args.epochs:
        sys.argv += ['--epochs', str(args.epochs)]
    if args.lr:
        sys.argv += ['--lr', str(args.lr)]
    if args.batch_size:
        sys.argv += ['--batch_size', str(args.batch_size)]
    if args.resume:
        sys.argv += ['--resume', args.resume]

    from train import main as train_main
    train_main()


def cmd_train_video(args):
    """Fine-tune model on video frames."""
    import runpy
    runpy.run_path('train_video.py', run_name='__main__')


def cmd_test(args):
    """Evaluate the model."""
    sys.argv = ['test.py']
    if args.model:
        sys.argv += ['--model', args.model]
    sys.argv += ['--save_visuals']

    from test import main as test_main
    test_main()


def cmd_enhance_image(args):
    """Enhance images."""
    if not args.input:
        print("ERROR: --input is required. Provide an image path or directory.")
        return

    sys.argv = ['enhance_image.py', '--input', args.input]
    if args.output:
        sys.argv += ['--output', args.output]
    if args.model:
        sys.argv += ['--model', args.model]

    from enhance_image import main as enhance_main
    enhance_main()


def cmd_enhance_video(args):
    """Enhance video."""
    if not args.input:
        print("ERROR: --input is required. Provide a video path.")
        return

    sys.argv = ['enhance_video.py', '--input', args.input]
    if args.output:
        sys.argv += ['--output', args.output]
    if args.model:
        sys.argv += ['--model', args.model]

    from enhance_video import main as enhance_main
    enhance_main()


def cmd_fusion(args):
    """Multi-exposure fusion."""
    if not args.input:
        print("ERROR: --input is required. Provide an image path.")
        return

    sys.argv = ['multi_exposure_fusion.py', '--input', args.input]
    if args.output:
        sys.argv += ['--output', args.output]
    if args.model:
        sys.argv += ['--model', args.model]

    from multi_exposure_fusion import main as fusion_main
    fusion_main()


def cmd_app(args):
    """Launch Streamlit web application."""
    import subprocess
    port = args.port or 8501
    print(f"  Launching Streamlit app on port {port}...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py",
                    "--server.port", str(port)])


def cmd_demo(args):
    """Run a complete demonstration pipeline."""
    import config

    print_banner()
    print("  Running Complete Demo Pipeline")
    print("=" * 65)

    # Step 1: Check dataset
    print("\n  [1/4] Checking dataset...")
    train_low = config.TRAIN_LOW_DIR
    if os.path.exists(train_low) and len(os.listdir(train_low)) > 0:
        print(f"    Dataset found: {len(os.listdir(train_low))} training images")
    else:
        print("    Dataset not found. Run: python main.py download")
        return

    # Step 2: Check model
    print("\n  [2/4] Checking model...")
    model_path = args.model or 'pretrained/best_model.pth'
    if os.path.exists(model_path):
        print(f"    Model found: {model_path}")
    else:
        print("    Model not found. Run: python main.py train")
        return

    # Step 3: Quick evaluation
    print("\n  [3/4] Running evaluation on test set...")
    sys.argv = ['test.py', '--model', model_path, '--save_visuals']
    from test import main as test_main
    test_main()

    # Step 4: Enhance sample images
    print("\n  [4/4] Enhancing test images...")
    sys.argv = ['enhance_image.py', '--input', config.TEST_LOW_DIR,
                '--output', 'results/demo', '--model', model_path]
    from enhance_image import main as enhance_main
    enhance_main()

    print("\n" + "=" * 65)
    print("  Demo Complete! Check the results/ directory for outputs.")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description='Night Surveillance Enhancement System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  download        Download the LOL dataset from Google Drive
  train           Train/fine-tune the Zero-DCE model
  test            Evaluate model on test set with metrics
  enhance_image   Enhance low-light image(s)
  enhance_video   Enhance surveillance video
  fusion          Apply multi-exposure fusion
  app             Launch Streamlit web interface
  demo            Run complete demo pipeline

Examples:
  python main.py download
  python main.py train --epochs 200 --lr 1e-4
  python main.py test --model pretrained/best_model.pth
  python main.py enhance_image --input dark_photo.jpg
  python main.py enhance_video --input surveillance.mp4
  python main.py fusion --input mixed_exposure.jpg
  python main.py app --port 7860
        """
    )

    parser.add_argument('command', choices=[
        'download', 'train', 'train_video', 'test', 'enhance_image',
        'enhance_video', 'fusion', 'app', 'demo'
    ], help='Command to execute')

    parser.add_argument('--input', '-i', help='Input file/directory')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--model', '-m', default='pretrained/best_model.pth', help='Model path')
    parser.add_argument('--epochs', type=int, help='Training epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--resume', help='Resume from checkpoint')
    parser.add_argument('--port', type=int, default=8501, help='Web app port')

    args = parser.parse_args()

    print_banner()

    commands = {
        'download':    cmd_download,
        'train':       cmd_train,
        'train_video': cmd_train_video,
        'test':        cmd_test,
        'enhance_image':  cmd_enhance_image,
        'enhance_video':  cmd_enhance_video,
        'fusion':      cmd_fusion,
        'app':         cmd_app,
        'demo':        cmd_demo,
    }

    commands[args.command](args)


if __name__ == '__main__':
    main()
