"""AQCF-Net Training Pipeline Runner"""

import os
import sys
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from train import run_training_pipeline
    from config import OUTPUT_DIR
    import torch
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required packages are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)

def setup_environment():
    print(" A-QCF-Net: Adaptive Quaternion Cross-Fusion Network")
    print("=" * 60)
    
    # Check PyTorch installation
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    print()

def check_prerequisites(data_dir, dataset_json):
    issues = []
    data_path = Path(data_dir)
    if not data_path.exists():
        issues.append(f"Dataset directory not found: {data_dir}")
    json_path = data_path / dataset_json
    if not json_path.exists():
        issues.append(f"Dataset JSON not found: {json_path}")

    output_path = Path(OUTPUT_DIR)
    if not output_path.exists():
        print(f"Creating output directory: {OUTPUT_DIR}")
        output_path.mkdir(parents=True, exist_ok=True)
    
    if issues:
        print(" Prerequisites check failed:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(" All prerequisites met!")
        return True

def run_training_pipeline_simple(args):
    try:
        if not check_prerequisites(args.data_dir, args.dataset_json):
            return False
        success = run_training_pipeline(
            data_dir=args.data_dir,
            dataset_json=args.dataset_json,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )
        
        if success:
            print("\n Training completed successfully!")
            return True
        else:
            print("\n Training failed!")
            return False
            
    except KeyboardInterrupt:
        print("\n Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n Training failed with error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="A-QCF-Net Training Pipeline")

    parser.add_argument("--data-dir", default=".", 
                       help="Directory containing the dataset (default: current directory)")
    parser.add_argument("--dataset-json", default="dataset_0.json",
                       help="Dataset JSON file name (default: dataset_0.json)")

    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size (default: 1)")
    parser.add_argument("--device", default="auto",
                       help="Device to use: 'auto', 'cuda', 'cpu' (default: auto)")
    
    parser.add_argument("--check-env", action="store_true",
                       help="Check environment and exit")
    
    args = parser.parse_args()
    
    if args.check_env:
        setup_environment()
        check_prerequisites(args.data_dir, args.dataset_json)
        return
    
    setup_environment()
    
    success = run_training_pipeline_simple(args)
    
    if success:
        print("\n Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()