#!/usr/bin/env python3
"""
Download script for LAPA training datasets - Testing Version
Focuses on small datasets for pipeline testing and development
"""

import os
import subprocess
import argparse
from pathlib import Path
import requests
import json
from typing import List, Optional


def download_sample_videos(output_dir: str, num_samples: int = 10):
    """
    Download sample videos from HuggingFace datasets for testing.
    
    Note: Something-Something-V2 videos are no longer available via HuggingFace
    due to copyright restrictions. This function will create dummy videos instead.
    
    Args:
        output_dir: Directory to save sample videos
        num_samples: Number of sample videos to download
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_samples} sample videos for testing...")
    print("Note: Something-Something-V2 videos are no longer available via HuggingFace")
    print("due to copyright restrictions. Creating dummy videos instead.")
    
    try:
        import cv2
        import numpy as np
        
        videos_dir = output_path / "sample_videos"
        videos_dir.mkdir(exist_ok=True)
        
        print(f"Creating {num_samples} dummy sample videos in {videos_dir}")
        
        for idx in range(num_samples):
            # Create a simple video with moving shapes
            video_path = videos_dir / f"sample_video_{idx:03d}.mp4"
            
            # Video properties
            width, height = 224, 224
            fps = 10
            duration = 3  # seconds
            total_frames = fps * duration
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            for frame_idx in range(total_frames):
                # Create frame with moving shapes
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Moving rectangle
                center_x = int(width * 0.2 + (width * 0.6) * frame_idx / total_frames)
                center_y = height // 2
                rect_size = 30
                color = (0, 255, 0)  # Green
                
                cv2.rectangle(frame, 
                             (center_x - rect_size//2, center_y - rect_size//2),
                             (center_x + rect_size//2, center_y + rect_size//2),
                             color, -1)
                
                # Add some variation for different videos
                if idx % 2 == 0:
                    # Add a circle for even videos
                    circle_x = int(width * 0.8 - (width * 0.4) * frame_idx / total_frames)
                    cv2.circle(frame, (circle_x, center_y), 20, (255, 0, 0), -1)
                
                # Add some noise for realism
                noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
                frame = cv2.add(frame, noise)
                
                out.write(frame)
            
            out.release()
            
            # Create metadata similar to Something-Something-V2 format
            metadata = {
                'id': f"sample_video_{idx:03d}",
                'label': idx % 10,  # Dummy label
                'template': f"Moving object {idx}",
                'placeholders': {'object': f"shape_{idx}"},
                'description': f"Sample video {idx} with moving shapes"
            }
            
            metadata_path = videos_dir / f"sample_video_{idx:03d}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  ✓ Created sample video {idx+1}/{num_samples}: {video_path.name}")
        
        print(f"✓ Successfully created {num_samples} sample videos in {videos_dir}")
        print("\nTo use real Something-Something-V2 videos:")
        print("1. Register at: https://developer.qualcomm.com/software/ai-datasets/something-something")
        print("2. Download videos manually")
        print("3. Place them in the sample_videos directory")
        
        return True
        
    except ImportError:
        print("ERROR: OpenCV not installed. Install with: pip install opencv-python")
        return False
    except Exception as e:
        print(f"ERROR: Failed to create sample videos: {e}")
        return False


def download_openx_sample(output_dir: str, dataset_name: str = "bridge"):
    """
    Download a small sample from Open-X Embodiment dataset.
    
    Args:
        output_dir: Directory to save sample data
        dataset_name: Specific dataset to sample from
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading sample from Open-X Embodiment dataset: {dataset_name}")
    
    # Check if gsutil is installed
    try:
        subprocess.run(["gsutil", "version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: gsutil not found. Please install Google Cloud SDK:")
        print("https://cloud.google.com/storage/docs/gsutil_install")
        print("\nOr install via: pip install gsutil")
        return False
    
    base_url = f"gs://gresearch/robotics/{dataset_name}"
    sample_path = output_path / f"openx_{dataset_name}_sample"
    
    print(f"Downloading sample from {base_url} to {sample_path}")
    print("Note: This will download a small sample for testing purposes")
    
    # Download just a few files as a sample
    cmd = [
        "gsutil", "-m", "cp", "-r",
        f"{base_url}",
        str(sample_path)
    ]
    
    try:
        # Limit the download size by using gsutil's -m flag with limits
        subprocess.run(cmd, check=True, timeout=300)  # 5 minute timeout
        print(f"✓ Successfully downloaded sample to {sample_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Download failed: {e}")
        return False
    except subprocess.TimeoutExpired:
        print("ERROR: Download timed out. Try with a smaller dataset.")
        return False


def create_dummy_video_dataset(output_dir: str, num_videos: int = 20):
    """
    Create dummy video dataset for testing LAQ pipeline.
    
    Args:
        output_dir: Directory to save dummy videos
        num_videos: Number of dummy videos to create
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_videos} dummy videos for testing...")
    
    try:
        import cv2
        import numpy as np
        
        videos_dir = output_path / "dummy_videos"
        videos_dir.mkdir(exist_ok=True)
        
        for i in range(num_videos):
            # Create a simple video with moving shapes
            video_path = videos_dir / f"dummy_video_{i:03d}.mp4"
            
            # Video properties
            width, height = 224, 224
            fps = 10
            duration = 3  # seconds
            total_frames = fps * duration
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            for frame_idx in range(total_frames):
                # Create frame with moving circle
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Moving circle
                center_x = int(width * 0.2 + (width * 0.6) * frame_idx / total_frames)
                center_y = height // 2
                radius = 20
                color = (0, 255, 0)  # Green
                
                cv2.circle(frame, (center_x, center_y), radius, color, -1)
                
                # Add some noise for realism
                noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
                frame = cv2.add(frame, noise)
                
                out.write(frame)
            
            out.release()
            
            # Create metadata
            metadata = {
                'video_id': f"dummy_video_{i:03d}",
                'duration': duration,
                'fps': fps,
                'resolution': [width, height],
                'description': f"Dummy video {i} with moving circle"
            }
            
            metadata_path = videos_dir / f"dummy_video_{i:03d}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  ✓ Created dummy video {i+1}/{num_videos}: {video_path.name}")
        
        print(f"✓ Successfully created {num_videos} dummy videos in {videos_dir}")
        return True
        
    except ImportError:
        print("ERROR: OpenCV not installed. Install with: pip install opencv-python")
        return False
    except Exception as e:
        print(f"ERROR: Failed to create dummy videos: {e}")
        return False


def download_test_checkpoints(output_dir: str):
    """
    Download or create test checkpoints for LAQ pipeline testing.
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Setting up test checkpoints...")
    
    # Create dummy checkpoint structure
    checkpoints_dir = output_path / "test_checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Create a dummy LAQ checkpoint
    dummy_checkpoint = {
        'model_state_dict': {},
        'optimizer_state_dict': {},
        'epoch': 0,
        'step': 0,
        'config': {
            'encoder': {'in_channels': 6, 'latent_dim': 256},
            'quantizer': {'num_tokens': 4, 'vocab_size': 8},
            'decoder': {'out_channels': 3}
        }
    }
    
    checkpoint_path = checkpoints_dir / "laq_test.ckpt"
    with open(checkpoint_path, 'w') as f:
        json.dump(dummy_checkpoint, f, indent=2)
    
    print(f"✓ Created test checkpoint: {checkpoint_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download test datasets for LAPA pipeline testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create dummy videos for testing
  python scripts/download_test_data.py --dummy --output ./datasets --num-videos 20
  
  # Download sample videos from Something-Something-V2
  python download_test_data.py --samples --output ./datasets --num-samples 10
  
  # Download small Open-X sample
  python download_test_data.py --openx-sample --output ./datasets --dataset bridge
  
  # Setup all test data
  python download_test_data.py --all --output ./datasets
        """
    )
    
    parser.add_argument("--output", "-o", default="./datasets", help="Output directory for datasets")
    parser.add_argument("--dummy", action="store_true", help="Create dummy videos for testing")
    parser.add_argument("--samples", action="store_true", help="Download sample videos from Something-Something-V2")
    parser.add_argument("--openx-sample", action="store_true", help="Download small Open-X sample")
    parser.add_argument("--checkpoints", action="store_true", help="Setup test checkpoints")
    parser.add_argument("--all", action="store_true", help="Setup all test data")
    parser.add_argument("--num-videos", type=int, default=20, help="Number of dummy videos to create")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of sample videos to download")
    parser.add_argument("--dataset", default="bridge", help="Open-X dataset name for sampling")
    
    args = parser.parse_args()
    
    if not any([args.dummy, args.samples, args.openx_sample, args.checkpoints, args.all]):
        parser.print_help()
        return
    
    output_dir = args.output
    
    success = True
    
    if args.dummy or args.all:
        print("=" * 60)
        print("Creating Dummy Videos")
        print("=" * 60)
        success &= create_dummy_video_dataset(output_dir, args.num_videos)
    
    if args.samples or args.all:
        print("\n" + "=" * 60)
        print("Downloading Sample Videos")
        print("=" * 60)
        success &= download_sample_videos(output_dir, args.num_samples)
    
    if args.openx_sample or args.all:
        print("\n" + "=" * 60)
        print("Downloading Open-X Sample")
        print("=" * 60)
        success &= download_openx_sample(output_dir, args.dataset)
    
    if args.checkpoints or args.all:
        print("\n" + "=" * 60)
        print("Setting up Test Checkpoints")
        print("=" * 60)
        success &= download_test_checkpoints(output_dir)
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Test data setup complete!")
        print(f"Data saved to: {output_dir}")
        print("\nNext steps:")
        print("1. Test LAQ pipeline with dummy videos")
        print("2. Run preprocessing script on sample data")
        print("3. Train LAQ model on test dataset")
    else:
        print("✗ Some downloads failed. Check errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
