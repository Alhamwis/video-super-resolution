#!/usr/bin/env python3
"""
Compare Video Enhancement Methods
--------------------------------
This script creates a side-by-side comparison of different video enhancement methods.
"""

import os
import sys
import argparse
import subprocess
import tempfile
import cv2
import numpy as np
from tqdm import tqdm

def extract_frame(video_path, frame_number):
    """Extract a specific frame from a video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not extract frame {frame_number} from {video_path}")
    return frame

def get_video_info(video_path):
    """Get video information."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    return width, height, fps, total_frames

def create_comparison_video(original_video, enhanced_videos, output_path, sample_frames=None):
    """Create a side-by-side comparison video."""
    # Get video information
    _, _, fps, total_frames = get_video_info(original_video)
    
    # Determine which frames to sample
    if sample_frames is None:
        # Sample 10 frames evenly distributed
        sample_frames = [int(i * total_frames / 10) for i in range(10)]
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    
    # Process each sample frame
    for i, frame_num in enumerate(tqdm(sample_frames, desc="Creating comparison frames")):
        # Extract frames from each video
        original_frame = extract_frame(original_video, frame_num)
        enhanced_frames = [extract_frame(video, frame_num) for video in enhanced_videos]
        
        # Get dimensions
        oh, ow = original_frame.shape[:2]
        
        # Create a canvas for the comparison
        # Original + all enhanced videos in a row
        canvas_width = ow * (1 + len(enhanced_frames))
        canvas_height = oh
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Add original frame
        canvas[0:oh, 0:ow] = original_frame
        
        # Add enhanced frames
        for j, frame in enumerate(enhanced_frames):
            h, w = frame.shape[:2]
            # Resize if necessary
            if h != oh or w != ow:
                frame = cv2.resize(frame, (ow, oh))
            canvas[0:oh, (j+1)*ow:(j+2)*ow] = frame
        
        # Add labels
        cv2.putText(canvas, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for j, video_path in enumerate(enhanced_videos):
            label = os.path.basename(video_path).split('.')[0]
            cv2.putText(canvas, label, ((j+1)*ow + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save frame
        output_frame_path = os.path.join(temp_dir, f"comparison_{i:04d}.png")
        cv2.imwrite(output_frame_path, canvas)
    
    # Create video from frames
    frame_pattern = os.path.join(temp_dir, "comparison_%04d.png")
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', frame_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        output_path
    ]
    
    subprocess.run(cmd, check=True)
    print(f"Comparison video saved to: {output_path}")
    
    # Clean up temporary files
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

def main():
    parser = argparse.ArgumentParser(description="Create a side-by-side comparison of video enhancement methods")
    parser.add_argument('-o', '--original', required=True, help="Path to the original video")
    parser.add_argument('-e', '--enhanced', required=True, nargs='+', help="Paths to enhanced videos")
    parser.add_argument('-c', '--comparison', default="comparison.mp4", help="Output comparison video path")
    
    args = parser.parse_args()
    
    create_comparison_video(args.original, args.enhanced, args.comparison)

if __name__ == "__main__":
    main() 