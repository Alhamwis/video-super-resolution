#!/usr/bin/env python3
"""
Video Super-Resolution Enhancer with Audio Preservation
-------------------------------------------------------
This script enhances the resolution of input videos using AI-based super-resolution
and preserves the audio from the original video.
"""

import argparse
import os
import subprocess
import time

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhance video resolution and preserve audio.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input video file path')
    parser.add_argument('-o', '--output', type=str, default=None, 
                        help='Output video file path (default: input_enhanced.mp4)')
    parser.add_argument('-m', '--model', type=str, default='espcn', 
                        choices=['espcn', 'fsrcnn', 'lapsrn'],
                        help='Super-resolution model to use (default: espcn)')
    parser.add_argument('-s', '--scale', type=int, default=2, choices=[2, 3, 4],
                        help='Scale factor for resolution enhancement (default: 2)')
    parser.add_argument('-q', '--quality', type=int, default=95, 
                        help='Output video quality (0-100, default: 95)')
    return parser.parse_args()

def has_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def has_audio(video_path):
    """Check if a video file has audio."""
    try:
        cmd = [
            'ffprobe', '-i', video_path, 
            '-show_streams', '-select_streams', 'a', 
            '-loglevel', 'error'
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return len(result.stdout) > 0
    except:
        return False

def enhance_video_with_audio(input_path, output_path, model='espcn', scale=2, quality=95):
    """Enhance video resolution and preserve audio using FFmpeg."""
    if not has_ffmpeg():
        print("Error: FFmpeg is required for this script.")
        return False
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False
    
    # Create temporary directory for processing
    temp_dir = "temp_processing"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Temporary files
    temp_video = os.path.join(temp_dir, "temp_video.mp4")
    temp_audio = os.path.join(temp_dir, "temp_audio.aac")
    temp_enhanced = os.path.join(temp_dir, "temp_enhanced.mp4")
    
    try:
        # Step 1: Extract audio if present
        has_audio_track = has_audio(input_path)
        if has_audio_track:
            print("Extracting audio from original video...")
            cmd_extract_audio = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-vn', '-acodec', 'copy',
                temp_audio
            ]
            subprocess.run(cmd_extract_audio, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Step 2: Run the enhance_video.py script to enhance the video
        print(f"Enhancing video using {model.upper()} model with scale factor {scale}...")
        cmd_enhance = [
            'python', 'enhance_video.py',
            '-i', input_path,
            '-o', temp_enhanced,
            '-m', model,
            '-s', str(scale),
            '-q', '100'  # Use maximum quality for intermediate file
        ]
        subprocess.run(cmd_enhance, check=True)
        
        # Step 3: Combine enhanced video with original audio
        print("Combining enhanced video with original audio...")
        if has_audio_track:
            cmd_combine = [
                'ffmpeg', '-y',
                '-i', temp_enhanced,
                '-i', temp_audio,
                '-c:v', 'libx264',
                '-crf', str((100-quality)/5),
                '-preset', 'medium',
                '-c:a', 'aac',
                '-b:a', '192k',
                output_path
            ]
        else:
            # If no audio, just optimize the video
            cmd_combine = [
                'ffmpeg', '-y',
                '-i', temp_enhanced,
                '-c:v', 'libx264',
                '-crf', str((100-quality)/5),
                '-preset', 'medium',
                output_path
            ]
        
        subprocess.run(cmd_combine, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("Video enhancement completed successfully!")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error during processing: {e}")
        return False
    
    finally:
        # Clean up temporary files
        print("Cleaning up temporary files...")
        for file in [temp_video, temp_audio, temp_enhanced]:
            if os.path.exists(file):
                os.remove(file)
        
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)

def main():
    """Main function to run the script."""
    args = parse_arguments()
    
    # Set default output path if not specified
    if args.output is None:
        filename, ext = os.path.splitext(args.input)
        args.output = f"{filename}_enhanced.mp4"
    
    print(f"Enhancing video: {args.input}")
    print(f"Output will be saved to: {args.output}")
    
    # Check for FFmpeg
    if not has_ffmpeg():
        print("Error: FFmpeg is required for this script.")
        print("Please install FFmpeg and ensure it's in your system PATH.")
        return
    
    # Enhance the video and preserve audio
    start_time = time.time()
    
    success = enhance_video_with_audio(
        args.input, 
        args.output, 
        model=args.model, 
        scale=args.scale,
        quality=args.quality
    )
    
    elapsed_time = time.time() - start_time
    
    if success:
        print(f"Video enhancement completed in {elapsed_time:.2f} seconds!")
        print(f"Enhanced video saved to: {args.output}")
    else:
        print("Video enhancement failed.")

if __name__ == "__main__":
    main() 