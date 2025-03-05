#!/usr/bin/env python3
"""
Video Super-Resolution Enhancer
-------------------------------
This script enhances the resolution of input videos using AI-based super-resolution.
"""

import argparse
import cv2
import numpy as np
import os
import time
import subprocess
from tqdm import tqdm

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhance video resolution using super-resolution.')
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
    parser.add_argument('--keep-audio', action='store_true', default=True,
                        help='Preserve audio from the original video (default: True)')
    return parser.parse_args()

def get_sr_model(model_name, scale):
    """Load the super-resolution model."""
    if model_name == 'espcn':
        model_path = f'models/ESPCN_x{scale}.pb'
        model = cv2.dnn_superres.DnnSuperResImpl_create()
        model.readModel(model_path)
        model.setModel('espcn', scale)
    elif model_name == 'fsrcnn':
        model_path = f'models/FSRCNN_x{scale}.pb'
        model = cv2.dnn_superres.DnnSuperResImpl_create()
        model.readModel(model_path)
        model.setModel('fsrcnn', scale)
    elif model_name == 'lapsrn':
        model_path = f'models/LapSRN_x{scale}.pb'
        model = cv2.dnn_superres.DnnSuperResImpl_create()
        model.readModel(model_path)
        model.setModel('lapsrn', scale)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def download_models(model_name, scale):
    """Download the required model files if they don't exist."""
    import urllib.request
    import os
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_file = f'models/{model_name.upper()}_x{scale}.pb'
    
    # Special case for LapSRN x3 which is not available
    if model_name == 'lapsrn' and scale == 3:
        print("Warning: LapSRN x3 model is not available. Please use scale 2 or 4 with LapSRN.")
        return False
    
    if not os.path.exists(model_file):
        print(f"Downloading {model_name} model (x{scale})...")
        
        # Model file URLs
        model_urls = {
            'espcn': {
                2: 'https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb',
                3: 'https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x3.pb',
                4: 'https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb'
            },
            'fsrcnn': {
                2: 'https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb',
                3: 'https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x3.pb',
                4: 'https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb'
            },
            'lapsrn': {
                2: 'https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x2.pb',
                4: 'https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x4.pb'
            }
        }
        
        try:
            url = model_urls[model_name][scale]
            urllib.request.urlretrieve(url, model_file)
            print(f"Successfully downloaded {model_name.upper()}_x{scale} model")
        except Exception as e:
            print(f"Failed to download model: {e}")
            return False
    
    return True

def has_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def enhance_video(input_path, output_path, model_name='espcn', scale=2, quality=95, keep_audio=True):
    """Enhance video resolution using the selected super-resolution model."""
    try:
        # Ensure the OpenCV DNN super resolution module is available
        cv2.dnn_superres
    except AttributeError:
        print("Error: OpenCV DNN super resolution module not available.")
        print("Please install OpenCV with extra modules (cv2.dnn_superres):")
        print("pip install opencv-contrib-python")
        return False
    
    # Download the model if it doesn't exist
    if not download_models(model_name, scale):
        print("Error downloading model.")
        print("You may need to manually download the models to the 'models' directory.")
        return False
    
    # Get the model
    try:
        model = get_sr_model(model_name, scale)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False
    
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_path}")
        return False
    
    # Get input video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate new dimensions
    new_width = width * scale
    new_height = height * scale
    
    print(f"Input resolution: {width}x{height}")
    print(f"Output resolution: {new_width}x{new_height}")
    print(f"Using model: {model_name.upper()} with scale factor: {scale}")
    
    # Create a temporary output file for video without audio
    temp_output = output_path + ".temp.mp4"
    
    # Remove temp file if it already exists
    if os.path.exists(temp_output):
        os.remove(temp_output)
    
    # Create the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (new_width, new_height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video file: {temp_output}")
        return False
    
    # Process frames with super-resolution
    start_time = time.time()
    processed_frames = 0
    
    with tqdm(total=frame_count, desc="Enhancing video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply super-resolution
            upscaled = model.upsample(frame)
            
            # Write to output video
            out.write(upscaled)
            processed_frames += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()
    
    # Calculate processing statistics
    elapsed_time = time.time() - start_time
    fps_processing = processed_frames / elapsed_time
    
    print(f"\nProcessed {processed_frames} frames in {elapsed_time:.2f} seconds")
    print(f"Processing speed: {fps_processing:.2f} fps")
    
    # Check if the original video has audio
    has_audio = False
    if keep_audio and has_ffmpeg():
        try:
            # Check if input video has audio
            cmd = [
                'ffprobe', '-i', input_path, 
                '-show_streams', '-select_streams', 'a', 
                '-loglevel', 'error'
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            has_audio = len(result.stdout) > 0
        except:
            has_audio = False
    
    # Add audio from original video if requested and if original has audio
    if keep_audio and has_ffmpeg() and has_audio:
        print("Adding audio from original video...")
        try:
            # Remove output file if it already exists
            if os.path.exists(output_path):
                os.remove(output_path)
                
            # Use a different approach for copying audio
            # First, extract audio from original video
            audio_temp = input_path + ".audio.aac"
            if os.path.exists(audio_temp):
                os.remove(audio_temp)
                
            cmd_extract = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-vn', '-acodec', 'copy',
                audio_temp
            ]
            
            # Run the command to extract audio
            subprocess.run(cmd_extract, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Then combine video and audio
            cmd_combine = [
                'ffmpeg', '-y',
                '-i', temp_output,
                '-i', audio_temp,
                '-c:v', 'copy',
                '-c:a', 'copy',
                '-shortest',
                output_path
            ]
            
            # Run the command to combine
            subprocess.run(cmd_combine, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Remove temporary files
            if os.path.exists(temp_output):
                os.remove(temp_output)
            if os.path.exists(audio_temp):
                os.remove(audio_temp)
                
            print("Audio added successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error adding audio: {e}")
            print("Using video without audio.")
            # Make sure output file doesn't exist before renaming
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_output, output_path)
    else:
        if keep_audio and not has_ffmpeg():
            print("FFmpeg not found. Cannot add audio to the enhanced video.")
        elif keep_audio and not has_audio:
            print("Original video does not have audio. No audio to add.")
        
        # Make sure output file doesn't exist before renaming
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_output, output_path)
    
    # Optimize quality if requested and ffmpeg is available
    if quality < 100 and has_ffmpeg():
        print(f"Optimizing output quality ({quality}%)...")
        temp_output = output_path + ".temp.mp4"
        
        # Make sure temp file doesn't exist before renaming
        if os.path.exists(temp_output):
            os.remove(temp_output)
            
        os.rename(output_path, temp_output)
        
        try:
            # Command to optimize video quality
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_output,
                '-c:v', 'libx264',
                '-crf', str((100-quality)/5),
                '-preset', 'medium',
                '-c:a', 'copy',
                output_path
            ]
            
            # Run the command
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Remove temporary file
            if os.path.exists(temp_output):
                os.remove(temp_output)
            print("Quality optimization completed.")
        except subprocess.CalledProcessError as e:
            print(f"Error during quality optimization: {e}")
            print("Using unoptimized output instead.")
            # Make sure output file doesn't exist before renaming
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_output, output_path)
    
    return True

def main():
    """Main function to run the script."""
    args = parse_arguments()
    
    # Set default output path if not specified
    if args.output is None:
        filename, ext = os.path.splitext(args.input)
        args.output = f"{filename}_enhanced.mp4"
    
    print(f"Enhancing video: {args.input}")
    print(f"Output will be saved to: {args.output}")
    
    # Check dependencies
    try:
        import tqdm
    except ImportError:
        print("Missing dependency: tqdm")
        print("Please install it with: pip install tqdm")
        return
    
    try:
        import cv2.dnn_superres
    except (ImportError, AttributeError):
        print("Missing dependency: OpenCV with contrib modules")
        print("Please install it with: pip install opencv-contrib-python")
        return
    
    # Check for FFmpeg
    if not has_ffmpeg():
        print("Warning: FFmpeg not found. Audio preservation and quality optimization will be disabled.")
        print("To enable these features, please install FFmpeg and ensure it's in your system PATH.")
    
    # Enhance the video
    success = enhance_video(
        args.input, 
        args.output, 
        model_name=args.model, 
        scale=args.scale,
        quality=args.quality,
        keep_audio=args.keep_audio
    )
    
    if success:
        print(f"Video enhancement completed successfully!")
        print(f"Enhanced video saved to: {args.output}")
    else:
        print("Video enhancement failed.")

if __name__ == "__main__":
    main() 