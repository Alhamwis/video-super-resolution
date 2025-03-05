# Video Super Resolution

This project uses OpenCV's DNN Super Resolution module and PyTorch-based models to upscale videos using various super-resolution algorithms.

## Features

- Upscale videos using AI-based super-resolution algorithms
- Preserve audio from the original video
- Progress bar showing enhancement status
- Multiple super-resolution algorithms:
  - OpenCV-based: ESPCN, FSRCNN, LapSRN
  - PyTorch-based: ESRGAN, Real-ESRGAN, SRCNN
- Multiple scale factors (2x, 3x, 4x)
- Adjustable output quality
- Automatic model downloading
- Side-by-side comparison of different enhancement methods

## Requirements

### For OpenCV-based enhancement:
- Python 3.x
- OpenCV (with contrib modules)
- NumPy
- tqdm (for progress bar)
- FFmpeg (for audio preservation and quality optimization)

### For PyTorch-based enhancement:
- Python 3.x
- PyTorch
- NumPy
- OpenCV
- tqdm (for progress bar)
- FFmpeg (for audio preservation)
- Additional dependencies for specific models (automatically installed)

Install the required packages:

```bash
# For OpenCV-based enhancement
pip install opencv-python opencv-contrib-python numpy tqdm

# For PyTorch-based enhancement
pip install torch numpy opencv-python tqdm
```

## Available Models

### OpenCV-based Models:
- ESPCN (x2, x3, x4) - Fast but lower quality
- FSRCNN (x2, x3, x4) - Balanced speed and quality
- LapSRN (x2, x4) - Higher quality but slower

### PyTorch-based Models:
- ESRGAN (x4) - High quality enhancement
- Real-ESRGAN (x4) - State-of-the-art quality
- SRCNN (x4) - Classic super-resolution model

> **Note:** The PyTorch implementation includes a simplified version of the models that works without additional dependencies. This ensures compatibility across different systems but may not achieve the same quality as the full implementations.

## Usage

### OpenCV-based Enhancement (enhance_video.py)

```bash
python enhance_video.py -i INPUT_VIDEO [-o OUTPUT_VIDEO] [-m MODEL] [-s SCALE] [-q QUALITY]
```

### PyTorch-based Enhancement (torch_enhance_video.py)

```bash
python torch_enhance_video.py -i INPUT_VIDEO -o OUTPUT_VIDEO [-m MODEL] [-p PRESERVE_AUDIO] [-pm PROCESSING_MODE] [-q QUALITY]
```

### Comparison Tool (compare_methods.py)

```bash
python compare_methods.py -o ORIGINAL_VIDEO -e ENHANCED_VIDEO1 [ENHANCED_VIDEO2 ...] [-c COMPARISON_OUTPUT]
```

### Audio Preservation Method

If you're having issues with audio preservation in the enhanced script, you can use this three-step process:

1. Extract audio from the original video:
```bash
ffmpeg -i "input_video.mp4" -vn -acodec copy "original_audio.aac"
```

2. Enhance the video without audio:
```bash
python enhance_video.py -i "input_video.mp4" -o "enhanced_video.mp4" -m "fsrcnn" -s 2 -q 100
```

3. Combine the enhanced video with the original audio:
```bash
ffmpeg -i "enhanced_video.mp4" -i "original_audio.aac" -c:v copy -c:a libvo_aacenc -map 0:v:0 -map 1:a:0 -shortest "final_with_audio.mp4"
```

## OpenCV-based Enhancement Arguments

### Required Arguments
- `-i, --input`: Path to input video

### Optional Arguments
- `-o, --output`: Path to output video (default: input_enhanced.mp4)
- `-m, --model`: Super-resolution model to use (default: espcn, choices: espcn, fsrcnn, lapsrn)
- `-s, --scale`: Scale factor for resolution enhancement (default: 2, choices: 2, 3, 4)
- `-q, --quality`: Output video quality (0-100, default: 95)
- `--keep-audio`: Preserve audio from the original video (default: True)

## PyTorch-based Enhancement Arguments

### Required Arguments
- `-i, --input`: Path to input video
- `-o, --output`: Path to output video

### Optional Arguments
- `-m, --model`: Super-resolution model to use (default: esrgan, choices: esrgan, realesrgan, srcnn)
- `-mp, --model_path`: Custom model weights path
- `-p, --preserve`: Preserve audio from original video (default: audio, choices: audio, none)
- `-pm, --processing_mode`: Processing mode (default: full, choices: full, sample)
- `-q, --quality`: Output video quality (0-100, default: 95)

## Comparison Tool Arguments

### Required Arguments
- `-o, --original`: Path to the original video
- `-e, --enhanced`: Paths to enhanced videos (one or more)

### Optional Arguments
- `-c, --comparison`: Output comparison video path (default: comparison.mp4)

## Examples

### OpenCV-based Enhancement

Upscale a video by a factor of 2 using the ESPCN algorithm:

```bash
python enhance_video.py -i "input_video.mp4" -o "upscaled_video_x2.mp4" -m "espcn" -s 2
```

Upscale a video by a factor of 4 using the FSRCNN algorithm with higher quality:

```bash
python enhance_video.py -i "input_video.mp4" -o "upscaled_video_x4.mp4" -m "fsrcnn" -s 4 -q 98
```

### PyTorch-based Enhancement

Upscale a video using ESRGAN with audio preservation:

```bash
python torch_enhance_video.py -i "input_video.mp4" -o "enhanced_esrgan.mp4" -m "esrgan" -p "audio"
```

Upscale a video using Real-ESRGAN with sample processing mode (faster):

```bash
python torch_enhance_video.py -i "input_video.mp4" -o "enhanced_realesrgan.mp4" -m "realesrgan" -pm "sample"
```

### Comparison

Create a side-by-side comparison of different enhancement methods:

```bash
python compare_methods.py -o "original.mp4" -e "enhanced_opencv.mp4" "enhanced_pytorch.mp4" -c "comparison.mp4"
```

## Model Comparison

### OpenCV-based Models:
1. **ESPCN**: Fastest processing (5-6 fps), but lower quality enhancement
2. **FSRCNN**: Good balance between speed (5-6 fps) and quality
3. **LapSRN**: Better quality enhancement, but slower (1-1.5 fps)

### PyTorch-based Models:
1. **ESRGAN**: High-quality enhancement, slower processing
2. **Real-ESRGAN**: State-of-the-art quality, slowest processing
3. **SRCNN**: Classic model, good balance between quality and speed

For quick enhancement with decent quality, use FSRCNN (OpenCV) with scale factor 2.
For highest quality enhancement, use Real-ESRGAN (PyTorch), but expect longer processing times.

## Simplified Implementation

The PyTorch-based enhancement script includes a simplified implementation that works without additional dependencies. This implementation:

1. Uses a basic residual network for enhancement
2. Performs bicubic upscaling first, then applies enhancement
3. Works on any system with PyTorch installed
4. Processes at approximately 0.4 fps on CPU

While this simplified implementation may not achieve the same quality as the full models, it provides a good balance between quality and compatibility.

## Comparison Tool

The comparison tool creates a side-by-side video showing the original video alongside multiple enhanced versions. This allows for easy visual comparison of different enhancement methods. The tool:

1. Extracts sample frames from each video
2. Creates a composite image with all versions side by side
3. Adds labels to identify each version
4. Combines the composite images into a comparison video

This is particularly useful for evaluating the quality differences between enhancement methods and determining which one best suits your needs.

## Downloading Models

### OpenCV-based Models:
```bash
python download_all_models.py
```

### PyTorch-based Models:
```bash
python download_torch_models.py
```

These scripts will download all available models to the `models` directory.

## Notes

- The upscaling process can be computationally intensive, especially for larger videos or higher scale factors.
- PyTorch-based models generally provide higher quality but require more processing power.
- GPU acceleration is automatically used if available (CUDA for PyTorch, OpenCL for OpenCV).
- For best performance with PyTorch models, a CUDA-capable GPU is highly recommended.
- FFmpeg is required for audio preservation and quality optimization.
- The "sample" processing mode in PyTorch enhancement only processes 10 frames, useful for quick testing. 