import cv2
import argparse

# Define possible backends and targets
backends = {
    "opencv": cv2.dnn.DNN_BACKEND_OPENCV,
    "inference_engine": cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE,
    "halide": cv2.dnn.DNN_BACKEND_HALIDE,
}

targets = {
    "cpu": cv2.dnn.DNN_TARGET_CPU,
    "opencl": cv2.dnn.DNN_TARGET_OPENCL,
    "opencl_fp16": cv2.dnn.DNN_TARGET_OPENCL_FP16,
    "myriad": cv2.dnn.DNN_TARGET_MYRIAD,
    "fpga": cv2.dnn.DNN_TARGET_FPGA,
}

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Path to input video")
parser.add_argument("-o", "--output", required=True, help="Path to output video")
parser.add_argument("-m", "--model", required=True, help="Path to model file")
parser.add_argument("-a", "--algorithm", required=True, help="Algorithm name (edsr, espcn, fsrcnn, lapsrn)")
parser.add_argument("-s", "--scale", type=int, required=True, help="Scale factor")
parser.add_argument("--backend", choices=backends.keys(), default="opencv", help="Computation backend")
parser.add_argument("--target", choices=targets.keys(), default="cpu", help="Computation target")
args = parser.parse_args()

# Create the super resolution object
sr = cv2.dnn_superres.DnnSuperResImpl_create()

# Read the model first
sr.readModel(args.model)

# Set the model and scale
sr.setModel(args.algorithm, args.scale)

# Set backend and target after reading the model
sr.setPreferableBackend(backends[args.backend])
sr.setPreferableTarget(targets[args.target])

# Open input video
input_video = cv2.VideoCapture(args.input)

# Check if video opened successfully
if not input_video.isOpened():
    print("Error opening video file")
    exit()

# Get video properties
fps = input_video.get(cv2.CAP_PROP_FPS)
width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate output resolution
output_width = width * args.scale
output_height = height * args.scale

# Open output video
output_video = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))

# Check if output video writer is initialized correctly
if not output_video.isOpened():
    print("Error opening output video file")
    exit()

# Process each frame
while True:
    ret, frame = input_video.read()
    if not ret:
        break

    # Upsample the frame
    upscaled_frame = sr.upsample(frame)

    # Write to output video
    output_video.write(upscaled_frame)

# Release resources
input_video.release()
output_video.release()

print("Video upscaled successfully.") 