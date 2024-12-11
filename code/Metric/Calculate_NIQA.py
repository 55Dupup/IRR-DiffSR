import sys
sys.path.append('/usr/local/Matlab/R2020a/extern/engines/python')
import matlab.engine
import os
import numpy as np
from PIL import Image

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Define folder path
folder_path = '/data/55d/DiffIR-RealSR/results/test_AID_IRR_DiffSRGANModel/visualization'

# Define output file path
output_file_path = '/data/55d/DiffIR-RealSR/results/test_AID_IRR_DiffSRGANModel/visualization/AID_NIQA_Result.py'

# Define image extensions
image_extensions = ['.jpg', '.jpeg', '.png']

# Define quality metrics
quality_metrics = ['niqe', 'brisque', 'piqe']

# Check if folder path exists
if not os.path.exists(folder_path):
    print(f"Error: Folder path {folder_path} does not exist.")
    exit()

# Get subfolders
subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

# Loop through subfolders
for i, subfolder in enumerate(subfolders):
    # Get images
    images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(subfolder) for f in filenames if os.path.splitext(f)[1].lower() in image_extensions]
    # Loop through quality metrics
    results = []
    for quality_metric in quality_metrics:
        # Calculate quality metric for each image
        quality_metric_results = []
        for j, image in enumerate(images):
            try:
                # Load image
                img = Image.open(image)
                # Convert image to grayscale
                img = img.convert('L')
                # Convert image to MATLAB array
                img = matlab.double(np.array(img).tolist())
                # Calculate quality metric
                quality_metric_result = eng.eval(f"{quality_metric}({img})", nargout=1)
                # Append result to list
                quality_metric_results.append(quality_metric_result)
            except Exception as e:
                print(f"Error processing image {image}: {e}")
            # Display progress
            print(f"Processing image {j+1} of {len(images)} in subfolder {i+1} of {len(subfolders)}")
        # Calculate average result
        average_result = sum(quality_metric_results) / len(quality_metric_results)
        # Append average result to list
        results.append(average_result)
    # Write results to file
    with open(output_file_path, 'a') as f:
        f.write(f"{subfolder}\n")
        f.write(f"NIQE: {results[0]}\n")
        f.write(f"BRISQUE: {results[1]}\n")
        f.write(f"PIQE: {results[2]}\n")
        f.write("\n")
    # Display progress
    print(f"Processed subfolder {i+1} of {len(subfolders)}")

# Stop MATLAB engine
eng.quit()
