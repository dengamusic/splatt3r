#!/usr/bin/env python3
import os
import subprocess
import urllib.request

output_dir = "D:/ML_for3D/SAM"
weights_dir = os.path.join(output_dir, "weights")


# Download ViT-H SAM weights
url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
os.makedirs(weights_dir, exist_ok=True)
target = os.path.join(weights_dir, "sam_vit_h_4b8939.pth")

if not os.path.exists(target):
    print(f"Downloading ViT-H SAM weights to {target}…")
    urllib.request.urlretrieve(url, target)
    print("Download complete.")
else:
    print("Weights already exist:", target)
