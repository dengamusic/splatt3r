#!/usr/bin/env python3
import os
import argparse
import urllib.request

def parse_args():
    parser = argparse.ArgumentParser(description="Download SAM weights")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    output_dir = args.output_dir
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

if __name__ == "__main__":
    main()