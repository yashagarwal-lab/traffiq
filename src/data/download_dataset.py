# Download/convert public driving datasets to our training format
# Supports: Sully Chen driving data, Udacity simulator, synthetic generation

import os
import csv
import argparse
import urllib.request
import zipfile
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.training import config

FIELDNAMES = [
    "image_path", "speed", "direction",
    "g00", "g01", "g02", "g10", "g11", "g12", "g20", "g21", "g22",
]


def download_sully_chen(output_dir=None):
    output_dir = output_dir or os.path.join(config.DATA_DIR, "sullychen")
    os.makedirs(output_dir, exist_ok=True)
    
    url = "https://drive.google.com/uc?export=download&id=0B-KJCaaF7ellSmh1T0lYWTBOdjg"
    zip_path = os.path.join(output_dir, "driving_dataset.zip")
    
    print("Sully Chen Dataset (~2.2GB)")
    print(f"If auto-download fails, get it from: https://github.com/SullyChen/driving-datasets")
    print(f"and extract to: {output_dir}")
    
    data_txt = os.path.join(output_dir, "driving_dataset", "data.txt")
    if os.path.exists(data_txt):
        print("Already downloaded, converting...")
        return convert_sully_chen(output_dir)
    
    print(f"Downloading...")
    try:
        urllib.request.urlretrieve(url, zip_path, _progress_hook)
        print("\nExtracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(output_dir)
        os.remove(zip_path)
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("Download manually from GitHub")
        return None
    
    return convert_sully_chen(output_dir)


def convert_sully_chen(dataset_dir):
    # data.txt format: "image_filename steering_angle"
    # steering in degrees, positive=right, negative=left (matches our convention)
    data_txt = os.path.join(dataset_dir, "driving_dataset", "data.txt")
    img_dir = os.path.join(dataset_dir, "driving_dataset")
    output_csv = os.path.join(config.DATA_DIR, "labels.csv")
    
    if not os.path.exists(data_txt):
        print(f"Error: {data_txt} not found")
        return None
    
    # find max angle for normalization
    max_angle = 0.0
    with open(data_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                max_angle = max(max_angle, abs(float(parts[1])))
    if max_angle == 0:
        max_angle = 1.0
    
    count = 0
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        
        with open(data_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                img_path = os.path.join(img_dir, parts[0])
                if not os.path.exists(img_path):
                    continue
                
                # normalize angle to [-1, 1]
                direction = np.clip(float(parts[1]) / max_angle, -1.0, 1.0)
                # estimate speed (no speed data in this dataset)
                speed = max(0.1, 0.6 - 0.4 * abs(direction))
                
                writer.writerow({
                    "image_path": os.path.abspath(img_path),
                    "speed": f"{speed:.3f}", "direction": f"{direction:.3f}",
                    "g00": "0", "g01": "0", "g02": "0",
                    "g10": "0", "g11": "0", "g12": "0",
                    "g20": "0", "g21": "0", "g22": "0",
                })
                count += 1
    
    print(f"  Converted {count} frames -> {output_csv}")
    return output_csv


def convert_udacity_sim(data_dir):
    # driving_log.csv format: center_img, left_img, right_img, steering, throttle, brake, speed
    log_path = os.path.join(data_dir, "driving_log.csv")
    output_csv = os.path.join(config.DATA_DIR, "labels.csv")
    
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found")
        return None
    
    count = 0
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        
        with open(log_path, 'r') as f:
            for row in csv.reader(f):
                if len(row) < 7:
                    continue
                
                center_img = row[0].strip()
                steering = float(row[3].strip())
                throttle = float(row[4].strip())
                
                img_path = center_img
                if not os.path.isabs(img_path):
                    img_path = os.path.join(data_dir, center_img)
                if not os.path.exists(img_path):
                    img_path = os.path.join(data_dir, "IMG", os.path.basename(center_img))
                if not os.path.exists(img_path):
                    continue
                
                direction = np.clip(steering, -1.0, 1.0)
                speed = np.clip(throttle, -1.0, 1.0)
                if speed < 0.05:
                    speed = max(0.1, 0.5 - 0.3 * abs(direction))
                
                writer.writerow({
                    "image_path": os.path.abspath(img_path),
                    "speed": f"{speed:.3f}", "direction": f"{direction:.3f}",
                    "g00": "0", "g01": "0", "g02": "0",
                    "g10": "0", "g11": "0", "g12": "0",
                    "g20": "0", "g21": "0", "g22": "0",
                })
                count += 1
    
    print(f"  Converted {count} frames -> {output_csv}")
    return output_csv


def create_synthetic_dataset(num_samples=500, output_dir=None):
    # generates simple road images with lane lines and random obstacles
    # just for testing the pipeline, not for real training
    import cv2
    
    output_dir = output_dir or os.path.join(config.DATA_DIR, "synthetic")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(config.DATA_DIR, "labels.csv")
    
    print(f"Generating {num_samples} synthetic training samples...")
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        
        for i in range(num_samples):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img[200:, :] = [100, 100, 100]  # road
            img[:200, :] = [180, 200, 220]  # sky
            
            curve = np.random.uniform(-1, 1)
            offset = int(curve * 150)
            cx = 320 + offset
            
            # lane lines
            cv2.line(img, (cx - 200, 480), (cx - 50, 250), (255, 255, 255), 3)
            cv2.line(img, (cx + 200, 480), (cx + 50, 250), (255, 255, 255), 3)
            
            # random obstacle
            grid = [0] * 9
            has_obs = np.random.random() < 0.3
            if has_obs:
                ox = np.random.randint(100, 540)
                oy = np.random.randint(250, 420)
                sz = np.random.randint(20, 60)
                cv2.rectangle(img, (ox, oy), (ox + sz, oy + sz), (0, 0, 200), -1)
                gr = min(2, (oy - 200) * 3 // 280)
                gc = min(2, ox * 3 // 640)
                grid[gr * 3 + gc] = 1
            
            # noise
            img = cv2.add(img, np.random.randint(0, 15, img.shape, dtype=np.uint8))
            
            filepath = os.path.join(output_dir, f"synthetic_{i:05d}.jpg")
            cv2.imwrite(filepath, img)
            
            direction = np.clip(curve * 0.8, -1.0, 1.0)
            speed = 0.0 if (has_obs and grid[4] == 1) else (0.4 if has_obs else 0.6)
            
            writer.writerow({
                "image_path": os.path.abspath(filepath),
                "speed": f"{speed:.3f}", "direction": f"{direction:.3f}",
                "g00": str(grid[0]), "g01": str(grid[1]), "g02": str(grid[2]),
                "g10": str(grid[3]), "g11": str(grid[4]), "g12": str(grid[5]),
                "g20": str(grid[6]), "g21": str(grid[7]), "g22": str(grid[8]),
            })
    
    print(f"  Generated {num_samples} samples -> {output_csv}")
    print(f"  Images saved to {output_dir}")
    return output_csv


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        print(f"\r  {pct}% ({downloaded // 1024 // 1024}MB)", end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["sullychen", "udacity", "synthetic"])
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()
    
    if args.dataset == "sullychen":
        download_sully_chen()
    elif args.dataset == "udacity":
        if not args.path:
            print("Need --path for udacity data")
        else:
            convert_udacity_sim(args.path)
    elif args.dataset == "synthetic":
        create_synthetic_dataset(args.samples)
