# Data collection - drive with keyboard, save frames + labels
#
# Controls: WASD or arrow keys
#   W/Up    = forward
#   S/Down  = reverse
#   A/Left  = left (-1)
#   D/Right = right (+1)  
#   Space   = stop
#   Q       = quit and save

import os
import csv
import time
import argparse
import cv2
import numpy as np


def collect(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "labels.csv")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Can't open camera")
        return
    
    fieldnames = [
        "image_path", "speed", "direction",
        "g00", "g01", "g02", "g10", "g11", "g12", "g20", "g21", "g22",
    ]
    
    csvfile = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    speed = 0.0
    direction = 0.0
    frame_idx = 0
    
    print("Recording... WASD to drive, Q to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # show current state on frame
        display = frame.copy()
        cv2.putText(display, f"Speed: {speed:+.1f}  Dir: {direction:+.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Frame: {frame_idx}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("TraffIQ Data Collection", display)
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord("q"):
            break
        elif key == ord("w") or key == 82:
            speed = 0.5
        elif key == ord("s") or key == 84:
            speed = -0.5
        elif key == ord("a") or key == 81:
            direction = -1.0
        elif key == ord("d") or key == 83:
            direction = +1.0
        elif key == ord(" "):
            speed = 0.0
            direction = 0.0
        
        # save frame
        filename = f"frame_{frame_idx:06d}.jpg"
        filepath = os.path.join(frames_dir, filename)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filepath, frame)
        
        writer.writerow({
            "image_path": os.path.abspath(filepath),
            "speed": f"{speed:.3f}",
            "direction": f"{direction:.3f}",
            "g00": "0", "g01": "0", "g02": "0",
            "g10": "0", "g11": "0", "g12": "0",
            "g20": "0", "g21": "0", "g22": "0",
        })
        frame_idx += 1
        
        # decay direction back to 0
        direction *= 0.85
        if abs(direction) < 0.05:
            direction = 0.0
    
    csvfile.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {frame_idx} frames to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/raw/session_001")
    args = parser.parse_args()
    collect(args.output)
