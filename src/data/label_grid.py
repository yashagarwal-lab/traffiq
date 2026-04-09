# Grid labeling tool - click on 3x3 grid cells to mark obstacles
# Used to create obstacle labels for the training data

import os
import argparse
import cv2
import numpy as np
import pandas as pd


class GridLabeler:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.grid = np.zeros((3, 3), dtype=int)
        self.current_idx = 0
    
    def draw_grid(self, image):
        h, w = image.shape[:2]
        cell_w, cell_h = w // 3, h // 3
        
        display = image.copy()
        
        for r in range(3):
            for c in range(3):
                x1, y1 = c * cell_w, r * cell_h
                x2, y2 = x1 + cell_w, y1 + cell_h
                
                if self.grid[r, c] == 1:
                    overlay = display.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                    display = cv2.addWeighted(overlay, 0.3, display, 0.7, 0)
                
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        cv2.putText(display, f"Frame {self.current_idx}/{len(self.df)} | Click=toggle, N=next, P=prev, S=save, Q=quit",
                    (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return display
    
    def on_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        
        img = param
        h, w = img.shape[:2]
        col = x // (w // 3)
        row = y // (h // 3)
        if 0 <= row < 3 and 0 <= col < 3:
            self.grid[row, col] = 1 - self.grid[row, col]  # toggle
    
    def save_labels(self):
        for r in range(3):
            for c in range(3):
                self.df.at[self.current_idx, f"g{r}{c}"] = self.grid[r, c]
    
    def run(self):
        cv2.namedWindow("Grid Labeler")
        
        while self.current_idx < len(self.df):
            row = self.df.iloc[self.current_idx]
            img_path = row["image_path"]
            
            img = cv2.imread(img_path)
            if img is None:
                self.current_idx += 1
                continue
            
            # load existing labels
            for r in range(3):
                for c in range(3):
                    self.grid[r, c] = int(row.get(f"g{r}{c}", 0))
            
            cv2.setMouseCallback("Grid Labeler", self.on_click, img)
            
            while True:
                display = self.draw_grid(img)
                cv2.imshow("Grid Labeler", display)
                key = cv2.waitKey(50) & 0xFF
                
                if key == ord("n"):
                    self.save_labels()
                    self.current_idx += 1
                    break
                elif key == ord("p"):
                    self.save_labels()
                    self.current_idx = max(0, self.current_idx - 1)
                    break
                elif key == ord("s"):
                    self.save_labels()
                    self.df.to_csv(self.csv_path, index=False)
                    print(f"Saved to {self.csv_path}")
                elif key == ord("q"):
                    self.save_labels()
                    self.df.to_csv(self.csv_path, index=False)
                    cv2.destroyAllWindows()
                    return
        
        self.df.to_csv(self.csv_path, index=False)
        cv2.destroyAllWindows()
        print("Done labeling")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()
    GridLabeler(args.csv).run()
