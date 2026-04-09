# TraffIQ

Self-driving car AI for the TraffIQ competition. Takes camera input, outputs speed and direction.

## How it works

MobileNetV2 backbone with two output heads:
- Navigation: outputs [speed, direction] 
- Obstacle grid: 3x3 grid of obstacle probabilities

Direction: right=+1, left=-1. Speed=0 with direction = spin in place.

## Setup

```bash
# needs conda env with python 3.11 + tensorflow
./run.sh -m src.data.download_dataset --dataset synthetic --samples 500
./run.sh -m src.training.train --backbone v2
./run.sh -m src.training.quantize --model models/run_XXX/final_model.keras --validate
```

## Structure

```
src/
├── data/       - data collection, preprocessing, augmentation
├── model/      - backbone, model architecture, loss
├── training/   - training, evaluation, quantization
├── inference/  - tflite engine, obstacle fusion, safety
└── deploy/     - camera thread, main loop, vehicle control
```
