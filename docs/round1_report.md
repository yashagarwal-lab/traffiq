# TraffIQ - Round 1 Report

**Team SafeNSound** — Sahil Sharma & Yash Agarwal

## What we're building

A self-driving car AI that takes camera images and outputs speed + direction. Runs on Raspberry Pi 4B.

- Input: camera image (640x480)
- Output: [speed, direction], both between -1 and 1
- Direction: +1 = right, -1 = left
- Speed = 0 with direction = car spins in place

## Our model

We went with MobileNetV2 as the base — it's a pretrained image classifier from Google that's designed for phones and small devices. We use it as a feature extractor and add our own layers on top.

The model has two outputs:
1. **Navigation** - predicts speed and direction (2 numbers)
2. **Obstacle grid** - splits the image into a 3x3 grid and predicts if each section has an obstacle (9 numbers)

We went with this grid idea instead of full object detection (like YOLO) because object detection is way too slow on a Pi — it needs extra processing steps that eat up the time budget.

## Preprocessing

Before feeding images to the model we:
1. Crop the top 40% (ceiling is useless)
2. Resize to 224x224 (what MobileNetV2 expects)
3. Apply CLAHE — its a contrast enhancement that works locally, so it handles different lighting conditions (bright spots, dark corners) much better than just normalizing
4. Scale pixels to [-1, 1]

## Training

We used publicly available driving datasets since we don't have access to the arena yet. We also generated some synthetic images — just road lines and colored boxes as obstacles — to train the obstacle grid part.

For augmentation we do random brightness changes, blur, noise, color shifts, and horizontal flips. The flip is tricky because you also have to negate the direction label (left becomes right) and mirror the obstacle grid.

Trained on our laptop with a RTX 3070 Ti using TensorFlow.

## Quantization

To run fast on the Pi, we convert the model from 32-bit floats to 8-bit integers (INT8). This makes it ~4x smaller and faster.

We actually tried MobileNetV3 first because its smaller, but when we quantized it the outputs drifted too much (predictions were off by ~8%). MobileNetV2 only drifted ~3.5%. This is because V3 uses an activation function (HardSwish) that doesn't handle the rounding well. V2 uses ReLU6 which is simpler and quantizes cleanly.

Final model size: 1.19 MB

## Safety

We have 6 conditions that trigger an emergency stop:
- Camera feed is too dark (probably camera failure)
- Camera feed is too bright/white (glare)
- Camera feed isn't changing (frozen/stuck)
- Model is very uncertain for multiple frames
- Obstacle detected ahead but model says drive forward (override)
- Any crash/error in the code

## Deployment

On the Pi, the camera runs in a background thread so it doesn't block the main loop. The main loop just grabs the latest frame, runs inference, checks safety, and sends commands.

## What's next

- Train on actual arena data once we get access
- Fine-tune obstacle detection for the specific obstacles used
- Tune speed control based on the car's actual handling
