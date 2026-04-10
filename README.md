# TraffIQ — Team SafeNSound

Autonomous RC car navigation for the TraffIQ competition (Round 2 Finals).

Built by **Sahil Sharma** and **Yash Agarwal**.

## What it does

The car navigates a walled indoor track with:
- White walls, black floor, white ceiling
- A traffic light (must stop on red, go on green)
- A box obstacle on the road (must dodge it)
- A stop sign at the end (must come to a full stop)

## How it works

We use a **hybrid approach** — no single model does everything well, so we split the job:

1. **Wall following** (OpenCV): Threshold the frame to find white walls, compute the gap between them, steer to stay centred. When a wall appears ahead, scan left/right for an opening and turn hard.

2. **Obstacle detection** (OpenCV + HSV): Traffic lights are red/green circles in HSV space. The stop sign is a red octagon. The box is anything with enough saturation sitting on the road.

3. **TFLite model** (MobileNetV2): Fallback for outdoor or unknown environments. INT8 quantised, runs in ~2ms per frame.

The system auto-detects whether it's on the indoor competition track (dark floor) and switches strategies accordingly.

## Files

```
Model.py                 ← main navigation logic (what gets judged)
config.py                ← camera resolution (640×480)
model_base.py            ← competition boilerplate (provided by organisers)
participant/
  model.tflite           ← quantised MobileNetV2 (~1.2 MB)
track_sim.py             ← 3D raycasting simulator for testing
requirements.txt         ← runtime dependencies
```

## Running the simulator

```bash
conda activate traffiq
python track_sim.py
```

Controls: `R` to reset, `T` to toggle the traffic light, `ESC` to quit.

The simulator renders a first-person 3D view of the track using a Wolfenstein-style raycaster. All track parameters (corridor lengths, obstacle positions, car dimensions) are configurable at the top of `track_sim.py`.

## Tuning on competition day

All thresholds live at the top of `Model.py` as class constants:

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `WALL_THRESH` | 170 | Grayscale cutoff for walls |
| `BASE_SPEED` | 0.40 | Cruise speed on straights |
| `CORNER_SPEED` | 0.15 | Speed during hard turns |
| `RED_MIN_AREA` | 150 | Min red blob size for traffic light |
| `STOP_MIN_AREA` | 300 | Min red blob size for stop sign |
| `BOX_MIN_AREA` | 400 | Min coloured blob size for box |

If the lighting conditions are different from expected, tweak `WALL_THRESH` first. If the car is hitting walls, lower `BASE_SPEED`. If it's missing the traffic light, lower `RED_MIN_AREA`.

## Dependencies

- Python 3.11
- numpy
- opencv-python
- ai-edge-litert (or tflite-runtime)
- pygame (simulator only)
