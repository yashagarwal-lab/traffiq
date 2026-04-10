# SafeNSound — Round 2 Technical Report

## Team
- **Sahil Sharma**
- **Yash Agarwal**

## Approach

We use a hybrid pipeline: OpenCV for the indoor track and a quantised TFLite model (MobileNetV2) as fallback. The system auto-detects indoor vs outdoor by checking floor colour.

### Navigation

The car follows white walls by thresholding to grayscale and computing centring error across three horizontal scan lines. When a wall blocks the path ahead (upper-centre ROI is >70% white), the car scans left/right for an opening and turns hard.

### Obstacle Handling

| Obstacle | Detection | Action |
|----------|-----------|--------|
| Red light | HSV red mask + circularity check | Full stop until green |
| Green light | HSV green mask + circularity check | Resume driving |
| Stop sign | HSV red mask + polygon vertices (6-12 = octagon) | Permanent stop after 5 frames |
| Box obstacle | HSV saturation filter on road strip | Slow to 15%, steer ±0.8 away |

### Performance

- Inference: 2.1ms average, well within 100ms limit
- Model: INT8 quantised MobileNetV2, 1.2MB
- Runtime: ai-edge-litert (pure TFLite, no TensorFlow)

## Files

- `Model.py` — navigation logic (269 lines)
- `config.py` — resolution config
- `participant/model.tflite` — quantised model
- `track_sim.py` — raycasting simulator for testing

## Verification

All tests pass consistently:
- ✅ Red light stop
- ✅ Green light resume
- ✅ Stop sign detection
- ✅ Box avoidance (22cm clearance with 16cm car)
