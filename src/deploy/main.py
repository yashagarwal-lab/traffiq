#Main loop - camera -> model -> vehicle control.

import argparse
import time
import sys
import numpy as np

from src.deploy.camera import CameraThread
from src.deploy.vehicle_control import VehicleController
from src.inference.engine import TFLiteEngine
from src.inference.decision_fusion import DecisionFusion
from src.inference.safe_stop import SafeStopController
from src.data.preprocess import preprocess


def main(model_path, dry_run=False):
    print(f"Starting TraffIQ - model: {model_path}")
    print(f"Mode: {'dry run' if dry_run else 'LIVE'}")
    
    camera = CameraThread(resolution=(640, 480), use_picamera=(not dry_run))
    engine = TFLiteEngine(model_path, num_threads=4)
    fusion = DecisionFusion()
    safety = SafeStopController()
    vehicle = VehicleController(dry_run=dry_run)
    
    camera.start()
    
    # wait for camera
    for _ in range(50):
        if camera.get_latest_frame() is not None:
            break
        time.sleep(0.1)
    else:
        print("Camera failed to start")
        camera.stop()
        sys.exit(1)
    
    # warm up inference
    dummy = np.random.uniform(-1, 1, (1, 224, 224, 3)).astype(np.float32)
    engine.predict(dummy)
    
    print("Running...")
    frame_count = 0
    
    try:
        while True:
            t0 = time.perf_counter()
            
            frame = camera.get_latest_frame()
            if frame is None:
                continue
            
            processed = preprocess(frame, add_batch_dim=True)
            nav, grid, confidence = engine.predict(processed)
            fused = fusion.fuse(nav, grid)
            final = safety.check(frame, fused, confidence, grid)
            vehicle.send(final[0], final[1])
            
            frame_count += 1
            if frame_count % 100 == 0:
                fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
                print(f"frame {frame_count} | fps ~{fps:.0f} | speed={final[0]:.2f} dir={final[1]:.2f}")
    
    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        vehicle.emergency_stop()
        camera.stop()
        vehicle.cleanup()
        print(f"Done. {frame_count} frames processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.model, args.dry_run)
