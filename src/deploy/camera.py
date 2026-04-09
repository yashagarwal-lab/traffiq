#Camera capture - runs in background thread.

import threading
import numpy as np


class CameraThread(threading.Thread):
    def __init__(self, resolution=(640, 480), use_picamera=True):
        super().__init__(daemon=True)
        self.resolution = resolution
        self.use_picamera = use_picamera
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self._camera = None
    
    def run(self):
        if self.use_picamera:
            try:
                self._run_picamera()
            except ImportError:
                print("picamera2 not available, using opencv")
                self._run_opencv()
        else:
            self._run_opencv()
    
    def _run_picamera(self):
        from picamera2 import Picamera2
        self._camera = Picamera2()
        cfg = self._camera.create_preview_configuration(
            main={"size": self.resolution, "format": "RGB888"}
        )
        self._camera.configure(cfg)
        self._camera.start()
        
        while self.running:
            frame = self._camera.capture_array()
            with self.lock:
                self.frame = frame
    
    def _run_opencv(self):
        import cv2
        self._camera = cv2.VideoCapture(0)
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        while self.running:
            ret, frame = self._camera.read()
            if ret:
                frame = frame[:, :, ::-1].copy()  # BGR -> RGB
                with self.lock:
                    self.frame = frame
    
    def get_latest_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.running = False
        if self._camera is not None:
            if hasattr(self._camera, "stop"):
                self._camera.stop()
            elif hasattr(self._camera, "release"):
                self._camera.release()
