"""
Model.py — Team SafeNSound (Sahil Sharma, Yash Agarwal)
TraffIQ Round 2 Finals

Hybrid autonomous navigation:
  - OpenCV wall-following with corner detection (white walls, black floor)
  - HSV-based obstacle detection (traffic light, stop sign, box)
  - TFLite CNN as fallback for outdoor/simulator tracks
"""

from model_base import BaseModel
import numpy as np
import cv2
import os

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter


class Model(BaseModel):

    # ── Tuneable constants ──────────────────────────────────────────────
    WALL_THRESH      = 170   # grayscale threshold for white walls
    WALL_AHEAD_RATIO = 0.70  # above this → wall blocking path ahead
    OPENING_MIN      = 0.15  # min dark ratio to count as a gap
    STEER_GAIN       = 1.5   # multiplier for centering error
    BASE_SPEED       = 0.40  # cruise speed on straights
    TURN_SPEED       = 0.25  # speed during mild turns (|steer| > 0.3)
    CORNER_SPEED     = 0.15  # speed during hard turns (|steer| > 0.6)
    RED_MIN_AREA     = 150   # min contour area for red light
    GREEN_MIN_AREA   = 150   # min contour area for green light
    STOP_MIN_AREA    = 300   # min contour area for stop sign
    BOX_MIN_AREA     = 400    # min contour area for box obstacle
    # ────────────────────────────────────────────────────────────────────

    def load(self):
        self.interpreter = Interpreter(
            model_path=os.path.join("participant", "model.tflite"),
            num_threads=4,
        )
        self.interpreter.allocate_tensors()

        inp = self.interpreter.get_input_details()[0]
        self.input_idx   = inp["index"]
        self.is_quantized = inp["dtype"] == np.uint8
        if self.is_quantized:
            self.input_scale, self.input_zp = inp["quantization"]
        self.output_details = self.interpreter.get_output_details()

        self.stopped_for_red  = False
        self.stop_sign_seen   = False
        self.stop_sign_frames = 0
        self.frame_count      = 0

        # warmup inference
        self._run_model(np.zeros((1, 224, 224, 3), np.float32))

    # ── Main entry point ────────────────────────────────────────────────

    def predict(self, frame):
        self.frame_count += 1

        if self._is_indoor(frame):
            direction = self._wall_follow(frame)
            red   = self._detect_red(frame)
            green = self._detect_green(frame)
            stop  = self._detect_stop(frame)
            box   = self._detect_box(frame)
        else:
            direction = self._model_steer(frame)
            red = green = stop = False
            box = None

        direction = float(np.clip(direction, -1.0, 1.0))
        speed = self.BASE_SPEED

        # ── Traffic light logic ──
        if red and not green:
            self.stopped_for_red = True
        if green:
            self.stopped_for_red = False
        if self.stopped_for_red:
            return 0.0, 0.0

        # ── Stop sign logic ──
        if stop:
            self.stop_sign_frames += 1
            if self.stop_sign_frames > 5:
                self.stop_sign_seen = True
        elif self.stop_sign_frames < 3:
            self.stop_sign_frames = 0

        if self.stop_sign_seen:
            return 0.0, 0.0
        if self.stop_sign_frames > 0:
            speed = 0.20

        # ── Box avoidance (steer wide for clearance) ──
        if box == "center":
            speed, direction = 0.15, -0.8
        elif box == "left":
            speed, direction = 0.15, max(direction, 0.8)
        elif box == "right":
            speed, direction = 0.15, min(direction, -0.8)

        # ── Speed scaling for turns ──
        if abs(direction) > 0.6:
            speed = min(speed, self.CORNER_SPEED)
        elif abs(direction) > 0.3:
            speed = min(speed, self.TURN_SPEED)

        return float(speed), float(direction)

    # ── Environment detection ───────────────────────────────────────────

    def _is_indoor(self, frame):
        """Indoor track = black floor (bottom 30% is mostly dark)."""
        h = frame.shape[0]
        gray = cv2.cvtColor(frame[int(h * 0.7):], cv2.COLOR_RGB2GRAY)
        return float(np.sum(gray < 60)) / gray.size > 0.3

    # ── Wall following ──────────────────────────────────────────────────

    def _wall_follow(self, frame):
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.25):int(h * 0.75)]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, self.WALL_THRESH, 255, cv2.THRESH_BINARY)
        rh, rw = mask.shape

        # Phase 1: is there a wall blocking the path ahead?
        upper = mask[:rh // 2, rw // 4:rw * 3 // 4]
        if float(np.sum(upper > 128)) / upper.size >= self.WALL_AHEAD_RATIO:
            return self._find_opening(mask, rh, rw)

        # Phase 2: open road — stay centred between walls
        return self._centre_between_walls(mask, rh, rw)

    def _find_opening(self, mask, rh, rw):
        """Wall ahead — steer toward the side with most floor visible."""
        left  = float(np.sum(mask[:, :rw // 4] < 128)) / (rh * rw // 4)
        right = float(np.sum(mask[:, rw * 3 // 4:] < 128)) / (rh * rw // 4)

        if right > left and right > self.OPENING_MIN:
            return 1.0
        if left > self.OPENING_MIN:
            return -1.0
        return 1.0 if right >= left else -1.0

    def _centre_between_walls(self, mask, rh, rw):
        """Scan horizontal rows to find wall edges, steer to gap centre."""
        errors = []
        mid = rw // 2
        for frac in (0.3, 0.5, 0.7):
            row = mask[int(rh * frac)]
            whites = np.where(row > 128)[0]
            if len(whites) < 5:
                continue
            lw = whites[whites < mid]
            rw_cols = whites[whites >= mid]
            if len(lw) == 0 and len(rw_cols) == 0:
                continue
            left_edge  = lw[-1]       if len(lw) > 0      else 0
            right_edge = rw_cols[0]   if len(rw_cols) > 0  else rw - 1
            gap_centre = (left_edge + right_edge) / 2.0
            errors.append((gap_centre - rw / 2.0) / (rw / 2.0))

        if not errors:
            return 0.0
        return float(np.clip(np.mean(errors) * self.STEER_GAIN, -1.0, 1.0))

    # ── Traffic light detection ─────────────────────────────────────────

    def _detect_red(self, frame):
        roi = frame[:int(frame.shape[0] * 0.55)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mask = (cv2.inRange(hsv, (0, 120, 120), (10, 255, 255))
              | cv2.inRange(hsv, (165, 120, 120), (180, 255, 255)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return self._has_circular_contour(mask, self.RED_MIN_AREA)

    def _detect_green(self, frame):
        roi = frame[:int(frame.shape[0] * 0.55)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (35, 100, 100), (85, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return self._has_circular_contour(mask, self.GREEN_MIN_AREA)

    @staticmethod
    def _has_circular_contour(mask, min_area):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            a = cv2.contourArea(c)
            p = cv2.arcLength(c, True)
            if a >= min_area and p > 0 and 4 * np.pi * a / (p * p) > 0.5:
                return True
        return False

    # ── Stop sign detection ─────────────────────────────────────────────

    def _detect_stop(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = (cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
              | cv2.inRange(hsv, (165, 100, 100), (180, 255, 255)))
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            a = cv2.contourArea(c)
            if a < self.STOP_MIN_AREA:
                continue
            verts = len(cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True))
            if 6 <= verts <= 12 and a > 400:
                return True
        return False

    # ── Box obstacle detection ──────────────────────────────────────────

    def _detect_box(self, frame):
        h, w = frame.shape[:2]
        mx = int(w * 0.25)
        roi = frame[int(h * 0.15):int(h * 0.85), mx:w - mx]
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mask = (cv2.inRange(hsv, (0, 40, 60), (180, 255, 220))
              | cv2.inRange(hsv, (10, 30, 60), (30, 200, 200)))
        k = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rw = roi.shape[1]
        for c in contours:
            if cv2.contourArea(c) < self.BOX_MIN_AREA:
                continue
            cx = cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] // 2
            if cx < rw * 0.35:
                return "left"
            if cx > rw * 0.65:
                return "right"
            return "center"
        return None

    # ── TFLite model inference (fallback for outdoor tracks) ────────────

    def _model_steer(self, frame):
        h, w = frame.shape[:2]
        img = cv2.resize(frame[int(h * 0.4):], (224, 224))
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        inp = np.expand_dims(img.astype(np.float32) / 127.5 - 1.0, 0)
        return float(np.clip(self._run_model(inp)[1], -1.0, 1.0))

    def _run_model(self, data):
        if self.is_quantized:
            data = (data / self.input_scale + self.input_zp).astype(np.uint8)
        else:
            data = data.astype(np.float32)
        self.interpreter.set_tensor(self.input_idx, data)
        self.interpreter.invoke()
        for d in self.output_details:
            t = self.interpreter.get_tensor(d["index"])[0]
            if t.shape == (2,):
                return t
        return np.zeros(2)
