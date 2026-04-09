#TFLite inference engine for running on the Pi.

import time
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite


class TFLiteEngine:
    def __init__(self, model_path, num_threads=4):
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=num_threads,
        )
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        
        self.is_quantized = self.input_details["dtype"] == np.uint8
        if self.is_quantized:
            self.input_scale = self.input_details["quantization"][0]
            self.input_zero_point = self.input_details["quantization"][1]
        
        self._latencies = []
    
    def predict(self, preprocessed_input):
        # Run inference. Returns (nav, grid, confidence).
        t0 = time.perf_counter()
        
        if self.is_quantized:
            input_data = (preprocessed_input / self.input_scale + self.input_zero_point).astype(np.uint8)
        else:
            input_data = preprocessed_input.astype(np.float32)
        
        self.interpreter.set_tensor(self.input_details["index"], input_data)
        self.interpreter.invoke()
        
        # match outputs by shape since tflite can reorder them
        nav_output = None
        obstacle_grid = None
        for detail in self.output_details:
            tensor = self.interpreter.get_tensor(detail["index"])[0]
            if tensor.shape == (2,):
                nav_output = tensor
            elif tensor.shape == (9,):
                obstacle_grid = tensor
        
        if nav_output is None:
            nav_output = np.zeros(2)
        if obstacle_grid is None:
            obstacle_grid = np.zeros(9)
        
        confidence = float(np.mean(np.abs(obstacle_grid - 0.5)) * 2)
        
        self._latencies.append(time.perf_counter() - t0)
        return nav_output, obstacle_grid, confidence
    
    @property
    def avg_latency_ms(self):
        if not self._latencies:
            return 0.0
        return np.mean(self._latencies[-100:]) * 1000
    
    def benchmark(self, num_runs=100):
        from src.training.config import INPUT_SHAPE
        dummy = np.random.uniform(-1, 1, (1, *INPUT_SHAPE)).astype(np.float32)
        
        latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            self.predict(dummy)
            latencies.append((time.perf_counter() - t0) * 1000)
        
        return {
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "fps": float(1000 / np.mean(latencies)),
        }
