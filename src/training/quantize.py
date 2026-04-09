#Quantize model to TFLite for Pi deployment.

import os
import argparse
import numpy as np
import tensorflow as tf
from src.training import config


def representative_data_gen(num_samples=100):
    # Generate calibration data for INT8 quantization.
    import cv2
    from src.data.preprocess import preprocess
    
    img_dir = config.PROCESSED_DIR
    images = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))] if os.path.exists(img_dir) else []
    
    if not images:
        print("No processed images found, using random calibration data")
        for _ in range(num_samples):
            yield [np.random.uniform(-1, 1, (1, *config.INPUT_SHAPE)).astype(np.float32)]
        return
    
    np.random.shuffle(images)
    for name in images[:num_samples]:
        frame = cv2.imread(os.path.join(img_dir, name))
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield [preprocess(frame, add_batch_dim=True)]


def quantize_ptq(model_path, output_path=None):
    # Post-training quantization to INT8.
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    
    tflite_model = converter.convert()
    
    output_path = output_path or model_path.replace(".keras", "_int8.tflite")
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"INT8 model saved: {output_path} ({size_mb:.2f} MB)")
    return output_path


def quantize_fp16(model_path, output_path=None):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    output_path = output_path or model_path.replace(".keras", "_fp16.tflite")
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"FP16 model saved: {output_path} ({size_mb:.2f} MB)")
    return output_path


def validate_quantization(keras_path, tflite_path, num_samples=200):
    # Compare FP32 vs INT8 predictions to check for drift.
    keras_model = tf.keras.models.load_model(keras_path)
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    keras_preds, tflite_preds = [], []
    
    for sample in representative_data_gen(num_samples):
        input_data = sample[0]
        keras_out = keras_model.predict(input_data, verbose=0)
        
        if input_details[0]["dtype"] == np.uint8:
            scale, zp = input_details[0]["quantization"]
            tfl_input = (input_data / scale + zp).astype(np.uint8)
        else:
            tfl_input = input_data
        
        interpreter.set_tensor(input_details[0]["index"], tfl_input)
        interpreter.invoke()
        
        # match nav output by shape (tflite can reorder outputs)
        tfl_nav = None
        for detail in output_details:
            tensor = interpreter.get_tensor(detail["index"])
            if tensor.shape[-1] == 2:
                tfl_nav = tensor
                break
        if tfl_nav is None:
            tfl_nav = interpreter.get_tensor(output_details[0]["index"])
        
        keras_preds.append(keras_out[0][0])
        tflite_preds.append(tfl_nav[0])
    
    keras_preds = np.array(keras_preds)
    tflite_preds = np.array(tflite_preds)
    
    mad = np.mean(np.abs(keras_preds - tflite_preds))
    max_diff = np.max(np.abs(keras_preds - tflite_preds))
    passed = mad < config.QUANT_ACCURACY_THRESHOLD
    
    print(f"\nQuantization validation:")
    print(f"  Mean diff: {mad:.4f}")
    print(f"  Max diff:  {max_diff:.4f}")
    print(f"  Threshold: {config.QUANT_ACCURACY_THRESHOLD}")
    print(f"  Result:    {'PASS' if passed else 'FAIL - try MobileNetV2 or QAT'}")
    
    return {"mean_diff": float(mad), "max_diff": float(max_diff), "passed": passed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--method", type=str, default="ptq", choices=["ptq", "fp16"])
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    
    if args.method == "ptq":
        tflite_path = quantize_ptq(args.model)
    else:
        tflite_path = quantize_fp16(args.model)
    
    if args.validate:
        validate_quantization(args.model, tflite_path)
