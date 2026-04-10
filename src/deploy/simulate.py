# Simulator bridge — uses our Model class to drive the Udacity simulator
# Requires: python-socketio==4.6.1, python-engineio==3.13.2

import base64
from io import BytesIO
import numpy as np
from PIL import Image

import socketio
import eventlet
import eventlet.wsgi
from flask import Flask

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model import Model

sio = socketio.Server()
app = socketio.Middleware(sio, Flask(__name__))

model = Model()
model.load()
print("Model loaded. Waiting for simulator on port 4567...")

MAX_SPEED = 15

@sio.on('telemetry')
def telemetry(sid, data):
    try:
        if data:
            current_speed = float(data["speed"])
            img_string = data["image"]
            image = Image.open(BytesIO(base64.b64decode(img_string)))
            frame = np.asarray(image)  # already RGB from PIL

            speed, direction = model.predict(frame)

            # scale throttle based on current simulator speed
            if current_speed < 5:
                throttle = max(speed, 0.3)
            elif current_speed > MAX_SPEED:
                throttle = -0.1
            else:
                throttle = speed * 0.5

            # if model says stop, actually stop
            if speed <= 0.01:
                throttle = -0.3

            print(f"steer: {direction:+.3f} | throttle: {throttle:+.3f} | model_speed: {speed:.2f} | sim_speed: {current_speed:.1f}")

            sio.emit('steer', data={
                'steering_angle': str(direction),
                'throttle': str(throttle)
            }, skip_sid=True)
        else:
            sio.emit('manual', data={}, skip_sid=True)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback; traceback.print_exc()

@sio.on('connect')
def connect(sid, environ):
    print(f">>> Simulator connected: {sid}")
    sio.emit('steer', data={'steering_angle': '0', 'throttle': '0'}, skip_sid=True)

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
