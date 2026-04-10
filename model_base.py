# model_base.py
# ─────────────────────────────────────────────────────────────────────────────
# Abstract base class for all participant models.
# Participants import this, subclass it, and implement load() and predict().
# This file is provided by the organiser — do not modify.
# ─────────────────────────────────────────────────────────────────────────────

from abc import ABC, abstractmethod
import numpy as np

from config import RESOLUTION

# Numpy shape for a single frame: (height, width, channels)
# RESOLUTION is (width, height) — we reverse it for numpy
FRAME_SHAPE = (RESOLUTION[1], RESOLUTION[0], 3)   # e.g. (480, 640, 3)


class BaseModel(ABC):
    """
    Base class for all participant models.

    Participants must:
      1. Subclass this class
      2. Name their class exactly `Model`
      3. Implement both load() and predict()
      4. Save their file as Model.py
      5. Place any weight files or assets in the participant/ folder

    Example
    -------
    from model_base import BaseModel
    import numpy as np

    class Model(BaseModel):

        def load(self):
            # load your weights here
            pass

        def predict(self, frame: np.ndarray) -> tuple:
            # your inference here
            return 0.5, 0.0  # (speed, direction)
    """

    @abstractmethod
    def load(self) -> None:
        """
        Called ONCE at startup before the race begins.

        Use this method to:
        - Load model weights from the participant/ folder
        - Initialise your ML framework (TFLite, ONNX Runtime, etc.)
        - Allocate any tensors or buffers

        Example paths for weight files:
            "participant/model.tflite"
            "participant/model.onnx"

        This method must complete within 60 seconds.
        Raise an exception if initialisation fails — the car will not start.
        """
        pass

    @abstractmethod
    def predict(self, frame: np.ndarray) -> tuple:
        """
        Called every frame during the race.

        Parameters
        ----------
        frame : np.ndarray
            Shape : (480, 640, 3)  — or whatever RESOLUTION is set to: (height, width, 3)
            Dtype : uint8
            Colour: RGB  — NOT BGR. If you use OpenCV internally, be aware it
                          defaults to BGR; convert with cv2.cvtColor if needed.

        Returns
        -------
        (speed, direction) : tuple of two floats
            Both must be Python floats (or numpy scalar) in the range [-1.0, 1.0].

            speed:
                 1.0  = full forward  (remapped to PWM_MAX internally)
                 0.0  = stop
                -1.0  = full reverse  (only active if REVERSE_ALLOWED = True)

            direction:
                 1.0  = full right
                 0.0  = straight
                -1.0  = full left

        Notes
        -----
        - This method must return within 100ms. Exceeding this triggers a safety stop.
        - Values outside [-1.0, 1.0] are clamped — they won't cause an error,
          but values at or near the limits are already maximum output.
        - Return a plain Python tuple: return speed, direction
        """
        pass
