import time, os.path
import json
import numpy as np
import cv2 as cv
import basecar
import basisklassen_cam
import datenlogger
import preprocess_frame as pf
import compute_lines as cl

class CamCar(basecar.BaseCar):
    """Die Klasse CamCar fuegt die Funktion der Kamera zur BaseCar-Klasse hinzu.

    Args:
        BaseCar (_type_): Erbt von der Klasse BaseCar.
    """

    def __init__(self):
        """Initialisierung der Klasse CamCar."""

        super().__init__()
        self.cam = basisklassen_cam.Camera()
        self._dl = datenlogger.Datenlogger(log_file_path="Logger")
        self._active = False
        self._lineframe = None
        self._frame_scale = 1
        self._canny_frame = False
        self._canny_lower = 50
        self._canny_upper = 125
        self._houghLP_frame = False
        self._houghes_threshold = 40
        self._houghes_minLineLength = 70
        self._houghes_maxLineGap = 30

    @property
    def drive_data(self):
        """Ausgabe der Fahrdaten fuer den Datenlogger.

        Returns:
            [list]: speed, direction, steering_angle
        """
        return [self.speed, self.direction, self.steering_angle]

    def get_image_bytes(self):
        """Generator for the images from the camera for the live view in dash

        Yields:
            bytes: Bytes string with the image information
        """
        while True:
            jepg = self.cam.get_jpeg(self._lineframe)

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jepg + b'\r\n\r\n')
            time.sleep(0.1)

    def parameter_tuning(self):
        """Funktion zur Ausfuerung einer Testfahrt."""
        self._active = True
        self.steering_angle = 90

        while self._active:

            fixed_scale = self._frame_scale

            raw_frame = self.cam.get_frame()
            scl_frame = pf.resize_frame(raw_frame, fixed_scale)
            result_frame = np.copy(scl_frame)

            canny_frame = pf.preprocess_frame(raw_frame, fixed_scale, self._canny_lower, self._canny_upper)

            try:
                houghes_frame, pl, pr, pm = cl.get_lines(canny_frame, self._houghes_threshold, self._houghes_minLineLength, self._houghes_maxLineGap)
            except:
                houghes_frame = np.copy(cv.cvtColor(canny_frame, cv.COLOR_GRAY2RGB))

            if self._canny_frame:
                canny_rgb_frame = cv.cvtColor(canny_frame, cv.COLOR_GRAY2RGB)
                result_frame = np.concatenate([result_frame, canny_rgb_frame], axis=0)

            if self._houghLP_frame:
                result_frame = np.concatenate([result_frame, houghes_frame], axis=0)

            self._lineframe = result_frame

            time.sleep(0.1)

        self._lineframe = None
        self.stop()
            

if __name__ == "__main__":
    car = CamCar()