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
        self._frame_scale = 1/3
        self._canny_frame = False
        self._canny_lower = 50
        self._canny_upper = 150
        self._houghLP = False

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
        self._dl.start()
        self.steering_angle = 90
        self.direction = 1
        self._houghLP = False

        while self._active:

            test = self.cam.get_frame()
            canny = pf.preprocess_frame(test, self._frame_scale, self._canny_lower, self._canny_upper)
            #houghes, pl, pr, pm = cl.get_lines(canny)

            if self._canny_frame:
                height, width, _ = test.shape
                img1 = cv.resize(test, (int(width*self._frame_scale), int(height*self._frame_scale)), interpolation = cv.INTER_CUBIC)
                img2 = cv.cvtColor(canny, cv.COLOR_GRAY2RGB)
                test = np.concatenate([img1, img2], axis=0)
            if self._houghLP:
                height, width, _ = test.shape
                img1 = cv.resize(test, (int(width*self._frame_scale), int(height*self._frame_scale)), interpolation = cv.INTER_CUBIC)
                test = np.concatenate([img1, houghes], axis=0)

            self._lineframe = test

            self._dl.append(self.drive_data)
            time.sleep(0.1)

        self._lineframe = None
        self.stop()
        self._dl.save()
            

if __name__ == "__main__":
    car = CamCar()
    car.testfahrt()