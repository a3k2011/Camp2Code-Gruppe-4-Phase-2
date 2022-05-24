import time, os.path
import json
import basecar
import basisklassen_cam
import datenlogger
import preprocess_frame


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

    def testfahrt(self, v):
        """Funktion zur Ausfuerung einer Testfahrt."""
        self._active = True
        self._dl.start()
        self.steering_angle = 90
        self.drive(v)

        while self._active:

            test = self.cam.get_frame()
            self._lineframe = preprocess_frame.preprocess_frame(test)
            self._dl.append(self.drive_data)
            time.sleep(0.1)

        self._lineframe = None
        self.stop()
        self._dl.save()
            

if __name__ == "__main__":
    car = CamCar()
    car.testfahrt()