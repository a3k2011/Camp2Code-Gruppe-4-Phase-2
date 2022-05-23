import time, os.path
import json
import uuid
import basecar
from basisklassen_cam import Camera
import datenlogger

class CamCar(basecar.BaseCar):
    """Die Klasse CamCar fuegt die Funktion der Kamera zur BaseCar-Klasse hinzu.

    Args:
        BaseCar (_type_): Erbt von der Klasse BaseCar.
    """

    def __init__(self):
        """Initialisierung der Klasse CamCar."""

        super().__init__()
        self.cam = Camera()
        self._dl = datenlogger.Datenlogger(log_file_path="Logger")

    @property
    def drive_data(self):
        """Ausgabe der Fahrdaten fuer den Datenlogger.

        Returns:
            [list]: speed, direction, steering_angle
        """
        return [self.speed, self.direction, self.steering_angle]

    def testfahrt(self, v):
        """Funktion zur Ausfuerung einer Testfahrt."""
        self._active = True
        self._dl.start()
        self.drive(v)

        while self._active:
            self._dl.append(self.drive_data)
            time.sleep(0.1)

        self.stop()
        self._dl.save()

    def get_camera(self):

        if not os.path.exists(os.path.join(os.getcwd(), "images")):
            os.makedirs(os.path.join(os.getcwd(), "images"))

        while True:
            frame = self.cam.get_frame()
            jepg = self.cam.get_jpeg(frame)

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jepg + b'\r\n\r\n')
            time.sleep(0.1)

            
if __name__ == "__main__":
    camcar = CamCar()
    camcar.testfahrt()