import time, os.path
from datetime import datetime
import json
import numpy as np
import cv2 as cv
import basecar
import basisklassen_cam
import datenlogger
import preprocess_frame as pf
import compute_lines as cl
import steering as st
import uuid


class CamCar(basecar.BaseCar):
    """Die Klasse CamCar fuegt die Funktion der Kamera zur BaseCar-Klasse hinzu.

    Args:
        BaseCar (_type_): Erbt von der Klasse BaseCar.
    """

    def __init__(self):
        """Initialisierung der Klasse CamCar.
        """
        super().__init__()
        self.cam = basisklassen_cam.Camera(thread=False, fps=30)
        self._dl = datenlogger.Datenlogger(log_file_path="Logger")
        self._active = False
        self._result_frame = None
        self._canny_frame = False
        self._houghLP_frame = False
        self._folder = ""
        self._create_img_logger_path()

        try: # Parameter aus config.json laden.
            with open("config.json", "rt")as f:
                params = json.load(f)
                self._frame_scale = params["scale"]
                self._frame_blur = params["blur"]
                self._frame_dilation = params["dilation"]
                self._canny_lower = params["canny lower"]
                self._canny_upper = params["canny upper"]
                self._houghes_threshold = params["hough threshold"]
                self._houghes_minLineLength = params["hough min line length"]
                self._houghes_maxLineGap = params["hough max line gap"]
        except: # Dateien nicht vorhanden oder Werte nicht enthalten.
            self._frame_scale = 1
            self._frame_blur = 1
            self._frame_dilation = 2
            self._canny_lower = 50
            self._canny_upper = 125
            self._houghes_threshold = 40
            self._houghes_minLineLength = 70
            self._houghes_maxLineGap = 30

    def _create_img_logger_path(self):
        """Funktion erstellt Ordner IMG_Logger.
        """
        if not os.path.exists('IMG_Logger'):
            os.makedirs('IMG_Logger')

    def _create_img_logger_folder(self):
        """Funktion erstellt Unterordner im IMG_Logger.
        """
        nowTimestamp = str(datetime.now())
        strMainFolder = 'IMG_Logger//'

        self._folder = strMainFolder + (nowTimestamp.replace("-", "").replace(":", "").replace(" ", "_")[:15])
        os.mkdir(self._folder)

    def _start_drive_mode(self, v=None):
        """Funktion zum Starten des Fahrzustandes.
        """
        self._create_img_logger_folder()
        self._active = True
        self.steering_angle = 90

        if v != None:
            self.drive(v)

    @property
    def _end_drive_mode(self):
        """Funktion zum Beenden des Fahrzustandes.
        """
        self._result_frame = None
        self.stop()
        self.steering_angle = 90

    @property
    def drive_data(self):
        """Ausgabe der Fahrdaten fuer den Datenlogger.

        Returns:
            [list]: speed, direction, steering_angle
        """
        return [self.speed, self.direction, self.steering_angle]

    def save_img(self, frame, angle):
        """Funktion zur Speicherung von Trainingsbildern.

        Returns:
            [.jpg]: JPG-File mit Lenkwinkel
        """
        img_id = str(uuid.uuid4())
        filename = str(angle) + "_id_" + img_id + ".jpg"
        filepath = os.path.join(self._folder, filename)
        cv.imwrite(filepath, frame)

    def save_parameters(self):
        """Funktion zur Speicherung der Paramter aus dem Dashboard.

        Returns:
            [config.json]: JSON-File mit Einstellungen zum PiCar
        """
        data = None
        try:
            with open("config.json", "r") as f:
                data = json.load(f)
            with open("config.json", "w") as f:
                data["scale"] = self._frame_scale
                data["blur"] = self._frame_blur
                data["dilation"] = self._frame_dilation
                data["canny lower"] = self._canny_lower
                data["canny upper"] = self._canny_upper
                data["hough threshold"] = self._houghes_threshold
                data["hough min line length"] = self._houghes_minLineLength
                data["hough max line gap"] = self._houghes_maxLineGap
                json.dump(data, f, indent="    ")
            print("Parameters saved to config.json")
        except:
            print("config.json File Error")
        
    def get_image_bytes(self):
        """Generator for the images from the camera for the live view in dash

        Yields:
            [bytes]: Bytes string with the image information
        """
        while True:
            jepg = self.cam.get_jpeg(self._result_frame)

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jepg + b'\r\n\r\n')
            time.sleep(0.1)

    def build_dash_cam_view(self, frame_scale, raw_frame, canny_frame, houghes_frame):
        """Funktion zur Erzeugung des Live-Views im Dashboard.
        """
        result_frame = pf.resize_frame(raw_frame, frame_scale)

        if self._canny_frame:
            canny_rgb_frame = cv.cvtColor(canny_frame, cv.COLOR_GRAY2RGB)
            result_frame = np.concatenate([result_frame, canny_rgb_frame], axis=0)

        if self._houghLP_frame:
            result_frame = np.concatenate([result_frame, houghes_frame], axis=0)

        self._result_frame = result_frame

    def fp_opencv(self, v=None):
        """Funktion zur Ausfuerung des Fahrparcours auf Basis OpenCV.
        """
        self._start_drive_mode(v)

        while self._active:

            start = time.perf_counter()

            raw_frame = self.cam.get_frame()
            fixed_scale = self._frame_scale

            canny_frame, roi = pf.preprocess_frame(raw_frame, fixed_scale, self._frame_blur, self._frame_dilation, self._canny_lower, self._canny_upper)
            houghes_frame, line_angle = cl.get_lines(canny_frame, self._houghes_threshold, self._houghes_minLineLength, self._houghes_maxLineGap)

            steering_angle = st.steering_angle(line_angle)
            if steering_angle != 360:
                self.steering_angle = steering_angle

            self.build_dash_cam_view(fixed_scale, raw_frame, canny_frame, houghes_frame)
            self.save_img(roi, steering_angle)

            print(time.perf_counter()-start)
            
        self._end_drive_mode

    def fp_deepnn(self, v=None):
        """Funktion zur Ausfuerung des Fahrparcours auf Basis DeepNN.
        """
        self._start_drive_mode(v)

        time.sleep(1)
        self._active = False

        self._end_drive_mode
    

if __name__ == "__main__":
    car = CamCar()