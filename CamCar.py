import time, os.path
from datetime import datetime
import os
import glob
import re
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
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


class CamCar(basecar.BaseCar):
    """Die Klasse CamCar fuegt die Funktion der Kamera zur BaseCar-Klasse hinzu.

    Args:
        BaseCar (_type_): Erbt von der Klasse BaseCar.
    """

    def __init__(self):
        """Initialisierung der Klasse CamCar.
        """
        super().__init__()
        self.cam = basisklassen_cam.Camera(skip_frame=2, thread=False, fps=30)
        self._dl = datenlogger.Datenlogger(log_file_path="Logger")
        self._active = False
        self._img_logging = False
        self._result_frame = None
        self._canny_frame = False
        self._houghLP_frame = False
        self._folder = ""
        self._create_img_logger_path()
        self._cnn_model = None
        self._lmodel = None

        try: # Parameter aus config.json laden.
            with open("config.json", "rt")as f:
                params = json.load(f)
                self._frame_scale = params["scale"]
                self._hsv_lower = params["hsv lower"]
                self._hsv_upper = params["hsv upper"]
                self._frame_blur = params["blur"]
                self._frame_dilation = params["dilation"]
                self._canny_lower = params["canny lower"]
                self._canny_upper = params["canny upper"]
                self._houghes_threshold = params["hough threshold"]
                self._houghes_minLineLength = params["hough min line length"]
                self._houghes_maxLineGap = params["hough max line gap"]
        except: # Dateien nicht vorhanden oder Werte nicht enthalten.
            self._frame_scale = 1
            self._hsv_lower = 0
            self._hsv_upper = 360
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
                data["hsv lower"] = self._hsv_lower
                data["hsv upper"] = self._hsv_upper
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

            # start = time.perf_counter()

            raw_frame = self.cam.get_frame()
            fixed_scale = self._frame_scale

            canny_frame, roi = pf.preprocess_frame(raw_frame, fixed_scale, self._hsv_lower, self._hsv_upper, self._frame_blur, self._frame_dilation, self._canny_lower, self._canny_upper)
            houghes_frame, line_angle = cl.get_lines(canny_frame, self._houghes_threshold, self._houghes_minLineLength, self._houghes_maxLineGap)

            steering_angle = st.steering_angle(line_angle)
            if steering_angle != 360:
                self.steering_angle = steering_angle

            self.build_dash_cam_view(fixed_scale, raw_frame, canny_frame, houghes_frame)

            if self._img_logging:
                self.save_img(roi, steering_angle)

            # print(time.perf_counter()-start)
            
        self._end_drive_mode

    def fp_deepnn(self, v=None, tflite=False):
        """Funktion zur Ausfuerung des Fahrparcours auf Basis DeepNN.
        """
        
        if self._cnn_model != None:

            # Keras
            input_shape = self._cnn_model.layers[0].input_shape
            for layer in self._cnn_model.layers:
                layer.trainable = False

            if tflite:
                # TF-Lite
                lmodel = LiteModel.from_keras_model(self._cnn_model)
                print('TF-Lite erzeugt!')
            
            # Starte Drive-Mode
            self._start_drive_mode(v)

            while self._active:

                # start = time.perf_counter()

                raw_frame = self.cam.get_frame()
                roi, img = pf.preprocess_frame_cnn(raw_frame, 1, input_shape)

                if tflite:
                    y_pred = lmodel.predict_single(img[0])
                else:
                    y_pred = self._cnn_model(img).numpy()

                steering_angle = st.steering_angle_deepnn(y_pred)

                if steering_angle != 360:
                    self.steering_angle = steering_angle

                self._result_frame = np.concatenate([raw_frame, roi], axis=0)

                # print(time.perf_counter()-start)

        self._end_drive_mode

class LiteModel():
    
    @classmethod
    def from_keras_model(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        converter.experimental_new_converter=False
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))
    
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]
    
    def predict_single(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out