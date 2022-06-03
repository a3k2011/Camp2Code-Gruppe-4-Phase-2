import cv2 as cv
import numpy as np
import basisklassen_cam


def resize_frame(frame, resize_faktor):
    """Funktion zur Anpassung der Bildgroesse.
    
    Args:
        [resize_faktor]: Faktor zur Anpassung des Bildes
    """
    if resize_faktor != 1:
        height, width, _ = frame.shape
        frame = cv.resize(frame, (int(width*resize_faktor), int(height*resize_faktor)), interpolation = cv.INTER_CUBIC)
    return frame

def crop_roi(frame):
    """Funktion zur Bildung der Region of Interest.
    """
    height = frame.shape[0]
    lower_border = int(height*0.40)
    upper_border = int(height*0.85)
    
    return frame[lower_border:upper_border, :]

def filter_hsv(frame, hsv_lower, hsv_upper):
    """Funktion zur Anwendung des HSV-Filters.

    Args:
        [hsv_lower]: Untere Grenze des HSV-Filters
        [hsv_upper]: Obere Grenze des HSV-Filters
    """
    if hsv_lower != 0 or hsv_upper != 360:
        hsv_lower_ar = np.array([hsv_lower/2, 0, 0])
        hsv_upper_ar = np.array([hsv_upper/2, 255, 255])
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        mask_hsv = cv.inRange(frame_hsv, hsv_lower_ar, hsv_upper_ar)
        frame = cv.bitwise_and(frame, frame, mask=mask_hsv)

    return frame

def blur_image(frame, repetitions_blur):
    """Funktion zum Weichzeichnen des Bildes.

    Args:
        [repetitions_blur]: Faktor zur Anpassung des Bildes
    """
    for i in range(repetitions_blur):
        frame = cv.GaussianBlur(frame, (3,3), 1)
    return frame

def change_color_bgr2gray(frame):
    """Funktion zur Anpassung des Farbraums.
    """
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

def edge_detection(frame, low_border, upper_border):
    """Funktion zur Anwendung der Canny Edge Detection.

    Args:
        [low_border]: Untere Grenze der Canny Edge Detection
        [upper_border]: Obere Grenze der Canny Edge Detection
    """
    return cv.Canny(frame, low_border, upper_border)

def dilate_image(frame, kernel_size):
    """Funktion zur Anpassung der Strichstaerke.

    Args:
        [kernel_size]: Groesse der kernel size
    """
    return cv.dilate(frame, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)

def fit_shape(frame, input_shape):
    """Funktion zur Anpassung der input shape.
    """
    if input_shape[3] == 1:
        frame = change_color_bgr2gray(frame)

    if input_shape[1] != frame.shape[0] or input_shape[2] != frame.shape[1]:
        frame = cv.resize(frame, (input_shape[2], input_shape[1]), interpolation = cv.INTER_CUBIC)
    return frame

def preprocess_frame(raw_frame, resize_faktor=1, hsv_lower=0, hsv_upper=360, repetitions_blur=1, kernel_size=3, canny_lower=50, canny_upper= 125):
    """Funktionen fuer das Preprocessing fuer das OpenCV CamCar.
    """
    frame = resize_frame(raw_frame, resize_faktor=resize_faktor)
    roi = crop_roi(frame)
    frame = filter_hsv(roi, hsv_lower=hsv_lower, hsv_upper=hsv_upper)
    frame = blur_image(frame, repetitions_blur=repetitions_blur)
    frame = change_color_bgr2gray(frame)
    frame = edge_detection(frame, low_border=canny_lower, upper_border=canny_upper)
    frame = dilate_image(frame, kernel_size=kernel_size)

    return frame, roi

def preprocess_frame_cnn(raw_frame, resize_faktor=1, input_shape=(None, 216, 640, 3)):
    """Funktion fuer das Preprocessing fuer das DeepNN CamCar.
    """
    frame = resize_frame(raw_frame, resize_faktor=resize_faktor)
    roi = crop_roi(frame)
    frame = fit_shape(roi, input_shape=input_shape)
    frame = (frame/255).astype('float32')
    frame = frame.reshape(1, input_shape[1], input_shape[2], input_shape[3])

    return roi, frame

if __name__ == "__main__":
    """Main-Programm: Testfunktionen"""
    cam = basisklassen_cam.Camera()
    testbild = cam.get_frame()
    print(testbild.shape)

    cv.imshow("Originalbild", testbild)
    cv.imshow("Blurred Image", blur_image(testbild))
    cv.imshow("Preprocessed Image", preprocess_frame(testbild))

    cv.waitKey(0)
    cv.destroyAllWindows()