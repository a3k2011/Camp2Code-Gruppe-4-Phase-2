import cv2 as cv
import numpy as np
import basisklassen_cam

def resize_frame(frame, resize_faktor):
    """XXX"""
    height, width, _ = frame.shape
    frame = cv.resize(frame, (int(width*resize_faktor), int(height*resize_faktor)), interpolation = cv.INTER_CUBIC)
    return frame

def change_color_bgr2gray(frame):
    """XXX"""
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return frame

def edge_detection(frame, low_border, upper_border):
    """XXX"""
    frame = cv.Canny(frame, low_border, upper_border)
    return frame

def crop_roi(frame):
    """XXX"""
    height, _ = frame.shape
    lower_border = int(height*0.40)
    upper_border = int(height*0.85)
    
    frame = frame[lower_border:upper_border, :]
    return frame

def preprocess_frame(raw_frame, resize_faktor=1/1, canny_lower=50, canny_upper= 150):
    """XXX"""
    frame = np.copy(raw_frame)
    frame = resize_frame(frame, resize_faktor=resize_faktor)
    frame = change_color_bgr2gray(frame)
    frame = edge_detection(frame, low_border=canny_lower, upper_border=canny_upper)
    frame = crop_roi(frame)

    return frame

if __name__ == "__main__":
    cam = basisklassen_cam.Camera()
    testbild = cam.get_frame()
    print(testbild.shape)

    cv.imshow("Originalbild", testbild)
    cv.imshow("neues Testbild", preprocess_frame(testbild))

    cv.waitKey(0)
    cv.destroyAllWindows()    
