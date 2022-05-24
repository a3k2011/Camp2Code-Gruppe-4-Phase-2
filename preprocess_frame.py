import cv2 as cv
import numpy as np


def resize_frame(frame, resize_faktor):
    """XXX"""
    height, width, _ = frame.shape
    frame = cv.resize(frame, (int(width*resize_faktor), int(height*resize_faktor)), interpolation = cv.INTER_CUBIC)
    return frame

def change_color_space(frame):
    """XXX"""
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return frame

def blur_frame(frame):
    """XXX"""
    frame = cv.GaussianBlur(frame, (5,5), 0)
    return frame

def edge_detection(frame, low_border, upper_border):
    """XXX"""
    frame = cv.Canny(frame, low_border, upper_border)
    return frame

def crop_roi(frame):
    """XXX"""
    return frame

def preprocess_frame(cam_frame, resize_faktor=1/3, canny_lower=50, canny_upper= 150):
    """XXX"""
    frame = np.copy(cam_frame)
    frame = resize_frame(frame, resize_faktor=resize_faktor)
    frame = change_color_space(frame)
    frame = edge_detection(frame, low_border=canny_lower, upper_border=canny_upper)
    frame = crop_roi(frame)

    return frame