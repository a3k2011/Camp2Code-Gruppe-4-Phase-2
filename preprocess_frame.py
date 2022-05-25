import cv2 as cv
import numpy as np
import basisklassen_cam

def resize_frame(frame, resize_faktor):
    """XXX"""
    if resize_faktor != 1:
        height, width, _ = frame.shape
        frame = cv.resize(frame, (int(width*resize_faktor), int(height*resize_faktor)), interpolation = cv.INTER_CUBIC)
    return frame

def crop_roi(frame):
    """XXX"""
    height = frame.shape[0]
    lower_border = int(height*0.40)
    upper_border = int(height*0.85)
    
    frame = frame[lower_border:upper_border, :]
    return frame

def blur_image(frame, repetitions_blur):
    """XXX"""
    for i in range(repetitions_blur):
        frame = cv.GaussianBlur(frame, (3,3), 1)
    return frame

def change_color_bgr2gray(frame):
    """XXX"""
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return frame

def edge_detection(frame, low_border, upper_border):
    """XXX"""
    frame = cv.Canny(frame, low_border, upper_border)
    return frame

def dilate_image(frame, kernel_size):
    """XXX"""
    frame = cv.dilate(frame, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    return frame

def binary_threshold(frame):
    """XXX"""
    th, frame = cv.threshold(frame, 0, 255, cv.THRESH_BINARY)
    return frame

def preprocess_frame(raw_frame, resize_faktor=1, repetitions_blur=1, kernel_size=3, canny_lower=50, canny_upper= 125):
    """XXX"""
    frame = np.copy(raw_frame)
    frame = resize_frame(frame, resize_faktor=resize_faktor)
    frame = crop_roi(frame)
    frame = blur_image(frame, repetitions_blur=repetitions_blur)
    frame = change_color_bgr2gray(frame)
    frame = edge_detection(frame, low_border=canny_lower, upper_border=canny_upper)
    frame = dilate_image(frame, kernel_size=kernel_size)
    # frame = binary_threshold(frame)

    return frame

if __name__ == "__main__":
    """XXX"""
    cam = basisklassen_cam.Camera()
    testbild = cam.get_frame()
    print(testbild.shape)

    cv.imshow("Originalbild", testbild)
    cv.imshow("Blurred Image", blur_image(testbild))
    cv.imshow("Preprocessed Image", preprocess_frame(testbild))
    # cv.imshow("Preprocessed Image Dilate", dilate_image(preprocess_frame(testbild)))
    # cv.imshow("Preprocessed Image Dilate THRES", binary_threshold(dilate_image(preprocess_frame(testbild))))

    cv.waitKey(0)
    cv.destroyAllWindows()