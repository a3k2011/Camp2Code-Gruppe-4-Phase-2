import basisklassen_cam
import basecar as bc
import preprocess_frame as pf
import compute_lines as cl
import cv2 as cv

def check_boundaries(steering_angle):
    """Pruefung des linken und rechten Anschlaegs"""

    if steering_angle != 360:
        if steering_angle < 50:
            steering_angle = 50
        elif steering_angle > 130:
            steering_angle = 130

    return steering_angle

def steering_angle(line_angle):
    """Umrechung des Lenkwinkels anhand HoughesLineP Line Angle.

    Returns:
            [int]: Lenkwinkel
    """
    #line_angle = check_boundaries(line_angle)

    if line_angle != 360: 
        steering_angle_car = round(line_angle, 1)
        return steering_angle_car
    else:
        return 360

def steering_angle_deepnn(y_pred):
    """Umrechung des Lenkwinkels anhand CNN y_pred.

    Returns:
            [int]: Lenkwinkel
    """
    y_pred = y_pred[0][0]

    #steering_angle_car = check_boundaries(y_pred)
    steering_angle_car = round(steering_angle_car, 1)

    return steering_angle_car

if __name__ == "__main__":
    cam = basisklassen_cam.Camera()
    car = bc.BaseCar()

    testbild = cam.get_frame()
    testbild = pf.preprocess_frame(testbild, 0.5, 255, 255) #Parameter ggf. anpassen
    img, line_angle = cl.get_lines(testbild)

    cv.imshow("Testbild", img)
    print("TESTUMGEBUNG")
    print("Linienwinkel = ", line_angle)
    print("Lenkwinkel = ", steering_angle(line_angle))
    print()

    car.steering_angle = steering_angle(line_angle)

    cv.waitKey(0)
    cv.destroyAllWindows()