import basisklassen_cam
import basecar as bc
import preprocess_frame as pf
import compute_lines as cl
import cv2 as cv


def steering_angle(line_angle):
    """Umrechung des Lenkwinkels anhand HoughesLineP Line Angle.

    Returns:
            [float]: Lenkwinkel
    """
    if line_angle != 360: 
        return round(line_angle, 1)
    else:
        return 360

def steering_angle_deepnn(y_pred):
    """Umrechung des Lenkwinkels anhand CNN y_pred.

    Returns:
            [float]: Lenkwinkel
    """
    return round(y_pred[0][0], 1)


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