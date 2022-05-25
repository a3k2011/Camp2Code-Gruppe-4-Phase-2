import basisklassen_cam
import basecar as bc
import preprocess_frame as pf
import compute_lines as cl
import cv2 as cv

def steering_angle(line_angle):
    """
    Umrechung Lenkwinkel
    line_angle liegt zw. 45 und 135
    line_angle = 360 ist ungültig
    """

    if line_angle != 360: 
        steering_angle_car = round(line_angle, 1)
        return steering_angle_car
    else:
        return 360

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