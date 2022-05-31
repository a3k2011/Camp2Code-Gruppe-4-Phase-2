import basisklassen_cam
import basecar as bc
import preprocess_frame as pf
import compute_lines as cl
import cv2 as cv

def steering_angle(line_angle):
    """
    Umrechung Lenkwinkel
    line_angle_min & line_angle_max ==> Grenzwinkel, den die Funktion get_lines stabil erkennt
    car_angle_min & car_angle_max ==> Grenzwinkel, die das Auto fahren kann
    """
    
    line_angle_min = 65
    line_angle_max = 115
    
    car_angle_min = 50
    car_angle_max = 130
    
    m = ((car_angle_max - car_angle_min)/(line_angle_max - line_angle_min))
    n = car_angle_min - m*line_angle_min
    
    #print("m: ",m, "   n:",n)

    if line_angle >= line_angle_min or line_angle <= line_angle_max: 
        steering_angle_car = ((m*line_angle)+n)
        steering_angle_car = round(steering_angle_car, 1)
        return steering_angle_car
    else:
        return 360

if __name__ == "__main__":
    cam = basisklassen_cam.Camera()
    car = bc.BaseCar()

    testbild = cam.get_frame()
    testbild , _ = pf.preprocess_frame(testbild, 0.5, 5, 5, 175, 175) #Parameter ggf. anpassen
    #print(testbild.shape)
    img, line_angle = cl.get_lines(testbild)

    cv.imshow("Testbild", img)
    print("TESTUMGEBUNG")
    print("Linienwinkel = ", line_angle)
    print("Lenkwinkel = ", steering_angle(line_angle))
    print()

    car.steering_angle = steering_angle(line_angle)

    cv.waitKey(0)
    cv.destroyAllWindows()