def steering_angle(line_angle):
    """Umrechung Lenkwinkel"""
    steering_angle_car = line_angle*10

    return steering_angle_car

if __name__ == "__main__":
    import CamCar as car
    print("TESTUMGEBUNG")