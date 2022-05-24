import cv2 as cv
import numpy as np
from math import *
import sys 


def intercept_point(line1_p, line2_p):
    """calculates the intercept point between 2 parametrized lines

    Args:
        line1_p (_type_): [m,t]
        line2_p (_type_): [m,t]

    Returns:
        _type_: (x,y)
    """
    ml, tl = line1_p
    mr, tr = line2_p
    xp=int((tl-tr)/(mr-ml))
    yp = int(mr*xp+tr)
    return (xp, yp)

def coords_from_line_function(line_f, image):
    """calculates points at the border of the image to generate lines in full y-scale

    Args:
        line_f (_type_): [m,t]
        image (_type_): image to draw lines on

    Returns:
        _type_: [x1,y1,x2,y2]
    """
    height = image.shape[0]
    m,t = line_f
    y1 = height
    y2 = 0
    x1 = int((y1-t)/m)
    x2 = int((y2-t)/m)
    return [x1, y1, x2, y2]

def line_params_from_coords(line):
    """calculates parametrized line from line defined by 2 points

    Args:
        line (_type_): [x1,y1,x2,y2]

    Returns:
        _type_: [m,t]
    """
    x1, y1, x2, y2 = line
    #params = np.polyfit((x1,x2),(y1,y2),1)
    if x1-x2 == 0:
        m=np.infty
        t=0
    else:
        m= (y1-y2)/(x1-x2)
        t=y1-(m*x1) 
    return [m, t]

def average_lines(lines, image):
    """sorts lines according to the slope of the lines (positive / negative)
    horizontal lines are filtered out

    Args:
        lines (_type_): list of lines
        image (_type_): image containing the lines to draw the lines in

    Returns:
        _type_: [left_average_line, right_average_line, middle_line],slope_of_middle_line
    """
    lines_l = []
    lines_r = []
    for line in lines:
        line_f = (line_params_from_coords(line[0]))
        if line_f[0] != np.infty:
            if (line_f[0]<0) and (line_f[0]> -3):
                lines_l.append(line_f)
            elif (line_f[0]>0) and (line_f[0]<3):
                lines_r.append(line_f)
    ll_av = np.average(lines_l, axis=0)
    lr_av = np.average(lines_r, axis=0)
    
    line_l = coords_from_line_function(ll_av, image)
    line_r = coords_from_line_function(lr_av, image)
    xl, yl, _,_ = line_l
    xr, _, _,_ = line_r
    
    p_middle = (xl+(xr-xl)//2,yl)
    p_intercept = intercept_point(ll_av, lr_av)
    line_middle = [p_middle[0], p_middle[1],p_intercept[0], p_intercept[1]]
    
    line_middle_f = line_params_from_coords(line_middle)
    
    return [[line_l, line_r,line_middle], line_middle_f[0]]   
# Parameter for Hough 

rho = 1 #Pixel width of result
theta = np.pi/180 #
#threshold = 40 # Threshold: min number of sections to detect a line
#minLineLength = 70 #min 40 px length to detect a line
#maxLineGap = 30#less px gaps get closed


def get_lines(image, threshold=40, minLineLength=70,maxLineGap=30):
    """_summary_

    Args:
        image (_type_): needs output of Canny-Function

    Returns:
        _type_: image, direction in degree, 90 is straight
    """
    f_height, f_width = image.shape
    lines = cv.HoughLinesP(image, 
                           rho, 
                           theta, 
                           threshold, 
                           None, 
                           minLineLength, 
                           maxLineGap )
    if lines is not None:
        text_lines = str(len(lines))+" lines found"
        for line in lines:
            x1, y1, x2, y2 = line[0]
            frame_marked = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
            frame_marked = cv.line(frame_marked, (x1, y1),(x2, y2), (255,0,255),2)
        cv.putText(frame_marked,
                    text_lines, 
                    org=(10,20),
                    fontFace=cv.FONT_HERSHEY_COMPLEX, 
                    fontScale=0.6,
                    color=(255,0,255),
                    thickness=1)
        
        marker, middle_slope = average_lines(lines, frame_marked)
        img_res = frame_marked.copy()
        try:
            for i, line in enumerate(marker[0]):
                x1, y1, x2, y2 = line
                img_res = cv.line(img_res, (x1, y1),(x2, y2), (255,0,0),2)
            
            return img_res, degrees(atan(middle_slope))
        except:
            print("No lines found")
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
            return image, 90
    else:
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        return image, 90
        
        
def main():
    """
    zum testen der Funktion
    Aufrug mit Parameter -sb (save blur) speichert letztes Frame als Blur.jpg ab
    Aufrug mit Parameter -sc (save canny) speichert letztes Frame als Canny.jpg ab
    """
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
    # Abfrage eines Frames
        ret, frame = cap.read()
        height, width, _ = frame.shape
        frame = cv.resize(frame,(int(width*1/3), int(height*1/3)), interpolation = cv.INTER_CUBIC)
        frame = cv.flip(frame,0)
        frame = cv.flip(frame,1)
        roi_frame = frame[255:550,:750]
        roi_frame = cv.cvtColor(roi_frame, cv.COLOR_BGR2GRAY)
        #roi_frame = cv.cvtColor(roi_frame, cv.COLOR_hs)
        blur_frame = cv.GaussianBlur(roi_frame,(5,5),1)
        canny_frame =cv.Canny(blur_frame,180,220)
        
        """  lines = cv.HoughLinesP(canny_frame, rho, theta, threshold, None, minLineLength, maxLineGap )
        text_lines = str(len(lines))+" lines found"
        if len(blur_frame.shape) < 3:
            frame_marked = cv.cvtColor(roi_frame.copy(), cv.COLOR_GRAY2RGB)
        else:
            frame_marked = roi_frame.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            frame_marked = cv.line(frame_marked, (x1, y1),(x2, y2), (255,0,255),2)
        """
        image, angle = get_lines(canny_frame)
        
        print("Lenkwinkel:", angle)
        cv.imshow("Press q to quit", image)
        if cv.waitKey(20) &0xFF == ord("q"):
            if len(sys.argv)>1:
                for arg in sys.argv:
                    if arg == "-sc":
                        cv.imwrite("Canny.jpg",canny_frame)
                    elif arg == "-sb":
                        cv.imwrite("Blur.jpg",blur_frame)
            break
        # Wenn ret == TRUE, so war Abfrage erfolgreich
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
    cv.destroyAllWindows()
    cap.release()
    

if __name__ == "__main__":
    main()