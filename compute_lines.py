import cv2 as cv
import numpy as np
from math import *
import sys 


angle_invalid = 360

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
    try:
        m,t = line_f
    except:
        print("Line error", line_f)
    y1 = height
    y2 = 0
    if m==np.inf:
        x1 = x2 = t
    else:
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
    if x1-x2 == 0:
        m=np.infty
        t=x1
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
    height, width, _ = image.shape
    lines_l = []
    lines_r = []
    for line in lines:
        line_f = (line_params_from_coords(line[0]))
        if line_f[0] != np.infty:
            if (line_f[0]< -1):
                lines_l.append(line_f)
            elif (line_f[0]>1):
                lines_r.append(line_f)
    ll_av = np.average(lines_l, axis=0)
    lr_av = np.average(lines_r, axis=0)
    if len(lines_l)>0 and len(lines_r)>0:        
        line_l = coords_from_line_function(ll_av, image)
        line_r = coords_from_line_function(lr_av, image)
        xl, yl, _,_ = line_l
        xr, _, _,_ = line_r
        
        # berechnen der mittleren Linie,
        # der Offset ist zum Ausgleich eines Parallelen Versatzes zu den Linien
        x_line_mid = xl+(xr-xl)//2
        x_frame_mid = width//2
        x_offset = x_line_mid - x_frame_mid
        p_mid = (x_frame_mid,height)
        p_intercept = intercept_point(ll_av, lr_av)
        
        line_middle = [p_mid[0], p_mid[1],p_intercept[0]+x_offset, p_intercept[1]]
        
        slope, _ = line_params_from_coords(line_middle)
        angle = degrees(atan(slope))
        if angle <0:
            angle = 180 + angle
        return [[line_l, line_r,line_middle], angle]   
        
        
    else:
        # nicht genügend linien um etwas zu ermitteln -> Kreuz darstellen und Winkel auf 360 (ungültig)
        line_l = [0,height,width,0]
        line_r = [width,height,0,0]
        ll_av = line_params_from_coords(line_l)
        lr_av = line_params_from_coords(line_r)
        return [[line_l, line_r], angle_invalid] 
    
# Parameter for Hough 

rho = 1 #Pixel width of result
theta = np.pi/180 #
#threshold = 40 # Threshold: min number of sections to detect a line
#minLineLength = 70 #min 40 px length to detect a line
#maxLineGap = 30#less px gaps get closed


def get_lines(image, threshold=40, minLineLength=70,maxLineGap=30):
    """searches for lines in an image
    eliminates horizontal lines
    calculates the middle of similar sloped lines
    

    Args:
        image (_type_): needs output of Canny-Function

    Returns:
        _type_: image, direction in degree, 90 is straight
    """
    #f_height, f_width = image.shape
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
            # einzeichnen der von Hough gefundenen Linien
            frame_marked = cv.line(frame_marked, (x1, y1),(x2, y2), (0,0,255),3)
        cv.putText(frame_marked,
                    text_lines, 
                    org=(10,20),
                    fontFace=cv.FONT_HERSHEY_COMPLEX, 
                    fontScale=0.6,
                    color=(255,0,0),
                    thickness=1)
        
        marker, angle = average_lines(lines, frame_marked)
        try:
            for i, line in enumerate(marker):
                x1, y1, x2, y2 = line
                # einzeichnen der berechneten Linien
                color = (255,0,0)
                if i==2:
                    color = (0,255,0)
                frame_marked = cv.line(frame_marked, (x1, y1),(x2, y2), color,2)
            
                cv.putText(frame_marked,
                    (f"LW {angle:.1f}"), 
                    org=(10,50),
                    fontFace=cv.FONT_HERSHEY_COMPLEX, 
                    fontScale=0.8,
                    color=(0,255,0),
                    thickness=1)
            return frame_marked, angle
        except:
            print(marker)
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
            cv.putText(image,
                    "Error drawing line", 
                    org=(10,20),
                    fontFace=cv.FONT_HERSHEY_COMPLEX, 
                    fontScale=0.8,
                    color=(0,0,255),
                    thickness=1)
            cv.putText(image,
                    (f"LW invalid"), 
                    org=(10,50),
                    fontFace=cv.FONT_HERSHEY_COMPLEX, 
                    fontScale=0.8,
                    color=(0,255,0),
                    thickness=1)
            return image, angle_invalid
    else:
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        cv.putText(image,
                    "No lines found", 
                    org=(10,20),
                    fontFace=cv.FONT_HERSHEY_COMPLEX, 
                    fontScale=0.8,
                    color=(0,255,0),
                    thickness=1)
        cv.putText(image,
                (f"LW invalid"), 
                org=(10,50),
                fontFace=cv.FONT_HERSHEY_COMPLEX, 
                fontScale=0.8,
                color=(0,255,0),
                thickness=1)
        return image, angle_invalid
        
def main():
    """
    zum testen der Funktion
    Aufruf mit Parameter -sb (save blur) speichert letztes Frame als Blur.jpg ab
    Aufruf mit Parameter -sc (save canny) speichert letztes Frame als Canny.jpg ab
    """
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    while True:
    # Abfrage eines Frames
        ret, frame = cap.read()
        height, width, _ = frame.shape
        #target size = 800x600
        factor = 800 / width
        frame = cv.resize(frame,(int(width*factor), int(height*factor)), interpolation = cv.INTER_CUBIC)
        height, width, _ = frame.shape
        frame = cv.flip(frame,0)
        frame = cv.flip(frame,1)
        
        roi_frame = frame[int(height*0.4):int(height*0.85),:int(width*0.9)]
        roi_gray = cv.cvtColor(roi_frame, cv.COLOR_BGR2GRAY)
        #roi_frame = cv.cvtColor(roi_frame, cv.COLOR_hs)
        blur_frame = cv.GaussianBlur(roi_gray,(5,5),1)
        canny_frame =cv.Canny(blur_frame,180,220)
        
        #Hier kommt der Call der zu testenden Funktion
        image, angle = get_lines(canny_frame, threshold=20, minLineLength=80, maxLineGap=50)
        
        txt_angle = f"LW: {angle:.2f}"
        canny_rgb = cv.cvtColor(canny_frame, cv.COLOR_GRAY2RGB)
        
        
        frame_display = np.vstack((roi_frame, canny_rgb, image))
        
        cv.imshow("Press q to quit", frame_display)
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