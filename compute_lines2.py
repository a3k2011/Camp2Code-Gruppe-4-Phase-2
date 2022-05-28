import cv2 as cv
import numpy as np
from math import *
import sys 


angle_invalid = 360

def intercept_point(line1, line2):    
    """calculates the intercept point between 2 parametrized lines

    Args:
        line1
     (_type_): [m,t]
        line2
     (_type_): [m,t]

    Returns:
        _type_: (x,y)
    """
    ml, tl = line1
    mr, tr = line2

    xp=int((tl-tr)/(mr-ml))
    yp = int(mr*xp+tr)
    return (xp, yp)

def line_pt_from_line_nf(line_nf, image):
    """calculates points at the border of the image to generate lines in full y-scale

    Args:
        line_nf (_type_): [m,t]
        image (_type_): image to draw lines on

    Returns:
        _type_: [x1,y1,x2,y2]
    """
    #was führt hier zur exception???
    height = image.shape[0]
    m,t = line_nf
    y1 = height
    y2 = 0
    if m==np.inf:
        x1 = x2 = t
    else:
        x1 = int((y1-t)/m)
        x2 = int((y2-t)/m)
    return [x1, y1, x2, y2]
    

def line_nf_from_line_pt(line):
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

def line_calc(image, lines, offset, color):
    lines_nf = []
    lines_nf.clear()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x1 = x1+ offset
        x2 = x2+ offset
        line_nf = (line_nf_from_line_pt([x1, y1, x2, y2]))
        if line_nf[0] != np.infty:
            if (abs(line_nf[0])> 0.25): # Steigung: horizontale Linien werden ausgeblendet
                lines_nf.append(line_nf) 
                    # einzeichnen der selektierten Linien
                image = cv.line(image, (x1, y1),(x2, y2), color,3)
    return image, lines_nf

def average_lines(lines_pt_l, lines_pt_r, image):
    """sorts lines according to the slope of the lines (positive / negative)
    horizontal lines are filtered out

    Args:
        lines_pt_l (_type_): list of lines from left half
        lines_pt_r (_type_): list of lines from right half
        image (_type_): image containing the lines to draw the lines in

    Returns:
        _type_: [left_average_line, right_average_line, middle_line],slope_of_middle_line
    """
    height, width, _ = image.shape
    image, lines_nf_l = line_calc(image, lines_pt_l, 0, (255,0,255))
    image, lines_nf_r = line_calc(image, lines_pt_r, width//2, (0,0,255))
    
    if len(lines_nf_l)>0 and len(lines_nf_r)>0:
        line_left_av_nf = np.average(lines_nf_l, axis=0)
        line_right_av_nf = np.average(lines_nf_r, axis=0)
        line_left_lane_middle = line_pt_from_line_nf(line_left_av_nf, image)
        line_right_lane_middle = line_pt_from_line_nf(line_right_av_nf, image)
        try:
            xl, yl, _,_ = line_left_lane_middle
            xr, _, _,_ = line_right_lane_middle
        except:
            print("Error unpacking lane middle lines")
            print("Left middle:", line_left_lane_middle)
            print("Right middle:", line_right_lane_middle)
            
            return [[[0,0,0,0], [0,0,0,0],[0,0,0,0]], angle_invalid]   
        else:
            
        
        
        # berechnen der mittleren Linie,
        # der Offset ist zum Ausgleich eines Parallelen Versatzes zu den Linien
            x_line_mid = xl+(xr-xl)//2
            x_frame_mid = width//2
            x_offset = x_line_mid - x_frame_mid
            p_mid = (x_frame_mid,height)
            p_intercept = intercept_point(line_left_av_nf,line_right_av_nf)
            
            line_middle = [p_mid[0], p_mid[1],p_intercept[0]+x_offset, p_intercept[1]]
            
            slope, _ = line_nf_from_line_pt(line_middle)
            angle = degrees(atan(slope))
            if angle <0:
                angle = 180 + angle
            return [[line_left_lane_middle, line_right_lane_middle,line_middle], angle]   
        
        
    else:
        # nicht genügend linien um etwas zu ermitteln -> Kreuz darstellen und Winkel auf 360 (ungültig)
        line_left_lane_middle = [0,height,width,0]
        line_right_lane_middle = [width,height,0,0]
        nf = line_nf_from_line_pt(line_left_lane_middle)
        nf = line_nf_from_line_pt(line_right_lane_middle)
        return [[line_left_lane_middle, line_right_lane_middle], angle_invalid] 
    
# Parameter for Hough 

rho = 1 #Pixel width of result
theta = np.pi/180 #
#threshold = 40 # Threshold: min number of sections to detect a line
#minLineLength = 70 #min 40 px length to detect a line
#maxLineGap = 30#less px gaps get closed

def masked_image(image):
    """masks a triangle in the lower middle of the image because there is not sure if the line is from left or right
    
    Args:
        image (_type_): original image

    Returns:
        _type_: image with masked out area
    """
    height, width = image.shape
    polygons = np.array([
        [(0, height),
        (width//3, height),
        (width//2, height//2),
        (width//3*2, height),
        (width, height),
        (width, 0),
        (0,0)]
    ])
    mask = np.zeros_like(image)
    cv.fillPoly(mask, polygons,255)
    masked_image = cv.bitwise_and(image, mask)
    return masked_image

def get_lines(image, threshold=40, minLineLength=70,maxLineGap=30):
    """searches for lines in an image
    eliminates horizontal lines
    calculates the middle of similar sloped lines
    
    Args:
        image (_type_): needs output of Canny-Function

    Returns:
        _type_: image, direction in degree, 90 is straight
    """
    image = masked_image(image)
    
    height, width = image.shape
    image_l = image[:, :width//2]
    image_r = image[:,width//2:]
    lines_pt_l = cv.HoughLinesP(image_l,rho,theta,threshold,None,minLineLength,maxLineGap )
    lines_pt_r = cv.HoughLinesP(image_r,rho,theta,threshold,None,minLineLength,maxLineGap )
    if (lines_pt_l is not None) and (lines_pt_r is not None) :
        text_lines = str(len(lines_pt_l)+len(lines_pt_r))+" lines found"
        frame_marked = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        """
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # einzeichnen der von Hough gefundenen Linien
            frame_marked = cv.line(frame_marked, (x1, y1),(x2, y2), (0,0,255),3)
            """
        cv.putText(frame_marked,text_lines,org=(10,20),fontFace=cv.FONT_HERSHEY_COMPLEX,fontScale=0.6,color=(255,0,0),thickness=1)
        
        marker, angle = average_lines(lines_pt_l, lines_pt_r, frame_marked)
        try:
            for i, line in enumerate(marker):
                x1, y1, x2, y2 = line
                # einzeichnen der berechneten Linien
                color = (255,0,0)
                if i==2:
                    color = (0,255,0)
                frame_marked = cv.line(frame_marked, (x1, y1),(x2, y2), color,2)
                cv.putText(frame_marked,(f"LW {angle:.1f}"),org=(10,50),fontFace=cv.FONT_HERSHEY_COMPLEX,fontScale=0.8,color=(0,255,0),thickness=1)
            return frame_marked, angle
        except:
            print(marker)
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
            cv.putText(image,"Error drawing line",org=(10,20),fontFace=cv.FONT_HERSHEY_COMPLEX,fontScale=0.8,color=(0,0,255),thickness=1)
            cv.putText(image,"LW invalid",org=(10,50),fontFace=cv.FONT_HERSHEY_COMPLEX,fontScale=0.8,color=(0,255,0),thickness=1)
            return image, angle_invalid
    else:
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        cv.putText(image,"No lines found",org=(10,20),fontFace=cv.FONT_HERSHEY_COMPLEX,fontScale=0.8,color=(0,255,0),thickness=1)
        cv.putText(image,"LW invalid", org=(10,50),fontFace=cv.FONT_HERSHEY_COMPLEX,fontScale=0.8, color=(0,255,0),thickness=1)
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