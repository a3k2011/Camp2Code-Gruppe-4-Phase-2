import cv2 as cv
import numpy as np


def coords_from_line_function(line_f, image):
    """converts from line parameters to a 2-point-line from image bottom to 90% of image height

    Args:
        line_f (_type_): [m,t]
        image (_type_): _description_

    Returns:
        _type_: [x1,y1,x2,y2]
    """
    height = image.shape[0]
    y1 = height
    y2 = int(height*0.1)
    x1 = int((y1-line_f[1])/line_f[0])
    x2 = int((y2-line_f[1])/line_f[0])
    return [x1, y1, x2, y2]

def line_function_from_points(line):
    """Converts line from 2 points in line parameters

    Args:
        line (_type_): [x1, y1, x2, y2]

    Returns:
        list: [m,t]
    """
    x1, y1, x2, y2 = line
    params = np.polyfit((x1,x2),(y1,y2),1)
    m = params[0]
    t = params[1]
    return [m, t]

def center_lines(lines, image):
    """receives lines from Hough Function and generates the centered line for left and right side
    adittionaly delivers the point left, right and front center  

    Args:
        lines (_type_): [[x1,y1,x2,y2],[x3,y3,x4,y4],...]
        image (_type_): _description_

    Returns:
        _type_: [[[left center line],[right center line]],point left, point right, point center front]
    """
    lines_l = []
    lines_l.clear()
    lines_r = []
    lines_r.clear()
    for line in lines:
        line_f = (line_function_from_points(line[0]))
        if line_f[0] < 0:
            lines_l.append(line_f)
        else:
            lines_r.append(line_f)
    if len(lines_l)>0 and len(lines_r)>0:
        ll_av = np.average(lines_l, axis=0)
        lr_av = np.average(lines_r, axis=0)
    
        ml, tl = ll_av
        mr, tr = lr_av
        
        #t1-t2 /m2-m1 = *x
        xp=int((tl-tr)/(mr-ml))
        yp = int(mr*xp+tr)
        pm = (xp,yp)
        
        xp=1.123
        yp=2.654
        
        line_l = coords_from_line_function(ll_av, image)
        x1, y1, _, _ = line_l
        pl = (x1,y1)
        line_r = coords_from_line_function(lr_av, image)
        x1, y1, _, _ = line_r
        pr = (x1,y1)
        
        return [[[line_l, line_r]],pl, pr, pm]  

    return [[[]],(0, 0) ,(0, 0) ,(0, 0)]  


# Parameter for Hough 

rho = 1 #Pixel width of result
theta = np.pi/180 #
threshold = 40 # Threshold: min number of sections to detect a line
minLineLength = 70 #min 40 px length to detect a line
maxLineGap = 30#less px gaps get closed


def get_lines(image):
    """_summary_

    Args:
        image (_type_): needs output of Canny-Function

    Returns:
        _type_: image, point left, point right, point center
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
        
        marker, pl, pr, pm = center_lines(lines, frame_marked)
        img_res = frame_marked.copy()
        for i, line in enumerate(marker[0]):
            x1, y1, x2, y2 = line
            img_res = cv.line(img_res, (x1, y1),(x2, y2), (255,0,0),2)
        
        return img_res, pl, pr, pm
    return image, (0,0),(0,0),(0,0)
        
        