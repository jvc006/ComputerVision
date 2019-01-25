"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2

import numpy as np

class Line:
    def __init__(self, line):
        self.line = line
        self.length = np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2)
        denominator = line[2] - line[0]
        if denominator == 0:
            self.angle = 90
        else:
            self.angle = np.arctan((line[3] - line[1])/denominator)/np.pi * 180
        self.mid = ((line[2] + line[0])/2, (line[3] + line[1])/2)


def select_three(circles) :
    circles = np.asarray(circles)
    operator = circles[0]
    r, c = operator.shape
    circles_selected = []
    for i in range(r):
        for j in range(i+1, r):
            for k in range(j+1, r):
                if operator[i][0] == operator[j][0] and operator[j][0] == operator[k][0]:
                    waitList = (operator[i], operator[j], operator[k])
                    temp_list = [operator[i][1], operator[j][1], operator[k][1]]
                    sequence = np.argsort(temp_list)
                    a, b, c = sequence[0], sequence[1], sequence[2]
                    circles_selected.append(waitList[a])
                    circles_selected.append(waitList[b])
                    circles_selected.append(waitList[c])
                    return circles_selected
                if operator[i][1] == operator[j][1] and operator[j][0] == operator[k][1]:
                    waitList = (operator[i], operator[j], operator[k])
                    temp_list = [operator[i][0], operator[j][0], operator[k][0]]
                    sequence = np.argsort(temp_list)
                    a, b, c = sequence[0], sequence[1], sequence[2]
                    circles_selected.append(waitList[a])
                    circles_selected.append(waitList[b])
                    circles_selected.append(waitList[c])
                    return circles_selected


def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """

    thresh1 = 110
    thresh2 = 60
    cannyEdges = cv2.Canny(img_in, thresh1, thresh2)

    circles = cv2.HoughCircles(cannyEdges,cv2.HOUGH_GRADIENT, 1, 20, param1=50,param2=30,minRadius=0,maxRadius=0)
    circles_selected = select_three(circles)

    for circle in circles_selected:
        column = circle[0]
        row = circle[1]
        coordinates = (column, row)
        state_pixels = img_in[int(row), int(column), :]
        # print(state_pixels)
        if state_pixels[1] == 255 and state_pixels[2] != 255 :
            state = 'green'
        if state_pixels[1] != 255 and state_pixels[2] == 255 :
            state = 'red'
        if state_pixels[1] == 255 and state_pixels[2] == 255 :
            state = 'yellow'

    column_2 = circles_selected[1][0]
    row_2 = circles_selected[1][1]
    coordinates_2 = (column_2, row_2)
    state_pixels_2 = img_in[int(row_2), int(column_2), :]

    return coordinates_2, state
    raise NotImplementedError


def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    thresh1 = 110
    thresh2 = 60
    cannyEdges = cv2.Canny(img_in, thresh1, thresh2)
    # cv2.imshow("test", cannyEdges)

    lines = cv2.HoughLinesP(cannyEdges, rho=1, theta=np.pi /90, threshold=30, minLineLength=20, maxLineGap=1)

    Line_list_60 = []
    Line_list_M60 = []
    Angle_60 = []
    Angle_M60 = []

    for line in lines:
        line =  line.flatten()
        line_instance = Line(line)

        if line_instance.angle > 35 and line_instance.angle < 85:   
            # print(line_instance.line, line_instance.angle)
            Angle_60.append(line_instance.length)
            Line_list_60.append(line_instance)

        if line_instance.angle > -85 and line_instance.angle < -35:  
            # print(line_instance.line, line_instance.angle) 
            Angle_M60.append(line_instance.length)
            Line_list_M60.append(line_instance)
        
    index = np.argsort(Angle_60)
    line1 = Line_list_60[index[-1]].line
    # cv2.line(img_in,(line1[0],line1[1]), (line1[2], line1[3]),(255, 0, 0), 3)

    index = np.argsort(Angle_M60)
    line3 = Line_list_M60[index[-1]].line
    # cv2.line(img_in,(line3[0],line3[1]), (line3[2], line3[3]),(255, 0, 0), 3)

    # cv2.show('test', img_in)
    X_60 = max(line1[0], line1[2])
    X_M60 = min(line3[0], line3[2])
    column = int ((X_60 + X_M60)/2)

    left_Y = min(line1[1], line1[3])
    mid_Y_60 = max(line1[1], line1[3])
    mid_Y_M60 = max(line3[1], line3[3])
    right_Y = min(line3[1], line3[3])
    row = int ((left_Y + (mid_Y_60+mid_Y_M60)/2 + right_Y)/3)
    coordinates = (column, row)

    pixels = img_in[row, column, :]
    if pixels[0] > 220 and pixels[1] > 220 and pixels[2] > 220 :
        # cv2.circle(img_in, coordinates, 2, (255, 0, 0), 2)
        return coordinates
    else:
        return None


    raise NotImplementedError


def RedSide(img_in,line_instance):
    r ,c , channel = img_in.shape
    col_1 = int (line_instance.mid[0] + 10)
    row_1 = int (line_instance.mid[1] + 10)
    col_2 = int (line_instance.mid[0] - 10)
    row_2 = int (line_instance.mid[1] - 10)

    if col_1 < c and row_1 < r and ((img_in[row_1, col_1, 0] <20 and img_in[row_1, col_1, 1]<20) or \
                                               (img_in[row_2, col_2, 0]<20 and img_in[row_2, col_2, 1]<20)) :
        return True
    else:
        return False
    return True

def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """

    thresh1 = 110
    thresh2 = 60
    cannyEdges = cv2.Canny(img_in, thresh1, thresh2)
    # cv2.imshow("test", cannyEdges)

    lines = cv2.HoughLinesP(cannyEdges, rho=1, theta=np.pi /90, threshold=20, minLineLength=20, maxLineGap=1)

    Line_list = []
    Angle_45 = []
    Angle_M45 = []

    for line in lines:
        line =  line.flatten()
        line_instance = Line(line)
        if line_instance.length < 500 and line_instance.angle != 0 and RedSide(img_in,line_instance):
            Line_list.append(line_instance)
            Angle_45.append(np.abs(line_instance.angle - 45))
            Angle_M45.append(np.abs(line_instance.angle + 45))

    if len(Angle_45) == 0:
        return None
        
    # index = np.argsort(Angle_45)
    # line1 = Line_list[index[0]]
    # line2 = Line_list[index[1]]

    index = np.argsort(Angle_M45)
    line1 = Line_list[index[0]]
    line2 = Line_list[index[1]]
    #Mark the line we use to determine the center
    # cv2.line(img_in,(line1.line[0],line1.line[1]), (line1.line[2], line1.line[3]),(255, 0, 0), 3)
    # cv2.line(img_in,(line2.line[0],line2.line[1]), (line2.line[2], line2.line[3]),(255, 0, 0), 3)

    column45 = int((line1.mid[0] + line2.mid[0])/2)
    row45 = int((line1.mid[1] + line2.mid[1])/2)

    # columnM45 = int((line3.mid[0] + line4.mid[0])/2)
    # rowM45 = int((line3.mid[1] + line4.mid[1])/2)

    # column = (column45 + columnM45)//2 + 1
    # row = (row45 + rowM45)//2 + 1
    coordinates = (column45, row45)

    # cv2.circle(img_in, coordinates, 2, (255, 0, 0), 2)
    # cv2.imshow('detected lines',img_in)

    return coordinates
    raise NotImplementedError

def WarnSide(img_in,line_instance):
    r ,c , channel = img_in.shape
    col_1 = int (line_instance.mid[0] + 3)
    row_1 = int (line_instance.mid[1] + 3)
    col_2 = int (line_instance.mid[0] - 3)
    row_2 = int (line_instance.mid[1] - 3)

    if col_1 < c and row_1 < r and \
            ((img_in[row_1, col_1, 0] < 20 and img_in[row_1, col_1, 1] >230 and img_in[row_1, col_1, 2]>230) or \
            (img_in[row_2, col_2, 0] < 20 and img_in[row_2, col_2, 1] >230 and img_in[row_2, col_2, 2]>230)) :
        return True
    else:
        return False
    return True

def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """

    thresh1 = 110
    thresh2 = 60
    cannyEdges = cv2.Canny(img_in, thresh1, thresh2)
    # cv2.imshow("test", cannyEdges)

    lines = cv2.HoughLinesP(cannyEdges, rho=1, theta=np.pi /180, threshold=40, minLineLength=30, maxLineGap=2)

    Line_list = []
    Angle_45 = []
    Angle_M45 = []

    for line in lines:
        line =  line.flatten()
        line_instance = Line(line)
        if line_instance.length < 500 and line_instance.angle != 0 and WarnSide(img_in,line_instance):
            Line_list.append(line_instance) 
            # cv2.line(img_in,(line[0],line[1]), (line[2], line[3]),(255, 0, 0), 2)
            Angle_45.append(np.abs(line_instance.angle - 45))
            Angle_M45.append(np.abs(line_instance.angle + 45))

    if len(Angle_45) == 0:
        return None
    if len(Angle_M45) == 0:
        return None

    index = np.argsort(Angle_45)
    line1 = Line_list[index[0]]
    line2 = Line_list[index[1]]

    index = np.argsort(Angle_M45)
    line3 = Line_list[index[0]]
    line4 = Line_list[index[1]]

    # cv2.line(img_in,(line1.line[0],line1.line[1]), (line1.line[2], line1.line[3]),(255, 0, 0), 3)
    # cv2.line(img_in,(line2.line[0],line2.line[1]), (line2.line[2], line2.line[3]),(255, 0, 0), 3)
    # cv2.line(img_in,(line3.line[0],line3.line[1]), (line3.line[2], line3.line[3]),(255, 0, 0), 3)
    # cv2.line(img_in,(line4.line[0],line4.line[1]), (line4.line[2], line4.line[3]),(255, 0, 0), 3)

    column45 = int((line1.mid[0] + line2.mid[0])/2)
    row45 = int((line1.mid[1] + line2.mid[1])/2)

    columnM45 = int((line3.mid[0] + line4.mid[0])/2)
    rowM45 = int((line3.mid[1] + line4.mid[1])/2)

    column = (column45 + columnM45)//2 + 1
    row = (row45 + rowM45)//2 + 1
    coordinates = (column, row)
    # print(img_in[row, column, :])
    # cv2.circle(img_in, coordinates, 2, (255, 0, 0), 2)
    # cv2.imshow('detected lines',img_in)

    return coordinates
    raise NotImplementedError

def ConsSide(img_in,line_instance):
    r ,c , channel = img_in.shape
    col_1 = int (line_instance.mid[0] + 10)
    row_1 = int (line_instance.mid[1] + 10)
    col_2 = int (line_instance.mid[0] - 10)
    row_2 = int (line_instance.mid[1] - 10)

    if col_1 < c and row_1 < r and \
            ((img_in[row_1, col_1, 0]<20 and img_in[row_1, col_1, 1]>108 and img_in[row_1, col_1, 1]<148 and img_in[row_1, col_1, 2]>230) or \
            (img_in[row_2, col_2, 0]<20 and img_in[row_2, col_2, 1]>108 and img_in[row_2, col_2, 1]<148 and img_in[row_2, col_2, 2]>230) ) :
        return True
    else:
        return False
    return True

def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """

    thresh1 = 110
    thresh2 = 60
    cannyEdges = cv2.Canny(img_in, thresh1, thresh2)
    # cv2.imshow("test", cannyEdges)

    lines = cv2.HoughLinesP(cannyEdges, rho=1, theta=np.pi /180, threshold=40, minLineLength=30, maxLineGap=2)

    Line_list = []
    Angle_45 = []
    Angle_M45 = []

    for line in lines:
        line =  line.flatten()
        line_instance = Line(line)
        if line_instance.length < 500 and line_instance.angle != 0 and ConsSide(img_in,line_instance):
            Line_list.append(line_instance) 
            # cv2.line(img_in,(line[0],line[1]), (line[2], line[3]),(255, 0, 0), 3)
            Angle_45.append(np.abs(line_instance.angle - 45))
            Angle_M45.append(np.abs(line_instance.angle + 45))

    if len(Angle_45) == 0:
        return None
    if len(Angle_M45) == 0:
        return None

    index = np.argsort(Angle_45)
    line1 = Line_list[index[0]]
    line2 = Line_list[index[1]]

    index = np.argsort(Angle_M45)
    line3 = Line_list[index[0]]
    line4 = Line_list[index[1]]

    column45 = int((line1.mid[0] + line2.mid[0])/2)
    row45 = int((line1.mid[1] + line2.mid[1])/2)

    columnM45 = int((line3.mid[0] + line4.mid[0])/2)
    rowM45 = int((line3.mid[1] + line4.mid[1])/2)

    # print(line1.line, line1.angle, line1.length)
    # print(line3.line, line3.angle, line3.length)
    cv2.line(img_in,(line1.line[0],line1.line[1]), (line1.line[2], line1.line[3]),(255, 0, 0), 3)
    cv2.line(img_in,(line2.line[0],line2.line[1]), (line2.line[2], line2.line[3]),(255, 0, 0), 3)
    cv2.line(img_in,(line3.line[0],line3.line[1]), (line3.line[2], line3.line[3]),(255, 0, 0), 3)
    cv2.line(img_in,(line4.line[0],line4.line[1]), (line4.line[2], line4.line[3]),(255, 0, 0), 3)

    column = (column45 + columnM45)//2 + 1
    row = (row45 + rowM45)//2 + 1
    coordinates = (column, row)

    # cv2.circle(img_in, coordinates, 2, (255, 0, 0), 2)
    return coordinates
    raise NotImplementedError


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    thresh1 = 110
    thresh2 = 60
    cannyEdges = cv2.Canny(img_in, thresh1, thresh2)
    # cv2.imshow("Edge", cannyEdges)

    circles = cv2.HoughCircles(cannyEdges,cv2.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=0,maxRadius=0)
    for circle in circles[0, :]:
        #cv2.circle(img_in, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)
        column = circle[0]
        row = circle[1]
        coordinates = (column, row)
        state_pixels = img_in[int(row), int(column), :]
        if state_pixels[0] == 255 and state_pixels[1] == 255 and state_pixels[2] == 255 :
            # cv2.circle(img_in, coordinates, 2, (255, 0, 0), 2)
            # cv2.imshow("mark", img_in)
            return coordinates

    raise NotImplementedError


def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    raw_img = np.copy(img_in)
    DetectedObj = {}

    ###################################  
    ### Detecting the traffic light ###
    ###################################  

    thresh1 = 110
    thresh2 = 60
    cannyEdges = cv2.Canny(img_in, thresh1, thresh2)

    circles = cv2.HoughCircles(cannyEdges,cv2.HOUGH_GRADIENT, 1, 20, param1=50,param2=30,minRadius=0,maxRadius=50)
    circles_selected = select_three(circles)
    print(circles_selected)
    if circles_selected != None:
        column = circles_selected[1][0]
        row = circles_selected[1][1]
        coordinates = (column, row)
        DetectedObj['Traffic_Sign'] = coordinates
        #cv2.circle(img_in, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)


    ###################################  
    ### Detecting the No_Entry sign ###
    ###################################  

    for circle in circles[0, :]:
        column = circle[0]
        row = circle[1]
        coordinates = (column, row)
        state_pixels = img_in[int(row), int(column), :]
        if state_pixels[0] > 230 and state_pixels[1] > 230 and state_pixels[2] > 230 :
            DetectedObj['No_Entry'] = coordinates

    #################################  
    ### Detecting the Yield sign  ###
    #################################

    coordinates = yield_sign_detection(img_in)
    if coordinates != None:
        DetectedObj['Yield'] = coordinates

    #################################  
    ### Detecting the Stop  sign  ###
    #################################

    coordinates = stop_sign_detection(img_in)
    if coordinates != None:
        DetectedObj['Stop'] = coordinates

    #################################  
    ### Detecting the Construction###
    #################################

    coordinates = construction_sign_detection(img_in)
    if coordinates != None:
        DetectedObj['Construction'] = coordinates

    #################################  
    ### Detecting the Warning_Sign###
    #################################

    coordinates = warning_sign_detection(img_in)
    if coordinates != None:
        DetectedObj['Warning_Sign'] = coordinates

    return DetectedObj
    raise NotImplementedError

def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    filter_size = 5
    filter_sigma = 4
    filter_kernel = cv2.getGaussianKernel(filter_size, filter_sigma)
    filter_kernel = filter_kernel * filter_kernel.T
    img_in = cv2.filter2D(img_in, -1, filter_kernel)

    thresh1 = 110
    thresh2 = 60
    cannyEdges = cv2.Canny(img_in, thresh1, thresh2)
    # cv2.imshow('smoothed_img', img_in)
    # cv2.imshow('cannyEdges', cannyEdges)

    DetectedObj = traffic_sign_detection(img_in)

    return DetectedObj
    raise NotImplementedError


def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    raise NotImplementedError




    # lines = cv2.HoughLinesP(
    # cannyEdges,
    # rho=6,
    # theta=np.pi / 60,
    # threshold=160,
    # lines=np.array([]),
    # minLineLength=40,
    # maxLineGap=25
    # )
    # for line in lines:
    #     line =  line.flatten()
    #     cv2.line(cannyEdges,(line[0],line[1]), (line[2], line[3]),(255, 0, 0), 3)
    # cv2.imshow('detected circles',cannyEdges)