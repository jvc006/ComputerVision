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

    circles = cv2.HoughCircles(cannyEdges,cv2.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=0,maxRadius=0)
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

        if line_instance.angle > 55 and line_instance.angle < 65:   
            # print(line_instance.line, line_instance.angle)
            Angle_60.append(line_instance.length)
            Line_list_60.append(line_instance)

        if line_instance.angle > -65 and line_instance.angle < -55:  
            # print(line_instance.line, line_instance.angle) 
            Angle_M60.append(line_instance.length)
            Line_list_M60.append(line_instance)
        
    index = np.argsort(Angle_60)
    line1 = Line_list_60[index[-1]].line
    cv2.line(img_in,(line1[0],line1[1]), (line1[2], line1[3]),(255, 0, 0), 3)

    index = np.argsort(Angle_M60)
    line3 = Line_list_M60[index[-1]].line
    cv2.line(img_in,(line3[0],line3[1]), (line3[2], line3[3]),(255, 0, 0), 3)

    X_60 = max(line1[0], line1[2])
    X_M60 = min(line3[0], line3[2])
    column = int ((X_60 + X_M60)/2)

    left_Y = min(line1[1], line1[3])
    mid_Y_60 = max(line1[1], line1[3])
    mid_Y_M60 = max(line3[1], line3[3])
    right_Y = min(line3[1], line3[3])
    row = int ((left_Y + (mid_Y_60+mid_Y_M60)/2 + right_Y)/3)
    coordinates = (column, row)

    cv2.circle(img_in, coordinates, 2, (255, 0, 0), 2)
    cv2.imshow('detected lines',img_in)
    return coordinates
    raise NotImplementedError


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

    lines = cv2.HoughLinesP(cannyEdges, rho=1, theta=np.pi /100, threshold=30, minLineLength=20, maxLineGap=1)

    Line_list = []
    Angle_45 = []
    Angle_M45 = []

    for line in lines:
        line =  line.flatten()
        line_instance = Line(line)
        if line_instance.length < 100 and line_instance.angle != 0 :
            Line_list.append(line_instance) 
            cv2.line(img_in,(line[0],line[1]), (line[2], line[3]),(255, 0, 0), 3)
            Angle_45.append(np.abs(line_instance.angle - 45))
            Angle_M45.append(np.abs(line_instance.angle + 45))

        
    index = np.argsort(Angle_45)
    line1 = Line_list[index[0]]
    line2 = Line_list[index[1]]

    print(line1.line, line1.angle)
    print(line2.line, line1.angle)

    column = int((line1.mid[0] + line2.mid[0])/2)
    row = int((line1.mid[1] + line2.mid[1])/2)
    coordinates = (column, row)

    cv2.circle(img_in, coordinates, 2, (255, 0, 0), 2)
    cv2.imshow('detected lines',img_in)


    return coordinates
    raise NotImplementedError


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    raise NotImplementedError


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
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
        cv2.circle(img_in, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)

        column = circle[0]
        row = circle[1]
        coordinates = (column, row)
        state_pixels = img_in[int(row), int(column), :]
        if state_pixels[0] == 255 and state_pixels[1] == 255 and state_pixels[2] == 255 :
            cv2.circle(img_in, coordinates, 2, (255, 0, 0), 2)
            cv2.imshow("mark", img_in)
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