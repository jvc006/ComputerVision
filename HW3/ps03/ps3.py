"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
import time


def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """

    raise NotImplementedError


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    h = image.shape[0]
    w = image.shape[1]

    loc = []
    top_left = (0, 0)
    bottom_left = (0, h-1)
    top_right = (w-1, 0)
    bottom_right = (w-1, h-1)
    loc.append(top_left)
    loc.append(bottom_left)
    loc.append(top_right)
    loc.append(bottom_right)
    return loc
    raise NotImplementedError


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """

    h = template.shape[0]
    w = template.shape[1]

    H = image.shape[0]
    W = image.shape[1]

    # h, w = template[:,:,0].shape
    # H, W = image[:, :, 0].shape

    # for angle in range(-45, 20, 70):

    #     rotatedX = np.copy(template)
    #     rotatedX[:, :, :] = 255
    
    #     M = [[ np.cos(np.pi/180 * angle), -np.sin(np.pi/180 * angle)],
    #          [ np.sin(np.pi/180 * angle),  np.cos(np.pi/180 * angle)]]
    
    #     for x in range(w):
    #         for y in range(h):
    #             X = int(x - w/2)
    #             Y = int(y - h/2)
    #             if X**2 + Y**2 < 16**2 : 
    #                 src_points =  np.array([X, Y])
    #                 dst_points = np.matmul(M, src_points)
    
    #                 locX = int(dst_points[0] + w/2)
    #                 locY = int(dst_points[1] + h/2)
    #                 rotatedX[locY, locX, :] = template[y, x, :]
    #                 MatchedImageX = cv2.matchTemplate(image, rotatedX, cv2.TM_CCOEFF_NORMED)
    #                 if MatchedImageX.max() > MatchedImage.max():
    #                     MatchedImage = MatchedImageX
    

    rotated45 = np.copy(template)
    rotated45[:, :, :] = 255

    rotated25 = np.copy(template)
    rotated25[:, :, :] = 255

    rotatedN45 = np.copy(template)
    rotatedN45[:, :, :] = 255

    rotatedN25 = np.copy(template)
    rotatedN25[:, :, :] = 255

    for x in range(w):
        for y in range(h):
            X = int(x - w/2)
            Y = int(y - h/2)
            if (Y < 0 and (2*X) > (Y)  and (0.4*X) < (-Y) and X**2 + Y**2 <= 15**2) \
                                    or (Y > 0 and (2*X) < (Y) and (0.4*X) > (-Y) and X**2 + Y**2 <= 15**2): 
                locX = int(X + w/2)
                locY = int(Y + h/2)
                rotated45[locY, locX, :] = 0
            if X**2 + Y**2 < 15.5**2 and X**2 + Y**2 > 14.5**2: 
                locX = int(X + w/2)
                locY = int(Y + h/2)
                rotated45[locY, locX, :] = 0

            if (Y < 0 and (3*X) > (Y)  and (0.2*X) < (-Y) and X**2 + Y**2 <= 15**2) \
                                    or (Y > 0 and (3*X) < (Y) and (0.2*X) > (-Y) and X**2 + Y**2 <= 15**2): 
                locX = int(X + w/2)
                locY = int(Y + h/2)
                rotated25[locY, locX, :] = 0
            if X**2 + Y**2 < 15.5**2 and X**2 + Y**2 > 14.5**2: 
                locX = int(X + w/2)
                locY = int(Y + h/2)
                rotated25[locY, locX, :] = 0
            
            ## Get the N 45 
            if (X < 0 and (2*X) < (-Y)  and (0.4*X) < (Y) and X**2 + Y**2 <= 15**2) \
                                    or (X > 0 and (2*X) > (-Y) and (0.4*X) > (Y) and X**2 + Y**2 <= 15**2): 
                locX = int(X + w/2)
                locY = int(Y + h/2)
                rotatedN45[locY, locX, :] = 0
            if X**2 + Y**2 < 15.5**2 and X**2 + Y**2 > 14.5**2: 
                locX = int(X + w/2)
                locY = int(Y + h/2)
                rotatedN45[locY, locX, :] = 0

            if (X < 0 and (3*X) < (-Y)  and (0.2*X) < (Y) and X**2 + Y**2 <= 15**2) \
                                    or (X > 0 and (3*X) > (-Y) and (0.2*X) > (Y) and X**2 + Y**2 <= 15**2): 
                locX = int(X + w/2)
                locY = int(Y + h/2)
                rotatedN25[locY, locX, :] = 0
            if X**2 + Y**2 < 15.5**2 and X**2 + Y**2 > 14.5**2: 
                locX = int(X + w/2)
                locY = int(Y + h/2)
                rotatedN25[locY, locX, :] = 0

    rotatedTemp = np.copy(template)
    for c in range(3):
        rotatedTemp[:, :, c] = np.rot90(rotatedTemp[:, :, c])

    # cv2.imshow('rotated45', rotated45)
    # cv2.imshow('rotated25', rotated25)
    # cv2.imshow('rotatedN45', rotatedN45)
    # cv2.imshow('rotatedN25', rotatedN25)

    MatchedImage = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    MatchedImageR = cv2.matchTemplate(image, rotatedTemp, cv2.TM_CCOEFF_NORMED)
    MatchedImage45 = cv2.matchTemplate(image, rotated45, cv2.TM_CCOEFF_NORMED)
    MatchedImage25 = cv2.matchTemplate(image, rotated25, cv2.TM_CCOEFF_NORMED)
    MatchedImageN45 = cv2.matchTemplate(image, rotatedN45, cv2.TM_CCOEFF_NORMED)
    MatchedImageN25 = cv2.matchTemplate(image, rotatedN25, cv2.TM_CCOEFF_NORMED)

    if MatchedImageR.max() > MatchedImage.max():
        MatchedImage = MatchedImageR

    if MatchedImage45.max() > MatchedImage.max():
        MatchedImage = MatchedImage45

    if MatchedImage25.max() > MatchedImage.max():
        MatchedImage = MatchedImage25

    if MatchedImageN45.max() > MatchedImage.max():
        MatchedImage = MatchedImageN45

    if MatchedImageN25.max() > MatchedImage.max():
        MatchedImage = MatchedImageN25

    # flat45 = MatchedImage45.flatten()
    # flat45.sort()
    # flat = MatchedImage.flatten()
    # flat.sort()

    # print(flat45[-4:])
    # print(flat[-4:])

    locs = []

    for i in range(4):
        y, x = np.unravel_index(MatchedImage.argmax(), MatchedImage.shape)
        temp = (x+int(w/2), y+int(h/2))

        color = (0, 50, 255)

        MatchedImage[y-5 : y+5, x-5 : x+5] = 0.
        locs.append(temp)

    locs = sorted(locs, key=lambda t: (t[0]*H + t[1]))
    locs[0:2] = sorted(locs[0:2], key=lambda t: (t[1]*W + t[0]))
    locs[2:4] = sorted(locs[2:4], key=lambda t: (t[1]*W + t[0]))

    return locs
    raise NotImplementedError


def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """

    cv2.line(image,(markers[0]), (markers[1]),(0, 50, 255), thickness)
    cv2.line(image,(markers[0]), (markers[2]),(0, 50, 255), thickness)
    cv2.line(image,(markers[3]), (markers[1]),(0, 50, 255), thickness)
    cv2.line(image,(markers[3]), (markers[2]),(0, 50, 255), thickness)

    return image
    raise NotImplementedError


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    h, w = imageA[:, :, 0].shape
    H, W = imageB[:, :, 0].shape

    src_points = np.zeros((3, h*w), np.int32)
    src_points[2, :] = 1
    for x in range(w):
        src_points[0, x*h : (x+1)*h] = x
        src_points[1, x*h : (x+1)*h] = np.arange(h)

    dst_points = np.matmul(homography, src_points)
    dst_points[:,:] = dst_points[:,:]/dst_points[2,:]

    x = np.array(src_points[0, :])
    y = np.array(src_points[1, :])
    locX = np.array(dst_points[0, :])
    locY = np.array(dst_points[1, :])
    locX = np.clip(locX, 0, W-1)
    locY = np.clip(locY, 0, H-1)
    locX = locX.astype(int)
    locY = locY.astype(int)
    imageB[locY,  locX, :] = imageA[y, x, :]

    # start_time = time.time()
    # for x in range(w):
    #     for y in range(h): 
    #         locX = int(dst_points[0, x*h + y])
    #         locY = int(dst_points[1, x*h + y])
    #         if locX < W and locY < H and locX >= 0 and locY >= 0:
    #             imageB[locY,  locX, :] = imageA[y, x, :]

    # elapsed_time = time.time() - start_time
    # print("Finish : ", elapsed_time)

    return imageB

    raise NotImplementedError


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """

    M = []
    n = len(src_points)
    for i in range(n):
        x, y = src_points[i][0], src_points[i][1]
        u, v = dst_points[i][0], dst_points[i][1]
        M.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        M.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    M = np.asarray(M)
    U, Sigma, Vt = np.linalg.svd(M)
    L = Vt[-1, :]/Vt[-1, -1]
    res = L.reshape(3, 3)
    return res

    raise NotImplementedError


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None


    raise NotImplementedError



