"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np


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

    h, w = template[:,:,0].shape
    H, W = image[:, :, 0].shape

    rotatedTemp = np.copy(template)
    for c in range(3):
        rotatedTemp[:, :, c] = np.rot90(rotatedTemp[:, :, c])

    MatchedImage = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    MatchedImageR = cv2.matchTemplate(image, rotatedTemp, cv2.TM_CCOEFF_NORMED)

    if MatchedImageR.max() > MatchedImage.max():
        MatchedImage = MatchedImageR

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
    video = None

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    raise NotImplementedError
