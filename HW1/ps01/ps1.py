import math
import numpy as np
import cv2
import sys

# # Implement the functions below.


def extract_red(image):
    """ Returns the red channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the red channel.
    """
    temp_image = np.copy(image)
    temp_image = image[:, :, 2]
    return temp_image
    raise NotImplementedError


def extract_green(image):
    """ Returns the green channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the green channel.
    """
    temp_image = np.copy(image)
    temp_image = image[:, :, 1]
    return temp_image

    raise NotImplementedError


def extract_blue(image):
    """ Returns the blue channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the blue channel.
    """
    raise NotImplementedError


def swap_green_blue(image):
    """ Returns an image with the green and blue channels of the input image swapped. It is highly
    recommended to make a copy of the input image in order to avoid modifying the original array.
    You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 3D array with the green and blue channels swapped.
    """
    temp_image = np.copy(image)
    temp_image[:, :, 0] = image[:, :, 1]
    temp_image[:, :, 1] = image[:, :, 0]
    return temp_image

    raise NotImplementedError


def copy_paste_middle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        the src into the range [1:2,1:2] of the dst.

    Args:
        src (numpy.array): 2D array where the rectangular shape will be copied from.
        dst (numpy.array): 2D array where the rectangular shape will be copied to.
        shape (tuple): Tuple containing the height (int) and width (int) of the section to be
                       copied.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    src_mid_hight = src.shape[0]
    src_mid_width = src.shape[1]
    src_hight_start = int((src_mid_hight - shape[0])/2)
    src_hight_end = int((src_mid_hight + shape[0])/2)
    src_width_start = int((src_mid_width - shape[1])/2)
    src_width_end = int((src_mid_width + shape[1])/2)

    dst_mid_hight = dst.shape[0]
    dst_mid_width = dst.shape[1]
    dst_hight_start = int((dst_mid_hight - shape[0])/2)
    dst_hight_end = int((dst_mid_hight + shape[0])/2)
    dst_width_start = int((dst_mid_width - shape[1])/2)
    dst_width_end = int((dst_mid_width + shape[1])/2)

    print("*************")
    print(dst_hight_start, dst_hight_end, dst_width_start, dst_width_end)
    print(src_hight_start, src_hight_end, src_width_start, src_width_end)

    temp_dst = np.copy(dst)


    temp_dst[dst_hight_start : dst_hight_end, dst_width_start : dst_width_end] = \
                src[src_hight_start : src_hight_end, src_width_start : src_width_end]

    return temp_dst
    raise NotImplementedError


def image_stats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    In order to become more familiar with Numpy, you should look for pre-defined functions
    that do these operations i.e. numpy.min.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.

    Returns:
        tuple: Four-element tuple containing:
               min (float): Input array minimum value.
               max (float): Input array maximum value.
               mean (float): Input array mean / average value.
               stddev (float): Input array standard deviation.
    """
    temp_image = np.copy(image)
    return temp_image.min()*1., temp_image.max()*1., temp_image.mean(), temp_image.std()
    raise NotImplementedError


def center_and_normalize(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        scale (int or float): scale factor.

    Returns:
        numpy.array: Output 2D image.
    """
    image = np.float64(image)
    temp_image = np.copy(image)

    mean = image.mean()
    std = temp_image.std()

    normalized = (temp_image - mean)/std
    result = normalized * scale + mean
    return result
    raise NotImplementedError


def shift_image_left(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.

    The returned image has the same shape as the original with
    the BORDER_REPLICATE rule to fill-in missing values.  See

    http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

    for further explanation.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        shift (int): Displacement value representing the number of pixels to shift the input image.
            This parameter may be 0 representing zero displacement.

    Returns:
        numpy.array: Output shifted 2D image.
    """
    temp_image = np.copy(image)
    r, c = image.shape
    print(r,c)
    replicate = temp_image[:, -1]

    for i in range(shift):
        temp_image[:, c - i - 1] = replicate

    if shift == 0:
        return temp_image
    else:
        temp_image[:,:-shift] = image[:, shift:]
        return temp_image
    raise NotImplementedError


def difference_image(img1, img2):
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255].

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        img1 (numpy.array): Input 2D image.
        img2 (numpy.array): Input 2D image.

    Returns:
        numpy.array: Output 2D image containing the result of subtracting img2 from img1.
    """    

    temp_img1 = np.copy(img1)
    temp_img2 = np.copy(img2)
    temp_out = temp_img1 - temp_img2


    temp_out = temp_out.astype('float64')
    Max = temp_out.max()
    Min = temp_out.min()

    if (Max - Min == 0):
        return temp_out*0.
    else :
        temp_out =  ((temp_out- Min)/(Max - Min) * 255)
        return temp_out

    #temp_out = cv2.normalize(temp_out, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    
    raise NotImplementedError


def add_noise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to
    channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise.

    The returned array values must not be clipped or normalized and scaled. This means that
    there could be values that are not in [0, 255].

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): input RGB (BGR in OpenCV) image.
        channel (int): Channel index value.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        numpy.array: Output 3D array containing the result of adding Gaussian noise to the
            specified channel.
    """
    r, c = image.shape[0], image.shape[1]
    noise = np.random.randn(r, c) * sigma
    temp_image = np.copy(image.astype(np.float64))
    temp_image[:, :, channel] =  temp_image[:, :, channel] + noise
    return temp_image

    raise NotImplementedError
