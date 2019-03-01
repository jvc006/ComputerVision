"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2
import os


# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    return cv2.Sobel(image, cv2.CV_64F, 1, 0, scale=0.125, ksize=3)
    raise NotImplementedError


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """

    return cv2.Sobel(image, cv2.CV_64F, 0, 1, scale=0.125, ksize=3)
    raise NotImplementedError


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """

    if k_type == 'gaussian' : 
        img_a = cv2.GaussianBlur(img_a, (k_size, k_size), sigma, sigma)
        img_b = cv2.GaussianBlur(img_b, (k_size, k_size), sigma, sigma)

    I_t = img_b - img_a
    Grad_X = gradient_x(img_b)
    Grad_Y = gradient_y(img_b)
    Grad_X_X = np.square(Grad_X)
    Grad_X_Y = np.multiply(Grad_X, Grad_Y)
    Grad_Y_Y = np.square(Grad_Y)
    Grad_X_t = np.multiply(Grad_X, I_t)
    Grad_Y_t = np.multiply(Grad_Y, I_t)

    h, w = img_a.shape
    A = np.zeros((2, 2, h, w), dtype = np.float32)
    b = np.zeros((2, h, w), dtype = np.float32)

    kernel = np.ones((k_size, k_size),np.float32)
    A[0, 0] = cv2.filter2D(Grad_X_X, -1, kernel)
    A[0, 1] = cv2.filter2D(Grad_X_Y, -1, kernel)
    A[1, 0] = cv2.filter2D(Grad_X_Y, -1, kernel)
    A[1, 1] = cv2.filter2D(Grad_Y_Y, -1, kernel)
    b[0] = -cv2.filter2D(Grad_X_t, -1, kernel)
    b[1] = -cv2.filter2D(Grad_Y_t, -1, kernel)
    # A[0, 0] = cv2.GaussianBlur(Grad_X_X, (k_size, k_size), sigma, sigma)
    # A[0, 1] = cv2.GaussianBlur(Grad_X_Y, (k_size, k_size), sigma, sigma)
    # A[1, 0] = cv2.GaussianBlur(Grad_X_Y, (k_size, k_size), sigma, sigma)
    # A[1, 1] = cv2.GaussianBlur(Grad_Y_Y, (k_size, k_size), sigma, sigma)
    # b[0] = -cv2.GaussianBlur(Grad_X_t, (k_size, k_size), sigma, sigma)
    # b[1] = -cv2.GaussianBlur(Grad_X_t, (k_size, k_size), sigma, sigma)

    inverse_A = np.copy(A)
    denominator = np.clip(A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0], 0.00000001, np.inf)
    # denominator = A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0]
    # factor = np.where(denominator != 0, 1/denominator, 0)
    factor = 1/denominator
    inverse_A[0, 0] = factor * A[1, 1]
    inverse_A[0, 1] = -factor * A[1, 0]
    inverse_A[1, 0] = -factor * A[0, 1]
    inverse_A[1, 1] = factor * A[0, 0]

    res1 = inverse_A[0, 0] * b[0] + inverse_A[0, 1] * b[1]
    res2 = inverse_A[1, 0] * b[0] + inverse_A[1, 1] * b[1]

    return (np.asarray(res1, dtype = np.float32), np.asarray(res2,dtype = np.float32))
    raise NotImplementedError


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    image_copy = np.copy(image)
    h = image.shape[0]//2
    w = image.shape[1]//2
    img_bd = np.ones([h, w], dtype = np.float64)
    temp = np.array([[0.0625, 0.25, 0.375, 0.25, 0.0625]], dtype = np.float64)
    kernel = temp * temp.T
    kernel = np.asarray(kernel, dtype = np.float64)

    # filter_size = 5
    # filter_sigma = 1
    # filter_kernel = cv2.getGaussianKernel(filter_size, filter_sigma)
    # filter_kernel = filter_kernel * filter_kernel.T

    img_bd = cv2.filter2D(image_copy, -1, kernel)[::2, ::2]
    return img_bd

    raise NotImplementedError


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    image_copy = np.copy(image)
    res = []
    for i in range(levels):
        res.append(image_copy)
        image_copy = reduce_image(image_copy)

    return res
    raise NotImplementedError


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """

    res = normalize_and_scale(img_list[0])
    N = len(img_list)
    for i in range(1, N, 1) :
        H, W = res.shape
        h, w = img_list[i].shape
        target = np.zeros((H, W + w), dtype = np.float32)
        target[: H, : W] = res
        target[: h, W : W + w] = normalize_and_scale(img_list[i])
        # target[h :, W : W + w] = 255.
        res = target

    return res

    raise NotImplementedError


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """

    h , w = image.shape
    H, W = 2*h, 2*w
    img_odd = np.zeros((H, W), dtype=np.float64)
    img_odd[0: H : 2, 0 : W : 2]= image[:,:]
    kernel = np.array([[0.125, 0.5, 0.75, 0.5, 0.125]], dtype = np.float64)
    res = cv2.sepFilter2D(img_odd, -1, kernel, kernel)

    return res

    raise NotImplementedError


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """


    res = []
    N = len(g_pyr)
    for i in range(N-1):
        pre = g_pyr[i]
        cur = expand_image(g_pyr[i+1])
        if pre.shape[0] < cur.shape[0]:
            cur = np.delete(cur, (-1), axis=0)
        if pre.shape[1] < cur.shape[1]:
            cur = np.delete(cur, (-1), axis=1)
        res.append(pre - cur)
    res.append(g_pyr[-1])
    return res
    raise NotImplementedError


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """

    h, w = image.shape
    X, Y = np.meshgrid(range(w), range(h))
    addX = (X + U).astype(np.float32)
    addY = (Y + V).astype(np.float32)
    res = cv2.remap(image, addX, addY, interpolation=interpolation, borderMode=border_mode)
    return res
    raise NotImplementedError


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """
    img_a_list = gaussian_pyramid(img_a, levels)
    img_b_list = gaussian_pyramid(img_b, levels)

    h, w = img_a.shape  
    h = h//(2**(levels-1))
    w = w//(2**(levels-1))
    u = np.zeros((h, w), dtype = np.float64)
    v = np.zeros((h, w), dtype = np.float64)

    for level_id in range(levels-1, -1, -1):
        if level_id != levels-1 :
            u = expand_image(u)
            v = expand_image(v)
            u = 2*u
            v = 2*v
        imageB = img_b_list[level_id]
        if level_id == levels-1 :
            imageB_wraped = imageB
        else:
            imageB_wraped = warp(imageB, u, v, interpolation, border_mode)
        imageA = img_a_list[level_id]
        du, dv = optic_flow_lk(imageA, imageB_wraped, k_size, k_type, sigma)
        u += du
        v += dv
    return (u, v)

    raise NotImplementedError
