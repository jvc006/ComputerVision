import cv2
import numpy as np 

def imshow(title, image):
	image = np.clip(image, 0, 255).astype(np.uint8)
	cv2.imshow(title, image.astype(np.uint8))


img = cv2.imread('images/saturn.png', 0)
cv2.imshow('Img', img)

noise_sigma = 25
r, c = img.shape
noise = np.random.randn(r, c) * noise_sigma
noisy_img = noise + img
imshow('Noisy Image', noisy_img)

filter_size = 11
filter_sigma = 2

filter_kernel = cv2.getGaussianKernel(filter_size, filter_sigma)
filter_kernel = filter_kernel * filter_kernel.T
smoothed = cv2.filter2D(noisy_img, -1, filter_kernel)

imshow('Smoothed image', smoothed)

cv2.waitKey(0)