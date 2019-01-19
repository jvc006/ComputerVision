import cv2
import numpy as np
import scipy.signal as sp

def normalize(img_in):
	img_out = np.zeros(img_in.shape)
	cv2.normalize(img_in, img_out, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
	return img_out

# Gradient Direction
def select_gdir(gmag, gdir, mag_min, angle_low, angle_high):
	result = gmag >= mag_min
	result &= gdir >= angle_low
	result &= gdir <= angle_high
	return result.astype(np.float) 

img = cv2.imread('images/octagon.png', 0) / 255.
cv2.imshow('Image', img)

gx = cv2.Sobel(img, -1, dx = 1, dy = 0)
gy = cv2.Sobel(img, -1, dx = 0, dy = 1)
cv2.imshow('Gx', gx)
cv2.imshow('Gy', gy)

gmag = np.sqrt(gx**2 + gy**2)

gdir = np.arctan2(-gy, gx) * 180 /np.pi
cv2.imshow('Gmag', gmag/(4*np.sqrt(2)))
cv2.imshow('Gdir', normalize(gdir).astype(np.uint8))

# Find pixels with desired gradient direction
my_grad = select_gdir(gmag, gdir, 1, 30, 60)
cv2.imshow('My Grad', my_grad)
cv2.waitKey(0)