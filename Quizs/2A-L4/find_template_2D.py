import cv2
import numpy as np
import scipy.signal as sp


def find_template_2D(template, img):
	c = sp.correlate2d(img, template, mode ='same')
	y, x = np.unravel_index(np.argmax(c), c.shape)
	print(y, x)
	return y - template.shape[0] // 2, x - template.shape[1] // 2

tablet = cv2.imread('images/tablet.png', 0)
cv2.imshow('Table', tablet)

glyph = tablet[74:165, 149:184]
cv2.imshow('Glyph', glyph)

tablet_2 = 1. * tablet - np.mean(tablet)
glyph_2 = 1. * glyph - np.mean(glyph)

y, x = find_template_2D(glyph_2, tablet_2)
print("Y: {}, X: {}".format(y, x))
cv2.waitKey(0)