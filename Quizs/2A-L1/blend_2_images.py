import cv2
import numpy as np

def blend(a, b, alpha):
	return alpha * a + (1. - alpha) * b


dolphin = cv2.imread("images/dolphin.png")
bicycle = cv2.imread("images/bicycle.png")

result = blend(dolphin, bicycle, 0.75)
cv2.imshow('Result', result.astype(np.uint8))
cv2.waitKey(0)