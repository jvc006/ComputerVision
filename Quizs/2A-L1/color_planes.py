import cv2

img = cv2.imread('images/fruit.png')
cv2.imshow("Fruit image", img)

print(img.shape)

red = img[:, :, 2]
green = img[:, :, 1]
blue = img[:, :, 0]

cv2.imshow("Red color plane", red)
cv2.imshow("Green color plane", green)
cv2.imshow("Blue color plane", blue)


green_bgr = cv2.cvtColor(green, cv2.COLOR_GRAY2BGR)

# You will notice that cv2.line uses x-y coordinates instead of row-cols
cv2.line(green_bgr, (0, 99), (green.shape[1], 99), (0, 0, 255))
cv2.imshow("50-th row drawn on the green color plane", green_bgr)


cv2.waitKey(0)