import cv2


img = cv2.imread('images/fall-leaves.png')
cv2.imshow('Image', img)


## create a gaussian filter
filter_size = 11
filter_sigma = 2
filter_kernel = cv2.getGaussianKernel(filter_size, filter_sigma)
filter_kernel = filter_kernel * filter_kernel.T

edge_param = {"replicate": cv2.BORDER_CONSTANT, "symmetric" : cv2.BORDER_REFLECT, "circular": cv2.BORDER_WRAP}

method = "circular"

if method == "circular":
	# add the boundary
	temp_img = cv2.copyMakeBorder(img, filter_size, filter_size, filter_size, filter_size, edge_param[method])
	# filter with borderType default (BORDER_REFLECT)
	smoothed = cv2.filter2D(temp_img, -1, filter_kernel)
	# crop the image to origianl size
	smoothed = smoothed[filter_size : -filter_size, filter_size : -filter_size]
else :
	# the second parater -1, the output image will have the same depth as the source (scr.depth())
	smoothed = cv2.filter2D(img, -1, filter_kernel, borderType = edge_param[method])

cv2.imshow('Smoothed', smoothed)
cv2.waitKey(0);