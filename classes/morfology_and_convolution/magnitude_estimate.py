import cv2 as cv
import numpy as np

clocks = cv.imread('images/clocks.jpg', cv.IMREAD_GRAYSCALE).astype(np.float32)
last_supper = cv.imread('images/lastsupper.jpg', cv.IMREAD_GRAYSCALE).astype(np.float32)

sobel_x = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype=np.float32)

sobel_y = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype=np.float32)

clocks_sobel_x = cv.filter2D(clocks, -1, sobel_x)
clocks_sobel_y = cv.filter2D(clocks, -1, sobel_y)

last_supper_x = cv.filter2D(last_supper, -1, sobel_x)
last_supper_y = cv.filter2D(last_supper, -1, sobel_y)

clocks_gradient = np.sqrt(np.square(clocks_sobel_x) + np.square(clocks_sobel_y))
last_supper_gradient = np.sqrt(np.square(last_supper_x) + np.square(last_supper_y))

cv.imshow('clocks', clocks_gradient / clocks_gradient.max())
cv.imshow('last supper', last_supper_gradient / last_supper_gradient.max())

cv.waitKey(0)

cv.destroyAllWindows()
