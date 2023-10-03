import cv2 as cv
import numpy as np

img = cv.imread('images/pliers.jpg', 1)
original = img.astype(np.float32)

k1 = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype=np.float32)

k2 = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype=np.float32)

# Add filters then convolution

sum_kernel = np.add(k1, k2)
img_sum_filters = cv.filter2D(original, -1, sum_kernel)

# Apply filters individually then sum

filtered_k1 = cv.filter2D(original, -1, k1)
filtered_k2 = cv.filter2D(original, -1, k2)

img_sum_filtered = cv.add(filtered_k1, filtered_k2)

cv.imshow('original', original)
cv.imshow('both filters', img_sum_filters)
cv.imshow('sequence of filters', img_sum_filtered)

cv.waitKey(0)

cv.destroyAllWindows()
