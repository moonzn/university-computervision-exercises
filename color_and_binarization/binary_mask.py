import cv2 as cv
import numpy as np

img = cv.imread('../images/legos.jpg', 1)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
(h, s, v) = cv.split(hsv)

mask_h = cv.inRange(h, 100, 125)
mask_s = cv.inRange(s, 50, 255)

mask = cv.bitwise_and(mask_h, mask_s)

masked = cv.bitwise_and(hsv, hsv, mask=mask)
masked = cv.cvtColor(masked, cv.COLOR_HSV2BGR)

cv.imshow('legos', masked)

cv.waitKey(0)

cv.destroyAllWindows()
