import cv2 as cv
import numpy as np

h = np.ones((360, 720), np.uint8)
s = np.ones((360, 720), np.uint8) * 255
v = np.ones((360, 720), np.uint8) * 255

for w in range(720):
    h[:, w] = w//4

hsv = cv.merge((h, s, v))

final = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

cv.imshow('hsv', final)

cv.waitKey(0)

cv.destroyAllWindows()
