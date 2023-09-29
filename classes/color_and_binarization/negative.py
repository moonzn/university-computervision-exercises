import cv2 as cv

img = cv.imread('images/lenna.png', 1)

for x in range(127, 383):
    for y in range(127, 383):
        for c in range(0, 3):
            img[x, y, c] = 255 - img[x, y, c]

cv.imshow('lenna', img)

cv.waitKey(0)

cv.destroyAllWindows()
