import cv2 as cv

img = cv.imread('images/corruptedRect.png', 1)

rect_strel = cv.getStructuringElement(cv.MORPH_RECT, (12, 12))

imgEroded = cv.erode(img, rect_strel)

rect_strel2 = cv.getStructuringElement(cv.MORPH_RECT, (19, 19))

imgDilated = cv.dilate(imgEroded, rect_strel2)

sequence = cv.hconcat([img, imgEroded, imgDilated])

cv.imshow('sequence', sequence)

cv.waitKey(0)

cv.destroyAllWindows()
