"""
APVC - Challenge 2 (Coin counter)

Instructions:
• To run this program you must place an "images" directory in the same directory as this script.
• Place images of coins in the "images" directory.

The images given with 6 and 10 coins are being classified with 7 and 11 respectively.
We weren't able to understand why.

Authors:
• Bernardo Grilo, n.º 93251
• Gonçalo Carrasco, n.º 109379
• Raúl Nascimento, n.º 87405
"""

import numpy as np
import cv2 as cv
import math
import os

directory = "images"
image_width = 200
image_height = 200


# Function corresponding to the counter.
# This function receives the image and returns the number of coins counted.
def coin_counter(img):
    # Convert to grayscale
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Image binarization
    (t, img_bin) = cv.threshold(gray_img, 0, 255, cv.THRESH_OTSU)
    img_bin = cv.bitwise_not(img_bin)

    # Application of morphological operations
    strel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
    dilated_img = cv.dilate(img_bin, strel)
    strel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (44, 44))
    eroded_img = cv.erode(dilated_img, strel)

    # Counting coins
    (numLabels, labels, boxes, centroids) = cv.connectedComponentsWithStats(eroded_img)

    return str(numLabels)


# Method that counts the number of correct predictions made by the classifier.
# It also writes the prediction on the image and colors the text green or red.
# This colorization depends on whether the prediction is correct.
def put_coin_count(img, count, prediction):
    if count == prediction:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    font = cv.FONT_HERSHEY_SIMPLEX
    org = (10, 185)
    cv.putText(img, f'{prediction}/{count}', org, font, 0.8, (0, 0, 0), 8, cv.LINE_AA)
    cv.putText(img, f'{prediction}/{count}', org, font, 0.8, color, 2, cv.LINE_AA)


# Reading the folder containing the images.
# The images and their corresponding coin count are read out.
def load_directory():
    images = []
    counts = []
    for filename in os.listdir(directory):
        img = cv.imread(os.path.join(directory, filename))
        if img is not None:
            images.append(img)
            counts.append(''.join([char for char in filename if char.isnumeric()]))
    return images, counts


# This function creates and returns a grid with the images and coin count in text (over the image)
# Images are resized AFTER being classified, just so they can fit nicely on the screen
def create_image_grid(images):
    rows = math.floor(math.sqrt(len(images)))
    cols = math.ceil(len(images) / rows)
    padding = rows * cols - len(images)

    def divide_chunks(ls, n):
        for i in range(0, len(ls), n):
            yield ls[i:i + n]

    rows_list = list(divide_chunks(images, cols))

    for p in range(padding):
        rows_list[len(rows_list) - 1].append(np.ones([image_width, image_height, 3], dtype=np.uint8) * 255)

    tile = cv.vconcat([cv.hconcat(list_h) for list_h in rows_list])

    return tile


def main():
    images, counts = load_directory()

    # Counting the coins for each image and structuring it for subsequent display of the image
    # and its coin count on a grid
    for idx, img in enumerate(images):
        count = coin_counter(img)
        images[idx] = cv.resize(img, (image_width, image_height))
        put_coin_count(images[idx], counts[idx], count)

    tile = create_image_grid(images)
    cv.imshow("Coin Counter", tile)
    cv.waitKey(0)


if __name__ == "__main__":
    main()
