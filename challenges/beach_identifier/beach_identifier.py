import numpy as np
import cv2 as cv
import math
import os

# Beach classifier
directory = "images/"


def classifier(img):
    # mask blues (water) on top half of the img?

    # mask yellows (sand) on bottom half of the img?

    return "prediction in string"


def color_code_prediction(img, clazz, prediction):
    if clazz == prediction:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    font = cv.FONT_HERSHEY_SIMPLEX

    org = (50, 50)

    font_scale = 1

    thickness = 2

    cv.putText(img, 'prediction', org, font,
               font_scale, color, thickness, cv.LINE_AA)


def metrics():
    print("ratio (corrects/total)")


def load_directory():
    images = []
    classes = []
    for filename in os.listdir(directory):
        img = cv.imread(os.path.join(directory, filename))
        if img is not None:
            images.append(img)
            if "nonbeach" in filename:
                classes.append("nonbeach")
            else:
                classes.append("beach")
    return images, classes


def create_image_grid(images):
    width = 200
    height = 200

    for idx, img in enumerate(images):
        images[idx] = cv.resize(img, (width, height))

    cols = math.floor(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    padding = rows * cols - len(images)

    print(rows)
    print(cols)
    print(padding)


def main():
    images, classes = load_directory()

    for idx, img in enumerate(images):
        prediction = classifier(img)
        color_code_prediction(img, classes[idx], prediction)

    create_image_grid(images)


if __name__ == "__main__":
    main()
