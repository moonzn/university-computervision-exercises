import numpy as np
import cv2 as cv
import random
import math
import os

# Beach classifier
directory = "images"
image_width = 200
image_height = 200
total = 0
correct = 0


def classifier(img):
    # convert image to hsv
    # crop twice, horizontal and vertical

    # mask blues (water) on top half of the img?
    # mask yellows (sand) on bottom half of the img?

    # OR

    # mask blues (water) on left half of the img?
    # mask yellows (sand) on right half of the img?

    # OR

    # mask blues (water) on right half of the img?
    # mask yellows (sand) on left half of the img?

    # if one of the above predicts beach then beach else nonbeach?

    # placeholder

    ops = ["beach", "nonbeach"]

    return random.choice(ops)


def color_code_prediction(img, clazz, prediction):
    global correct
    if clazz == prediction:
        color = (0, 255, 0)
        correct += 1
    else:
        color = (0, 0, 255)

    font = cv.FONT_HERSHEY_SIMPLEX

    org = (10, 185)

    cv.putText(img, f'{prediction}', org, font, 0.8, (0, 0, 0), 8, cv.LINE_AA)
    cv.putText(img, f'{prediction}', org, font, 0.8, color, 2, cv.LINE_AA)


def metrics():
    print(f"Accuracy: {math.ceil(correct / total * 100)}%")


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
    images, classes = load_directory()
    global total
    total = len(images)

    for idx, img in enumerate(images):
        prediction = classifier(img)
        images[idx] = cv.resize(img, (image_width, image_height))
        color_code_prediction(images[idx], classes[idx], prediction)

    tile = create_image_grid(images)

    metrics()
    cv.imshow("Result", tile)
    cv.waitKey(0)


if __name__ == "__main__":
    main()
