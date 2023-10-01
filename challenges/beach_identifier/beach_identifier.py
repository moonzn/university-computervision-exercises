import numpy as np
import cv2 as cv
import math
import os

# Beach classifier
directory = "images"
image_width = 200
image_height = 200


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

    org = (40, 175)

    font_scale = 0.8

    thickness = 2

    cv.putText(img, 'prediction', org, font, font_scale, color, thickness, cv.LINE_AA)


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

    for idx, img in enumerate(images):
        prediction = classifier(img)
        images[idx] = cv.resize(img, (image_width, image_height))
        color_code_prediction(images[idx], classes[idx], prediction)

    tile = create_image_grid(images)

    cv.imshow("Result", tile)
    cv.waitKey(0)


if __name__ == "__main__":
    main()
