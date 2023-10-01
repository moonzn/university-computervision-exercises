import numpy as np
import cv2 as cv
import math
import os

# Beach classifier
directory = "images"
image_width = 200
image_height = 200
total = 0
correct = 0
threshold = 0.45


def classifier(img):
    # convert image to hsv
    # crop twice, horizontal and vertical
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    (h, s, v) = cv.split(hsv)

    mask_h_blue = cv.inRange(h, 100, 125)
    mask_s_blue = cv.inRange(s, 5, 255)

    mask_h_yellow = cv.inRange(h, 0, 30)
    mask_s_yellow = cv.inRange(s, 5, 255)

    mask_blue = cv.bitwise_and(mask_h_blue, mask_s_blue)
    mask_yellow = cv.bitwise_and(mask_h_yellow, mask_s_yellow)

    def evaluate_mask(mask):
        size = mask.size
        n = cv.countNonZero(mask)
        return n / size >= threshold

    (height, width, c) = hsv.shape

    # mask blues (water) on top half of the img?
    mask_blue_top = mask_blue[0:height // 2]
    # mask yellows (sand) on bottom half of the img?
    mask_yellow_bottom = mask_yellow[height // 2:height]

    if evaluate_mask(mask_blue_top) and evaluate_mask(mask_yellow_bottom):
        return "beach"

    # OR

    # mask blues (water) on left half of the img?
    mask_blue_left = mask_blue[0:height, 0:width // 2]
    # mask yellows (sand) on right half of the img?
    mask_yellow_right = mask_yellow[0:height, width // 2:width]

    if evaluate_mask(mask_blue_left) and evaluate_mask(mask_yellow_right):
        return "beach"

    # OR

    # mask blues (water) on right half of the img?
    mask_blue_right = mask_blue[0:height, width // 2:width]
    # mask yellows (sand) on left half of the img?
    mask_yellow_left = mask_yellow[0:height, 0:width // 2]

    if evaluate_mask(mask_blue_right) and evaluate_mask(mask_yellow_left):
        return "beach"

    return "nonbeach"


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
