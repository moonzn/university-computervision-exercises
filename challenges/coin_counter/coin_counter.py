import numpy as np
import cv2 as cv
import math
import os

directory = "images"
image_width = 200
image_height = 200


def coin_counter(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Image binarization
    (t, img_bin) = cv.threshold(gray_img, 0, 255, cv.THRESH_OTSU)
    img_bin = cv.bitwise_not(img_bin)

    # Option 1
    # strel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    # dilated_img = cv.dilate(img_bin, strel)
    # strel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12, 12))
    # eroded_img = cv.erode(dilated_img, strel)

    # Option 2
    # strel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    # eroded_img = cv.erode(img_bin, strel)
    # strel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # dilated_img = cv.dilate(eroded_img, strel)

    canny_img = cv.Canny(img_bin, 0, 200, apertureSize=7, L2gradient=True)

    # Counting coins
    (numLabels, labels, boxes, centroids) = cv.connectedComponentsWithStats(canny_img)

    cv.imshow("Picture", img_bin)
    # cv.imshow("Opened Picture", img_opened)
    # cv.imshow("Eroded Picture", eroded_img)
    cv.imshow("Dilated Picture", canny_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return str(numLabels)


def put_coin_count(img, count, prediction):
    if count == prediction:
        print("true")
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    font = cv.FONT_HERSHEY_SIMPLEX
    org = (10, 185)
    cv.putText(img, f'{prediction}/{count}', org, font, 0.8, (0, 0, 0), 8, cv.LINE_AA)
    cv.putText(img, f'{prediction}/{count}', org, font, 0.8, color, 2, cv.LINE_AA)


def load_directory():
    images = []
    counts = []
    for filename in os.listdir(directory):
        img = cv.imread(os.path.join(directory, filename))
        if img is not None:
            images.append(img)
            counts.append(''.join([char for char in filename if char.isnumeric()]))
    return images, counts


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

    for idx, img in enumerate(images):
        count = coin_counter(img)
        images[idx] = cv.resize(img, (image_width, image_height))
        put_coin_count(images[idx], counts[idx], count)

    tile = create_image_grid(images)
    cv.imshow("Coin Counter", tile)
    cv.waitKey(0)


if __name__ == "__main__":
    main()
