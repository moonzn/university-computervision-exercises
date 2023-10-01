"""
APVC - Challenge 1 (Beach classifier)

Instructions:
• To run this program you must place an "images" directory in the same directory as this script.
• Place images of beaches in the "images" directory, the name of which should begin with "beach".
• The rest of the images should start with "nonbeach".

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
total = 0
total_beach = 0
total_non_beach = 0
correct_beach = 0
correct_non_beach = 0
threshold = 0.45


# Function corresponding to the classifier.
# This function receives the name of an image and returns the class prediction of that image.
# The prediction can be "beach" or "non-beach".
def classifier(img):
    # Convert image to hsv
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    (h, s, v) = cv.split(hsv)

    # Masking for blue (water)
    mask_h_blue = cv.inRange(h, 100, 125)
    mask_s_blue = cv.inRange(s, 5, 255)
    mask_blue = cv.bitwise_and(mask_h_blue, mask_s_blue)

    # Masking for yellow (sand)
    mask_h_yellow = cv.inRange(h, 0, 30)
    mask_s_yellow = cv.inRange(s, 5, 255)
    mask_yellow = cv.bitwise_and(mask_h_yellow, mask_s_yellow)

    # Function that evaluates the mask.
    # Checks whether the mask contains a percentage of white pixels equal to or greater than the set threshold.
    def evaluate_mask(mask):
        size = mask.size
        n = cv.countNonZero(mask)
        return n / size >= threshold

    (height, width, c) = hsv.shape

    # Mask the blues (water) in the upper half of the image and the yellows (sand) in the lower half
    mask_blue_top = mask_blue[0:height // 2]
    mask_yellow_bottom = mask_yellow[height // 2:height]
    if evaluate_mask(mask_blue_top) and evaluate_mask(mask_yellow_bottom):
        return "beach"

    # OR

    # Mask the blues (water) in the left half of the image and the yellows (sand) in the right half
    mask_blue_left = mask_blue[0:height, 0:width // 2]
    mask_yellow_right = mask_yellow[0:height, width // 2:width]
    if evaluate_mask(mask_blue_left) and evaluate_mask(mask_yellow_right):
        return "beach"

    # OR

    # Mask the blues (water) in the right half of the image and the yellows (sand) in the left half
    mask_blue_right = mask_blue[0:height, width // 2:width]
    mask_yellow_left = mask_yellow[0:height, 0:width // 2]
    if evaluate_mask(mask_blue_right) and evaluate_mask(mask_yellow_left):
        return "beach"

    # If none of the above conditions are met, it is classified as non-beach
    return "nonbeach"


# Method that counts the number of correct predictions made by the classifier.
# It also writes the prediction on the image and colors the text green or red.
# This colorization depends on whether the prediction is correct.
def color_code_prediction(img, clazz, prediction):
    global correct_beach, correct_non_beach
    if clazz == prediction:
        color = (0, 255, 0)
        if clazz == "beach":
            correct_beach += 1
        else:
            correct_non_beach += 1
    else:
        color = (0, 0, 255)

    font = cv.FONT_HERSHEY_SIMPLEX
    org = (10, 185)
    cv.putText(img, f'{prediction}', org, font, 0.8, (0, 0, 0), 8, cv.LINE_AA)
    cv.putText(img, f'{prediction}', org, font, 0.8, color, 2, cv.LINE_AA)


# Overall results of the classifier.
# Results relating to accuracy, precision and more are presented.
def metrics():
    total_correct = correct_beach + correct_non_beach
    print(f"Accuracy: {math.ceil(total_correct / total * 100)}%")
    print(f"True Positive Rate: {math.ceil(correct_beach / total_beach * 100)}%")
    print(f"True Negative Rate: {math.ceil(correct_non_beach / total_non_beach * 100)}%")
    n_false_positive = total_non_beach - correct_non_beach
    n_false_negative = total_beach - correct_beach
    print(f"Precision: {math.ceil(correct_beach / (correct_beach + n_false_positive) * 100)}%")
    print(f"Recall: {math.ceil(correct_beach / (correct_beach + n_false_negative) * 100)}%")


# Reading the folder containing the images.
# The images and their corresponding class (beach or non-beach) are read out.
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


# This function creates and returns a grid with the images and their classification in text (over the image)
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
    images, classes = load_directory()
    global total, total_beach, total_non_beach
    total = len(images)
    total_beach = classes.count("beach")
    total_non_beach = classes.count("nonbeach")

    # Classifying each image and structuring it for subsequent display of the image and its classification on a grid
    for idx, img in enumerate(images):
        prediction = classifier(img)
        images[idx] = cv.resize(img, (image_width, image_height))
        color_code_prediction(images[idx], classes[idx], prediction)

    # Display of metrics and grid with predictions made for each image
    metrics()
    tile = create_image_grid(images)
    cv.imshow("Beach Classifier", tile)
    cv.waitKey(0)


# Program start point
if __name__ == "__main__":
    main()
