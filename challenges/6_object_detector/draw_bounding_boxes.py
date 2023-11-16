"""
APVC - Challenge 6 (Performance of a cat detector)

This file implements the reading and visualization of the images from the dataset with the ground truth of the cats,
read from the annotations of each image.
This file also helps calculate the IoU (Intersection over Union) metric between two boxes.

Instructions:
• To run this program, you need to place a "dataset" and a "config" directory in the same directory as this script.
• The directory should be organized as follows:
    - config
        - object_detection_classes_coco.txt
        - yolov5s.onnx
    - dataset
        - images
            - cats_01.jpg
            - cats_02.jpg
            (...)
        - annotations
            - cats_01.xml
            - cats_02.xml
            (...)

Authors:
• Bernardo Grilo, n.º 93251
• Gonçalo Carrasco, n.º 109379
• Raúl Nascimento, n.º 87405
"""

import os
import cv2 as cv
import xml.etree.ElementTree as ET

IMAGES_PATH = "dataset/images"  # Directory with cat pictures
ANNOT_PATH = "dataset/annotations"  # Directory with the annotations of the bounding boxes of the cat images


# Reading the folder containing the images.
def load_images():
    images = []
    for filename in os.listdir(IMAGES_PATH):
        img = cv.imread(os.path.join(IMAGES_PATH, filename))
        if img is not None:
            images.append(img)
    return images


# Obtaining the ground truth coordinates of the images from the annotation files
def load_bounding_boxes():
    boxes = []
    total = 0
    for filename in os.listdir(ANNOT_PATH):
        path = os.path.join(ANNOT_PATH, filename)
        box, count = parse_xml(path)
        boxes.append(box)
        total += count
    return boxes, total


# Reading the ground truth coordinates of an image from its XML annotation file
def parse_xml(xml):
    # Create element tree object
    tree = ET.parse(xml)

    # Get root element
    root = tree.getroot()

    # Stores all boxes of an img
    boxes_in_img = []

    # Counts boxes
    count = 0

    for box in root.findall("./object/bndbox"):
        coords = []
        for coord in box:
            coords.append(coord.text)
        boxes_in_img.append(coords)
        count += 1

    return boxes_in_img, count


# Drawing the bounding boxes on an image
def draw_bounding_boxes(img, boxes):
    color = (135, 135, 135)
    # Line thickness of 2 px
    thickness = 2

    for coords in boxes:
        # Represents the top left corner of rectangle
        start_point = (int(coords[0]), int(coords[1]))
        # Represents the bottom right corner of rectangle
        end_point = (int(coords[2]), int(coords[3]))

        img = cv.rectangle(img, start_point, end_point, color, thickness)
    return img


# Converting the elements of a list with numeric strings to integer values
def cast_list_to_int(lst):
    return list(map(int, lst))


# Calculation of the IoU metric between two boxes (ground truth and detection bounding box)
def calculate_iou(box1, box2):
    x_inter1 = max(box1[0], box2[0])
    y_inter1 = max(box1[1], box2[1])
    x_inter2 = min(box1[2], box2[2])
    y_inter2 = min(box1[3], box2[3])

    if x_inter2 < x_inter1 or y_inter2 < y_inter1:
        return 0

    # Intersection area
    width_inter = abs(x_inter2 - x_inter1)
    height_inter = abs(y_inter2 - y_inter1)
    area_inter = width_inter * height_inter

    # Union area
    width_ground_truth = abs(box1[2] - box1[0])
    height_ground_truth = abs(box1[3] - box1[1])
    area_ground_truth = width_ground_truth * height_ground_truth
    width_prediction = abs(box2[2] - box2[0])
    height_prediction = abs(box2[3] - box2[1])
    area_detection = width_prediction * height_prediction
    area_union = area_detection + area_ground_truth - area_inter

    return area_inter / area_union


def main():
    images = load_images()
    boxes, count = load_bounding_boxes()

    for idx in range(len(images)):
        images[idx] = draw_bounding_boxes(images[idx], boxes[idx])

    # Display of the images with the rectangles (bounding boxes of the annotations)
    for img in images:
        cv.imshow("Bounding Boxes", img)
        cv.waitKey(0)


if __name__ == "__main__":
    main()
