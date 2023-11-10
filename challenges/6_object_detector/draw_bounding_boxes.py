import os
import cv2 as cv
import xml.etree.ElementTree as ET

IMAGES_PATH = "dataset/images"
LABELS_PATH = "dataset/labels"


def load_images():
    images = []
    for filename in os.listdir(IMAGES_PATH):
        img = cv.imread(os.path.join(IMAGES_PATH, filename))
        if img is not None:
            images.append(img)
    return images


def load_bounding_boxes():
    boxes = []
    for filename in os.listdir(LABELS_PATH):
        path = os.path.join(LABELS_PATH, filename)
        box = parse_xml(path)
        boxes.append(box)
    return boxes


def parse_xml(xml):
    # create element tree object
    tree = ET.parse(xml)

    # get root element
    root = tree.getroot()

    # stores all boxes of an img
    boxes_in_img = []

    for box in root.findall("./object/bndbox"):
        coords = []
        for coord in box:
            coords.append(coord.text)
        boxes_in_img.append(coords)

    return boxes_in_img


def draw_bounding_boxes(img, boxes):
    for coords in boxes:
        # represents the top left corner of rectangle
        start_point = (int(coords[0]), int(coords[1]))

        # represents the bottom right corner of rectangle
        end_point = (int(coords[2]), int(coords[3]))

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        img = cv.rectangle(img, start_point, end_point, color, thickness)
    return img


def main():
    images = load_images()
    boxes = load_bounding_boxes()

    for idx in range(len(images)):
        images[idx] = draw_bounding_boxes(images[idx], boxes[idx])

    for img in images:
        cv.imshow("Bounding Boxes", img)
        cv.waitKey(0)


if __name__ == "__main__":
    main()
