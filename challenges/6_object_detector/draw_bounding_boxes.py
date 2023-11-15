import os
import cv2 as cv
import xml.etree.ElementTree as ET

IMAGES_PATH = "dataset/images"
ANNOT_PATH = "dataset/annotations"


def load_images():
    images = []
    for filename in os.listdir(IMAGES_PATH):
        img = cv.imread(os.path.join(IMAGES_PATH, filename))
        if img is not None:
            images.append(img)
    return images


def load_bounding_boxes():
    boxes = []
    for filename in os.listdir(ANNOT_PATH):
        path = os.path.join(ANNOT_PATH, filename)
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
        color = (135, 135, 135)

        # Line thickness of 2 px
        thickness = 2

        img = cv.rectangle(img, start_point, end_point, color, thickness)
    return img


def cast_list_to_int(lst):
    return list(map(int, lst))


def calculate_iou(box1, box2):
    x_inter1 = max(box1[0], box2[0])
    y_inter1 = max(box1[1], box2[1])
    x_inter2 = min(box1[2], box2[2])
    y_inter2 = min(box1[3], box2[3])

    if x_inter2 < x_inter1 or y_inter2 < y_inter1:
        return 0

    # Área da interseção
    width_inter = abs(x_inter2 - x_inter1)
    height_inter = abs(y_inter2 - y_inter1)
    area_inter = width_inter * height_inter

    # Área da união
    width_ground_truth = abs(box1[2] - box1[0])
    height_ground_truth = abs(box1[3] - box1[1])
    area_ground_truth = width_ground_truth * height_ground_truth
    width_prediction = abs(box2[2] - box2[0])
    height_prediction = abs(box2[3] - box2[1])
    area_detection = width_prediction * height_prediction
    area_union = area_detection + area_ground_truth - area_inter

    # Cálculo do IoU
    return area_inter / area_union


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
