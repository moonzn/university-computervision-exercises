"""
APVC - Challenge 6 (Performance of a cat detector)

This file implements the detection of cats in the images of the dataset, using Yolo v5.

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

import cv2
import numpy as np
from draw_bounding_boxes import load_images, load_bounding_boxes, cast_list_to_int, calculate_iou, \
    draw_bounding_boxes

MODEL_FILE = "config/yolov5s.onnx"
CLASS_FILE = "config/object_detection_classes_coco.txt"

DATASET_PATH = "dataset/images"  # Path to the images

CONFIDENCE_THRESHOLD = 0.33  # Threshold for trust in bounding boxes
NMS_THRESHOLD = 0.4  # Threshold for the Non-maximum Suppression algorithm
IOU_THRESHOLD = 0.5  # Threshold for the IoU metric

WIDTH = 640   # Image width on Yolo v5
HEIGHT = 640  # Image height on Yolo v5

# Reading class names
with open(CLASS_FILE, 'r') as f:
    class_names = f.read().split('\n')

# Randomly generate colors for each class
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# Load the model, in this case Yolo v5
YoloModel = cv2.dnn.readNet(MODEL_FILE)
YoloModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

images = load_images()
boxes, total_boxes = load_bounding_boxes()

max_ious = []
false_positives, true_positives = 0, 0

for idx in range(len(images)):
    img = images[idx]
    img_boxes = boxes[idx]
    img_height, img_width, channels = img.shape

    # Conversion factor (actual image dimensions to dimensions read by Yolo)
    x_conversion_factor = img_width / WIDTH
    y_conversion_factor = img_height / HEIGHT

    blob = cv2.dnn.blobFromImage(image=img, scalefactor=1 / 255.0, size=(WIDTH, HEIGHT), swapRB=True)
    YoloModel.setInput(blob)
    predictions = YoloModel.forward()
    outputs = predictions[0]

    bboxes = []
    confidences = []
    classIDs = []

    for i in range(outputs.shape[0]):

        detection = outputs[i, :]

        # Obtain the confidence value of the bounding box
        confidence = detection[4]

        if confidence > CONFIDENCE_THRESHOLD:
            scores = detection[5:]

            classID = np.argmax(scores) + 1
            # Only bounding boxes where the detected objects are cats are considered
            if classID != 16:  # Cat class ID
                continue

            bbox_center_x = detection[0] * x_conversion_factor
            bbox_center_y = detection[1] * y_conversion_factor
            bbox_width = detection[2] * x_conversion_factor
            bbox_height = detection[3] * y_conversion_factor

            bbox_x = int(bbox_center_x - (bbox_width / 2))
            bbox_y = int(bbox_center_y - (bbox_height / 2))

            bboxes.append([bbox_x, bbox_y, int(bbox_width), int(bbox_height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

    # Remove additional bounding boxes using the Non-maximum Suppression algorithm
    idxs = cv2.dnn.NMSBoxes(bboxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(idxs) > 0:
        for i in idxs.flatten():
            iou_values = []

            # Extract the coordinates and dimensions of the resulting bounding boxes
            (bbox_x, bbox_y) = (bboxes[i][0], bboxes[i][1])
            (bbox_w, bbox_h) = (bboxes[i][2], bboxes[i][3])

            # Place rectangles and text to mark the identified objects
            class_name = class_names[classIDs[i]]
            color = (189, 240, 38)
            cv2.rectangle(img, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), color, 2)
            text = "{}: {:.4f}".format(class_name, confidences[i])
            cv2.putText(img, text, (bbox_x, bbox_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Calculating metrics
            for coord in img_boxes:
                prediction_box = [bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h]
                iou = calculate_iou(cast_list_to_int(coord), prediction_box)
                iou_values.append(iou)

            max_iou = max(iou_values)
            if max_iou >= IOU_THRESHOLD:
                true_positives += 1
            else:
                false_positives += 1

            max_ious.append(max_iou)

    # Bounding Boxes from Annotations
    img = draw_bounding_boxes(img, boxes[idx])

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / total_boxes
    average_iou = sum(max_ious) / len(max_ious)

    # Display of metrics in relation to the total bounding boxes of the entire dataset
    print(f"After {idx+1} images:")
    print(f"Precision: {round(precision, 3)}"
          f" | Recall: {round(recall, 3)}"
          f" | Average IoU: {round(average_iou, 3)}")
    print("")

    cv2.imshow('image', img)
    cv2.waitKey(0)
