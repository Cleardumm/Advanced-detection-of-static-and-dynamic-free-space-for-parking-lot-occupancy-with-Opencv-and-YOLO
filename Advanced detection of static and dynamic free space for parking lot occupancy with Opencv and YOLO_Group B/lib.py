import cv2
import numpy as np
import itertools
from math import cos
from collections import defaultdict


def draw_parking_contour(coordinates, image):
    for item in coordinates:
        vertex = np.array([item[0], item[1], item[2], item[3]], np.int32)
        cv2.polylines(image, [vertex], True, (0, 255, 255), 1)

    return image



def intersection_over_union(boxA, boxB):
    # determine coordinates of the intersection box
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])

    # calculate the intersection
    intersectionAB = (x2 - x1) * (y2 - y1)

    # calculate the union
    unionAB = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]) + (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]) - intersectionAB

    iou = intersectionAB / unionAB

    return iou


def is_common_box_slot(parking, box):
    Px1 = parking[0][0]
    Px2 = parking[2][0]
    Py1 = parking[0][1]
    Py2 = parking[2][1]
    Bx1 = box[0]
    Bx2 = box[2]
    By1 = box[1]
    By2 = box[3]

    if Px1 < Bx1 < Px2 and Px1 < Bx2 < Px2 and Py1 < By1 < Py2 and Py1 < By2 < Py2:
        return True
    elif Px1 < Bx1 < Px2 and Px2 < Bx2 < Px2:
        if (By1 < Py1 and Py1 < By2 < Py2) or (Py1 < By1 < Py2 and Py2 < By2):
            return True
    elif Px1 < Bx1 < Px2 and Px1 < Bx2 < Px2 and By1 < Py1 < By2 and By1 < Py2 < By2:
        return True
    else:
        return False


def get_car_boxes(boxes):  # Filter a list of detection results to get only the detected cars
    car_boxes = []

    for item in boxes:
        box = [item[0], item[1], item[0] + item[2], item[1] + item[3]]
        car_boxes.append(box)

    return np.array(car_boxes)


def slot_overlapse(carBoxes):
    iou_arr = []

    for a, b in itertools.combinations(carBoxes, 2):
        if (a - b).any():
            iou = intersection_over_union(a, b)
            iou_arr.append(iou)
        else:
            iou_arr.append(0)

    return np.array(iou_arr)





def count_distance(vertex):
    dX = 0.068

    return (vertex[1][0] - vertex[0][0]) * dX


def generate_parking_keys(parkingsTotal):
    n = 0
    keys = []
    while n < parkingsTotal:
        keys.append("parking" + str(n + 1))
        n = n + 1

    return keys
