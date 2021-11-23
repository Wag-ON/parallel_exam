import numpy as np
import math


def calc_dist(point_1, point_2):
    """
    Returns L2 distance between two points [x, y]
    """

    return np.sqrt((np.abs(point_1[0] - point_2[0])) ** 2 + (np.abs(point_1[1] - point_2[1])) ** 2)


def calc_squared_dist(point_1, point_2):
    """
    Returns square of L2 distance between two points [x, y]
    """

    return (np.abs(point_1[0] - point_2[0])) ** 2 + (np.abs(point_1[1] - point_2[1])) ** 2


def calc_iou(mask, polygonized):
    """
    Calculates IOU score IOU score between polygonized hose mask and original house mask
    """

    intersection = np.logical_and(mask, polygonized)
    union = np.logical_or(mask, polygonized)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


def calc_angle_of_line(point_1, point_2):
    """
    Calculates angle between Oy-axis and line formed by two points
    point [x, y]

    Return angle in 0 - 90 degrees
    """

    if (point_2[0] - point_1[0] == 0) or (point_2[1] - point_1[1] == 0):
        return 0

    k = (point_2[0] - point_1[0]) / (point_2[1] - point_1[1])

    angle_in_rad = math.atan(k)
    angle_in_degrees = math.degrees(angle_in_rad)

    if angle_in_degrees >= 90:
        angle_in_degrees -= 90
    elif angle_in_degrees < 0:
        angle_in_degrees += 90

    return angle_in_degrees


def longest_side(building_peck):
    """
    Finding longest side of house contour

    building_peck =  polygon in following format:
                    polygon = [point_1, ... point_m]
                    where point = [x, y]

    Returns
    two nodes with format [x, y]
    distance between them

    """
    node_from = building_peck[0][0]
    node_to = building_peck[1][0]
    max_dist = 0
    for i in range(len(building_peck) - 1):
        dist = calc_dist(building_peck[i][0], building_peck[i + 1][0])
        if dist > max_dist:
            node_from = building_peck[i][0]
            node_to = building_peck[i + 1][0]
            max_dist = dist
    dist = calc_dist(building_peck[0][0], building_peck[-1][0])
    if dist > max_dist:
        node_from = building_peck[0][0]
        node_to = building_peck[-1][0]
        max_dist = dist

    return node_from, node_to, max_dist