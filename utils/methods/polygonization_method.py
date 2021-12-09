import abc
import numpy as np
import cv2


class PolygonizationMethod(metaclass=abc.ABCMeta):
    def __init__(self, ):
        pass

    def prepare_single_house(self, single_contour):
        x_top_left_bb, y_top_left_bb, width_bb, height_bb = cv2.boundingRect(single_contour)
        house = np.zeros((height_bb, width_bb), np.uint8)
        for i in range(len(single_contour)):
            single_contour[i][0][0] -= x_top_left_bb
            single_contour[i][0][1] -= y_top_left_bb
        cv2.drawContours(house, np.array([single_contour]), contourIdx=-1, color=255, thickness=-1)
        return house, x_top_left_bb, y_top_left_bb

    @abc.abstractmethod
    def handle_single_house(self, single_contour):
        pass

