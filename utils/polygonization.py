import cv2
import numpy as np
from joblib import Parallel, delayed
import config

if __name__ == '__main__':
    from methods.grid_method import GridMethod
else:
    from utils.methods.grid_method import GridMethod

def my_cpu_function(single_contour):
    """
    Handle single contour to find best polygonized contour

    where contour- 3 dimensional ndarray [point_amount x 1 x 2]
    where contour[point_t][0][0] - x coordinate, contour[point_t][0][1] - y coordinate

    Return contour in following format:
    where contour - 3 dimensional ndarray [point_amount x 1 x 2]
    where contour[point_t][0][0] - x coordinate, contour[point_t][0][1] - y coordinate
    """
    if config.POLYGONIZATION_METHOD == 'grid':
        polygonizator = GridMethod()
    else:
        print('Polygonization method not defined. Exit.')
        exit(0)

    polygonized_contour = polygonizator.handle_single_house(single_contour)

    return polygonized_contour


def process_area(mask):
    """
    Transforms mask of area into polygonized contour
    area_mask = binary mask in ndarray H * W
    Returns mask with processed blobs
    """

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

    result_contours = Parallel(n_jobs=config.THREADS)(
        delayed(my_cpu_function)(contour) for contour in contours)

    result = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(result, np.array(result_contours, dtype=object), contourIdx=-1, color=255, thickness=-1)

    return result
