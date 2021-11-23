import skimage.io
import os
import numpy as np
import cv2
import config


def binarize(mask):
    threshold = config.THRESHOLD
    mask[mask >= threshold] = config.THRESHOLD_TOP
    mask[mask < threshold] = config.THRESHOLD_BOTTOM

    return mask


def filter_areas(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

    contours_filtered = []
    for contour in contours:
        area = cv2.contourArea(contour)

        if config.MIN_FILTER_AREA < area < config.MAX_FILTER_AREA:
            contours_filtered.append(contour)

    mask_filtered = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(mask_filtered, contours_filtered, contourIdx=-1, color=255, thickness=-1)

    return mask_filtered


def read_and_prepare_mask(filename):
    mask = skimage.io.imread(os.path.join(config.DIR_WITH_DATA, filename), as_gray=True)

    if mask.dtype == 'float64':
        mask = (mask * 255).astype('uint8')

    mask = binarize(mask)
    mask = filter_areas(mask)

    return mask


def save_result(mask, filename):
    filename = filename.split('.')[0] + '.' + config.OUTPUT_EXTENTION
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(config.OUTPUT_DIR, filename)
    skimage.io.imsave(save_path, mask)