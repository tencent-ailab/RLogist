import cv2
import numpy as np
import openslide
import re
import os
import sys
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(file_dir, "../"))
from utils import wsi_util


def get_thumbnail_with_target_mpp(slide, target_mpp=8.0):
    wsi_info = dict(slide.properties)
    level0_mpp = wsi_util.get_wsi_mpp(slide)

    target_downsamle = target_mpp / level0_mpp

    best_level = slide.get_best_level_for_downsample(target_downsamle)

    best_w = slide.level_dimensions[best_level][0]
    best_h = slide.level_dimensions[best_level][1]

    best_thumbnail = np.array(slide.read_region(
        (0, 0), best_level, (best_w, best_h)).convert('RGB'))

    target_h = int(level0_mpp * slide.level_dimensions[0][1] / target_mpp)
    target_w = int(level0_mpp * slide.level_dimensions[0][0] / target_mpp)

    target_thumbnail = cv2.resize(best_thumbnail, (target_w, target_h))

    return target_thumbnail


if __name__ == '__main__':
    wsi_name = "../wsi_registration/local_registration/18-1241-893728-ki67_reg_inference.tiff"
    slide = openslide.open_slide(wsi_name)
    print(slide.level_dimensions[0])
    img = get_thumbnail_with_target_mpp(slide, target_mpp=2.0)
    print(img.shape[0])
    cv2.imwrite("../wsi_registration/local_registration/18-1241-893728-ki67_reg_inference.jpg",
                img[:, :, ::-1])
