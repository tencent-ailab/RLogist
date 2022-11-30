# standard library

# 3rd part packages
import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(root_dir, "../"))

import cv2
import openslide
from common import debug_log

from wsi_tissue_segmentation import error_no
from wsi_tissue_segmentation.model import TfModel
from wsi_tissue_segmentation.wsi_tissue_mask import get_tissue_mask
from wsi_tissue_segmentation.get_thumbnail import get_thumbnail_with_target_mpp
from config import gpu_ids

algorithom_logger = debug_log.algorithom_logger


def do_inference(req_id, cache_file_path, result_file_path):

    model_path = "wsi_tissue_segmentation/inference_models/" \
                 "tissue_seg_v8_mpp16.0.pb"

    algorithom_logger.info("req_id:%s, Load input image from %s" % (req_id, cache_file_path))
    algorithom_logger.info(f"req_id:{req_id}, Method wsi tissue segmentation")
    if not os.path.exists(cache_file_path):
        algorithom_logger.error('Input WSI "{}" does not exist.'.format(cache_file_path))
        return error_no.FILE_NOT_EXIST

    try:
        slide = openslide.open_slide(cache_file_path)
    except:
        algorithom_logger.error('{} cannot be read ...'.format(os.path.basename(cache_file_path)))
        return error_no.FILE_CAN_NOT_READ

    tf_model = TfModel(model_path, gpu_id=gpu_ids[0])

    img = get_thumbnail_with_target_mpp(slide, target_mpp=8.0)
    predict, mask_bin = get_tissue_mask(img, tf_model)

    cv2.imwrite(result_file_path, mask_bin)
    algorithom_logger.info("Done 100%")
    algorithom_logger.info("Completed")
