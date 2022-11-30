import cv2
import sys
import os
import numpy as np
import math

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(root_dir, "../"))

from wsi_tissue_segmentation.model import TfModel


def padding_image(image, bord, pad_value=255):
    image = cv2.copyMakeBorder(
        image, bord, bord, bord, bord,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value)
    )
    return image


def crop_patch_and_inference(image, tf_model, patch_size=2048, overlap=256):
    h, w = image.shape[:2]
    image_pad = padding_image(image, overlap)
    y_list = list(range(0, h, patch_size))
    x_list = list(range(0, w, patch_size))
    if y_list[-1] + patch_size >= h:
        y_list[-1] = max(h - patch_size, 0)
    if x_list[-1] + patch_size >= w:
        x_list[-1] = max(w - patch_size, 0)

    result_image = np.zeros((h, w), dtype=np.uint8)

    for y in y_list:
        for x in x_list:
            image_patch = image_pad[
                y:y+patch_size+2*overlap,
                x:x+patch_size+2*overlap]
            patch_mask = tf_model.inference(image_patch)
            mask_h, mask_w = patch_mask.shape[:2]
            result_image[
                y:y+patch_size,
                x:x+patch_size] = patch_mask[overlap:mask_h-overlap,
                                             overlap:mask_w-overlap]
    return result_image


def get_tissue_mask(image, tf_model, mask_thr=127):

    predict = crop_patch_and_inference(image, tf_model)
    _, mask_bin = cv2.threshold(predict, mask_thr, 255, cv2.THRESH_BINARY)
    return predict, mask_bin


if __name__ == '__main__':
    model_path = "./inference_models/tissue_seg_v8_mpp16.0.pb"
    img = cv2.imread("./test_data/img.jpg")
    img = img[:, :, ::-1]

    tf_model = TfModel(model_path)

    predict, mask_bin = get_tissue_mask(img, tf_model)
    cv2.imwrite("./test_data/predict.jpg", predict)
    cv2.imwrite("./test_data/mask_bin.png", mask_bin)
