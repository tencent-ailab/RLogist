import re
import multiprocessing
import cv2
import numpy as np
from skimage import morphology, measure
import os

UNCERTAINTY_THRESHOLD = 0.85

# ############ Common WSI functions ####################

def get_wsi_mpp(slide):
    wsi_info = dict(slide.properties)
    if 'openslide.mpp-x' in wsi_info:
        mpp = round(float(wsi_info['openslide.mpp-x']), 4)
    elif 'tiff.XResolution' in wsi_info:
        res = float(wsi_info['tiff.XResolution'])
        unit = str(wsi_info['tiff.ResolutionUnit'])
        if unit == 'centimeter':
            mpp = round(10000 / res, 4)
        elif unit == 'inch':
            mpp = round(25400 / res, 4)
        else:
            raise ValueError('unsupported ResolutionUnit')
    else:
        raise ValueError('unsupported wsi type')
    return mpp


def get_level_mpp(slide, level):
    wsi_info = dict(slide.properties)
    # print(wsi_info)
    [w, _] = slide.level_dimensions[0]
    mpp = round(float(wsi_info['openslide.mpp-x']), 4)

    [level_w, _] = slide.level_dimensions[level]
    level_ratio = float(w) / level_w
    level_mpp = round(float(mpp * level_ratio), 3)
    return level_mpp


def get_tiff_level_mpp(slide, level):
    wsi_info = dict(slide.properties)
    [w, _] = slide.level_dimensions[0]
    res = float(wsi_info['tiff.XResolution'])
    unit = str(wsi_info['tiff.ResolutionUnit'])
    if unit == 'centimeter':
        mpp = round(10000/res, 4)
    elif unit == 'inch':
        mpp = round(25400 / res, 4)
    else:
        raise ValueError('Unsupported ResolutionUnit')

    [level_w, _] = slide.level_dimensions[level]
    level_ratio = float(w) / level_w
    level_mpp = round(float(mpp * level_ratio), 3)
    return level_mpp


def get_mask_target_level(slide, mask_downsamples):
    ret_level = 0
    for level, downsample in enumerate(slide.level_downsamples):
        if slide.level_dimensions[level][0] < 1000 or \
                slide.level_dimensions[level][1] < 1000:
            break
        if downsample >= mask_downsamples:
            return level
        ret_level = level
    return ret_level


def get_foreground_mask(slide, level=5):

    pimg = slide.read_region((0, 0),
                             level,
                             (slide.level_dimensions[level][0],
                             slide.level_dimensions[level][1])).convert('RGB')

    rgb = np.array(pimg)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower_red = np.array([5, 5, 5])
    upper_red = np.array([220, 220, 220])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask),
                                   cv2.MORPH_CLOSE,
                                   close_kernel)
    open_kernel = np.ones((20, 20), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close),
                                  cv2.MORPH_OPEN,
                                  open_kernel)
    contours, _ = cv2.findContours(image_open,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # bounding_boxes = [cv2.boundingRect(c) for c in contours]

    segmask = np.zeros((hsv.shape[0], hsv.shape[1]), np.uint8)
    cv2.drawContours(segmask, contours, -1, (255), -1)

    # test user draw mask
    # cv2.imwrite('segmask.png', segmask)
    # slide_mask = np.zeros_like(segmask)
    # slide_mask[900:1100, 300:500] = 255
    # cv2.imwrite('slide_mask.png', slide_mask)
    # segmask = slide_mask
    # test over

    kernel = np.ones((35, 35), dtype=np.uint8)
    dialte_segmask = cv2.dilate(segmask, kernel, iterations=1)

    contours, _ = cv2.findContours(dialte_segmask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    dialte_segmask = np.zeros((hsv.shape[0], hsv.shape[1]), np.uint8)
    cv2.drawContours(dialte_segmask, contours, -1, (255), -1)

    return segmask, dialte_segmask


def eliminate_small_labels(labels):

    ll = np.unique(labels)
    num = []
    sl = []
    for l in ll[1:]:
        if np.sum(labels == l) < 0.002 * labels.shape[0]*labels.shape[1]:
            labels[labels == l] = 0
        else:
            num.append(np.sum(labels == l))
            sl.append(l)

    sort_index = np.argsort(num)[::-1]
    sl = np.array(sl)
    sort_label = sl[sort_index]
    given_label = 1
    new_labels = labels.copy()
    for s in sort_label:
        new_labels[labels == s] = given_label
        given_label += 1

    return new_labels


def get_foreground_mask_and_components(slide, mask_downsamples):

    mask_target_level = get_mask_target_level(slide, mask_downsamples)
    mask_ds = slide.level_downsamples[mask_target_level]
    mask, dialte_segmask = get_foreground_mask(slide, level=mask_target_level)

    labels = measure.label(dialte_segmask, background=0)
    labels = eliminate_small_labels(labels)
    return mask, labels, mask_ds


def get_foreground_mask_and_components_model_process(slide, manager, gpu_id):

    tissue_seg_dict = manager.dict()
    # p = multiprocessing.Process(
    #     target=get_foreground_mask_and_components_model,
    #     args=(slide, tissue_seg_dict, gpu_id))
    # p.start()
    # p.join()
    get_foreground_mask_and_components_model(slide, tissue_seg_dict, gpu_id)
    mask = tissue_seg_dict['mask']
    labels = tissue_seg_dict['labels']
    mask_ds = tissue_seg_dict['mask_ds']
    image_thumb = tissue_seg_dict['image_thumb']
    # p.terminate()
    return mask, labels, mask_ds, image_thumb


def get_foreground_mask_and_components_model(
        slide, tissue_seg_dict, gpu_id=0):
    from wsi_tissue_segmentation.model import TfModel
    from wsi_tissue_segmentation.wsi_tissue_mask import get_tissue_mask
    from wsi_tissue_segmentation.get_thumbnail import get_thumbnail_with_target_mpp
    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root_dir, "../wsi_tissue_segmentation/inference_models/tissue_seg_v8_mpp16.0.pb")

    tf_model = TfModel(model_path, gpu_id=gpu_id)
    img_thumb = get_thumbnail_with_target_mpp(slide, target_mpp=8.0)
    thumb_prob, thumb_mask = get_tissue_mask(img_thumb, tf_model)
    tf_model.release_model()
    for i in range(15):
        thumb_prob = cv2.medianBlur(thumb_prob, 7)
        _, thumb_mask = cv2.threshold(thumb_prob, 50, 255, cv2.THRESH_BINARY)

    wsi_w = slide.level_dimensions[0][0]
    mask_w = thumb_mask.shape[1]
    mask_ds = wsi_w / mask_w

    mask = thumb_mask

    kernel = np.ones((15, 15), dtype=np.uint8)
    dialte_segmask = cv2.dilate(mask, kernel, iterations=2)
    # dialte_segmask = cv2.erode(mask, kernel, iterations=2)
    mask = dialte_segmask

    contours, _ = cv2.findContours(dialte_segmask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    dialte_segmask = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    cv2.drawContours(dialte_segmask, contours, -1, (255), -1)

    labels = measure.label(dialte_segmask, background=0)
    labels = eliminate_small_labels(labels)

    print(f'slide shape = {slide.level_dimensions[0]}')
    print(f'slide mpp = {get_wsi_mpp(slide)}')
    print(f'mask shape = {mask.shape}')
    print(f'labels shape = {labels.shape}')
    print(f'mask mpp = 8.0')
    print(f'mask_ds = {mask_ds}')

    tissue_seg_dict['mask'] = mask
    tissue_seg_dict['labels'] = labels
    tissue_seg_dict['mask_ds'] = mask_ds
    tissue_seg_dict['image_thumb'] = img_thumb
    return mask, labels, mask_ds


def align_max_size(mask, max_x, max_y, w, h, mask_ds):

    pad_x = int((max_x - w)/mask_ds)
    pad_y = int((max_y - h)/mask_ds)

    tmp = np.zeros((mask.shape[0]+pad_y, mask.shape[1]+pad_x), dtype='uint8')
    tmp[:mask.shape[0], :mask.shape[1]] = mask
    return tmp


def add_mask_padding(mask, scale_pad):

    tmp = np.zeros((mask.shape[0]+2*scale_pad, mask.shape[1]+2*scale_pad), dtype='uint8')
    h, w = tmp.shape[:2]
    tmp[scale_pad:h - scale_pad, scale_pad:w - scale_pad] = mask
    return tmp


def process_mask_labels(mask, labels, max_x, max_y, w, h, mask_ds, pad):
    mask = align_max_size(mask, max_x, max_y, w, h, mask_ds)
    labels = align_max_size(labels, max_x, max_y, w, h, mask_ds)

    scale_pad = int(pad/mask_ds)
    mask_pad = add_mask_padding(mask, scale_pad)
    labels_pad = add_mask_padding(labels, scale_pad)
    return mask_pad, labels_pad


def central_crop(image, pad):
    h, w = image.shape[:2]
    return image[pad:h - pad, pad:w - pad]


# ############ Patch extraction functions ####################
def fill_black_bg_as_white(patch_region):

    mask_tmp = cv2.inRange(patch_region, (0, 0, 0), (1, 1, 1))
    kernel = np.ones((5, 5), np.uint8)
    mask_tmp = cv2.dilate(mask_tmp, kernel, iterations=1)
    mask_tmp = morphology.remove_small_objects(
        mask_tmp.astype(bool),
        1000).astype('uint8')*255
    patch_region[mask_tmp != 0] = [255, 255, 255]
    return patch_region


def extract_patch_with_padding_components(
        x, y, slide,
        foreground_mask,
        components_labels,
        mask_ds=2,
        level=0,
        stride=3008,
        pad=256):

    patch_region = np.array(
        slide.read_region((x-pad, y-pad),
                          level,
                          (stride+2*pad, stride+2*pad)).convert('RGB'))
    patch_region = cv2.cvtColor(patch_region, cv2.COLOR_RGBA2BGR)

    # fill the black background area as white
    patch_region = fill_black_bg_as_white(patch_region)

    # convert slide coordinates into mask coordinates
    mask_stride = int(stride/mask_ds)
    mask_pad = int(pad/mask_ds)
    mask_x = int(x/mask_ds) + mask_pad
    mask_y = int(y/mask_ds) + mask_pad

    # extract foreground patch and check whether it is blank
    forground_patch = foreground_mask[mask_y-mask_pad:mask_y+mask_stride+mask_pad,
                                      mask_x-mask_pad:mask_x+mask_stride+mask_pad]
    is_blank = True
    mh, mw = forground_patch.shape[:2]
    if len(np.nonzero(forground_patch)[0]) > int(0.1* mh * mw):
        is_blank = False

    is_write = True
    if len(np.nonzero(forground_patch)[0]) < int(0.5* mh * mw):
        is_write = False

    clabel_patch = components_labels[mask_y-mask_pad:mask_y+mask_stride+mask_pad,
                                     mask_x-mask_pad:mask_x+mask_stride+mask_pad]
    # print(clabel_patch.shape)

    clabel_patch = cv2.resize(clabel_patch,
                              (patch_region.shape[0], patch_region.shape[1]),
                              interpolation=cv2.INTER_NEAREST)

    return patch_region, is_blank, is_write, clabel_patch


def extract_patch_no_pad(
        x, y, slide,
        foreground_mask,
        components_labels,
        level=0,
        mask_ds=2,
        stride=3008,
        pad=256):

    wsi_x = x
    wsi_y = y

    patch_region = np.array(
        slide.read_region((wsi_x, wsi_y),
                          level,
                          (stride, stride)).convert('RGB'))
    patch_region = cv2.cvtColor(patch_region, cv2.COLOR_RGBA2BGR)

    # fill the black background area as white
    patch_region = fill_black_bg_as_white(patch_region)

    # convert slide coordinates into mask coordinates
    mask_stride = int(stride/mask_ds)
    mask_pad = int(pad/mask_ds)
    mask_x = int(wsi_x/mask_ds) + mask_pad
    mask_y = int(wsi_y/mask_ds) + mask_pad

    # extract foreground patch and check whether it is blank
    forground_patch = foreground_mask[mask_y-mask_pad:mask_y+mask_stride+mask_pad,
                                      mask_x-mask_pad:mask_x+mask_stride+mask_pad]
    is_blank = True
    mh, mw = forground_patch.shape[:2]
    if len(np.nonzero(forground_patch)[0]) > int(0.05 * mh * mw):
        is_blank = False

    is_write = True
    if len(np.nonzero(forground_patch)[0]) < int(0.5 * mh * mw):
        is_write = False

    clabel_patch = components_labels[mask_y-mask_pad:mask_y+mask_stride+mask_pad,
                                     mask_x-mask_pad:mask_x+mask_stride+mask_pad]
    # print(clabel_patch.shape)

    clabel_patch = cv2.resize(clabel_patch,
                              (patch_region.shape[1], patch_region.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    return patch_region, is_blank, is_write, clabel_patch


def extract_patch_with_target_mpp(
        target_x, target_y, slide,
        foreground_mask,
        components_labels,
        mask_ds=2,
        stride=3008,
        pad=256,
        wsi_mpp=0.25,
        best_level=0,
        best_mpp=0.25,
        target_mpp=0.848,
        target_stride=2048,
        target_pad=256,
        target_w=2048,
        target_h=2048,
        target_size=2560):

    wsi_x = int(target_x * target_mpp / wsi_mpp)
    wsi_y = int(target_y * target_mpp / wsi_mpp)
    wsi_pad = int(target_pad * target_mpp / wsi_mpp)

    target_read_w = target_read_h = target_size

    best_read_w = int(target_read_w * target_mpp / best_mpp)
    best_read_h = int(target_read_h * target_mpp / best_mpp)

    patch_region = np.array(
        slide.read_region((wsi_x-wsi_pad, wsi_y-wsi_pad),
                          best_level,
                          (best_read_w, best_read_h)).convert('RGB'))
    patch_region = cv2.cvtColor(patch_region, cv2.COLOR_RGBA2BGR)
    patch_region = cv2.resize(
        patch_region, (target_read_w, target_read_h),
        interpolation=cv2.INTER_LINEAR)
    # fill the black background area as white
    patch_region = fill_black_bg_as_white(patch_region)

    # convert slide coordinates into mask coordinates
    mask_stride = int(stride/mask_ds)
    mask_pad = int(pad/mask_ds)
    mask_x = int(wsi_x/mask_ds) + mask_pad
    mask_y = int(wsi_y/mask_ds) + mask_pad

    # extract foreground patch and check whether it is blank
    forground_patch = foreground_mask[mask_y-mask_pad:mask_y+mask_stride+mask_pad,
                                      mask_x-mask_pad:mask_x+mask_stride+mask_pad]
    is_blank = True
    mh, mw = forground_patch.shape[:2]
    if len(np.nonzero(forground_patch)[0]) > int(0.05 * mh * mw):
        is_blank = False

    is_write = True
    if len(np.nonzero(forground_patch)[0]) < int(0.5 * mh * mw):
        is_write = False

    clabel_patch = components_labels[mask_y-mask_pad:mask_y+mask_stride+mask_pad,
                                     mask_x-mask_pad:mask_x+mask_stride+mask_pad]
    # print(clabel_patch.shape)

    clabel_patch = cv2.resize(clabel_patch,
                              (patch_region.shape[1], patch_region.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    return patch_region, is_blank, is_write, clabel_patch


def extract_mask_with_target_mpp(
        target_x, target_y, target_mpp, wsi_mpp,
        slide_mask,
        mask_ds=2,
        stride=3008,
        pad=256):

    # convert slide coordinates into mask coordinates
    wsi_x = int(target_x * target_mpp / wsi_mpp)
    wsi_y = int(target_y * target_mpp / wsi_mpp)
    mask_stride = int(stride/mask_ds)
    mask_pad = int(pad/mask_ds)
    mask_x = int(wsi_x/mask_ds) + mask_pad
    mask_y = int(wsi_y/mask_ds) + mask_pad

    patch_size = (stride+2*pad, stride+2*pad)

    # extract foreground patch and check whether it is blank
    slide_mask_patch_down = slide_mask[mask_y-mask_pad:mask_y+mask_stride+mask_pad,
                                       mask_x-mask_pad:mask_x+mask_stride+mask_pad]

    slide_mask_patch = cv2.resize(slide_mask_patch_down, patch_size,
                                  interpolation=cv2.INTER_NEAREST)

    return slide_mask_patch


def extract_mask_with_padding(
        x, y,
        slide_mask,
        mask_ds=2,
        stride=3008,
        pad=256):

    # convert slide coordinates into mask coordinates
    wsi_x = x
    wsi_y = y
    mask_stride = int(stride/mask_ds)
    mask_pad = int(pad/mask_ds)
    mask_x = int(wsi_x/mask_ds) + mask_pad
    mask_y = int(wsi_y/mask_ds) + mask_pad

    patch_size = (stride+2*pad, stride+2*pad)

    # extract foreground patch and check whether it is blank
    slide_mask_patch_down = slide_mask[mask_y-mask_pad:mask_y+mask_stride+mask_pad,
                                       mask_x-mask_pad:mask_x+mask_stride+mask_pad]

    slide_mask_patch = cv2.resize(slide_mask_patch_down, patch_size,
                                  interpolation=cv2.INTER_NEAREST)

    return slide_mask_patch


def extract_patch_with_padding(x, y, slide, mask,
                               mask_ds=2,
                               level=0,
                               stride=3008,
                               crop_size=3008,
                               pad=256):

    mask_x = int(x/mask_ds)
    mask_y = int(y/mask_ds)

    mask_stride = int(stride/mask_ds)
    mask_label = mask[mask_y:mask_y+mask_stride,
                      mask_x:mask_x+mask_stride]

    patch_region = np.array(
                        slide.read_region((x-pad, y-pad),
                        level,
                        (crop_size+2*pad, crop_size+2*pad)).convert('RGB'))
    patch_region = cv2.cvtColor(patch_region, cv2.COLOR_RGBA2BGR)

    # fill the black background area as white
    patch_region = fill_black_bg_as_white(patch_region)

    is_blank = True
    mh, mw = mask_label.shape[:2]
    if len(np.nonzero(mask_label)[0]) > int(0.1* mh * mw):
        is_blank = False
    return patch_region, is_blank


def extract_patch_with_padding_no_fill(
        x, y, slide,
        level=0,
        crop_size=3008,
        pad=256):

    patch_region = np.array(
        slide.read_region(
            (x-pad, y-pad),
            level,
            (crop_size+2*pad, crop_size+2*pad)).convert('RGB'))
    # cv2.imwrite('patch_region.png', patch_region)
    patch_region = cv2.cvtColor(patch_region, cv2.COLOR_RGBA2GRAY)
    return patch_region


# ############ Uncertainty functions ####################
def cal_uncertainty(prob, threshold):

    _, mask = cv2.threshold(prob, 127, 255, cv2.THRESH_BINARY)
    prob = prob.astype('float') / 255
    mask = mask.astype('int') / 255

    is_high = False
    if np.mean(prob[mask == 1]) < threshold:
        is_high = True

    return is_high


# ############ CPS region functions ####################
def merge_image_list_to_big_image(image_list, image_mpp, target_mpp, max_x, crop_size):
    n_hori_tiles = int(max_x / crop_size)
    n_verti_tiles = int(len(image_list) / n_hori_tiles)
    normal_image_h = int(float(crop_size * image_mpp) / target_mpp)
    normal_image_w = normal_image_h
    right_edge_image_w = int(
        float(image_list[-1].shape[1] * image_mpp) / target_mpp)
    bottom_edge_image_h = int(
        float(image_list[-1].shape[0] * image_mpp) / target_mpp)
    big_image_h = int((int(len(image_list) / n_hori_tiles) - 1
                       ) * normal_image_h + bottom_edge_image_h)
    big_image_w = int((n_hori_tiles - 1) * normal_image_w +
                      right_edge_image_w)
    if len(image_list[0].shape) == 3 and image_list[0].shape[2] == 3:
        big_image = np.zeros((big_image_h, big_image_w, 3), dtype=np.uint8)
    else:
        big_image = np.zeros((big_image_h, big_image_w), dtype=np.uint8)

    resize_list = []
    for i, image in enumerate(image_list):
        x_start = int(i % n_hori_tiles) * normal_image_w
        y_start = int(i // n_hori_tiles) * normal_image_h
        resize_h = normal_image_h
        resize_w = normal_image_w
        if (i % n_hori_tiles) == n_hori_tiles - 1:
            resize_w = right_edge_image_w
        if (i / n_hori_tiles) == n_verti_tiles:
            resize_h = bottom_edge_image_h
        # print(image.shape)
        # print(resize_w, resize_h)
        image_resize = cv2.resize(
            image, (resize_w, resize_h),
            interpolation=cv2.INTER_NEAREST)
        big_image[y_start:y_start+resize_h,
                  x_start:x_start+resize_w] = image_resize
        resize_list.append(
            [x_start, y_start,
             resize_w, resize_h,
             image.shape[1], image.shape[0]])
    return big_image, resize_list


def resize_small_patch_to_origin_size(big_image, resize_list):
    origin_image_list = []
    for resize_info in resize_list:
        x_start = resize_info[0]
        y_start = resize_info[1]
        now_w = resize_info[2]
        now_h = resize_info[3]
        ori_w = resize_info[4]
        ori_h = resize_info[5]
        image_patch = big_image[
            y_start:y_start+now_h,
            x_start:x_start+now_w]
        origin_patch = cv2.resize(
            image_patch, (ori_w, ori_h),
            interpolation=cv2.INTER_NEAREST)
        origin_image_list.append(origin_patch)
    return origin_image_list


# ############ Write WSI function ####################
def write_wsi(
        level_mpp, n_hori_tiles, images_list,
        tiff_name, tile_size=256):
    import pyvips
    pv_image_list = []
    for image in images_list:
        height, width = image.shape[:2]
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            channels = 3
        elif image.shape[2] == 3:
            channels = 3
        elif image.shape[2] == 4:
            channels = 4

        im_size = height * width * channels

        linear = image.reshape(im_size)
        pv_image_list.append(
            pyvips.Image.new_from_memory(
                linear.data, width, height,
                channels, 'uchar'))

    resolution = 1e3 / level_mpp
    join = pyvips.Image.arrayjoin(pv_image_list, across=n_hori_tiles)
    join.write_to_file(
        tiff_name,
        # tile=True, compression='jpeg',
        tile=True, compression='lzw',
        bigtiff=True, pyramid=True,
        tile_width=tile_size, tile_height=tile_size,
        xres=resolution, yres=resolution)
