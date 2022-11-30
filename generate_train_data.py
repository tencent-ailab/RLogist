# standard library
import os
import sys
import time
from multiprocessing import Manager
import multiprocessing
# 3rd part packages
import openslide
import cv2
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
# local source
from read_config import read_yaml
from models.feature_model.load_feature_model import load_resnet_imagenet_model
from models.feature_model.load_feature_model import load_moco_v2_imagenet_model
from models.feature_model.load_feature_model import load_moco_patho_model
from utils import wsi_save, wsi_util
from utils.utils import save_wsi_features_to_h5


def get_wsi_tissue(slide, gpu_id):
    manager = Manager()

    mask, labels, mask_ds, image_thumb = wsi_util.get_foreground_mask_and_components_model_process(
        slide, manager, gpu_id)

    return mask, labels, mask_ds, image_thumb


def get_result_dirs(result_root):
    patch_coord_dir = os.path.join(result_root, 'patch_coord')
    patch_coord_feature_dir = os.path.join(result_root, 'patch_coord_feature')
    patch_feature_dir = os.path.join(result_root, 'patch_feature')
    visual_seg_dir = os.path.join(result_root, 'visual_seg')
    visual_stitch_dir = os.path.join(result_root, 'visual_stitch')

    if not os.path.isdir(patch_coord_dir):
        os.makedirs(patch_coord_dir)
    if not os.path.isdir(patch_coord_feature_dir):
        os.makedirs(patch_coord_feature_dir)
    if not os.path.isdir(patch_feature_dir):
        os.makedirs(patch_feature_dir)
    if not os.path.isdir(visual_seg_dir):
        os.makedirs(visual_seg_dir)
    if not os.path.isdir(visual_stitch_dir):
        os.makedirs(visual_stitch_dir)

    return patch_coord_dir, patch_coord_feature_dir, patch_feature_dir, visual_seg_dir, visual_stitch_dir


def extract_single_feature(
        wsi_name, wsi_slide, tissue_mask, wsi_mpp, patch_coord_dir,
        patch_coord_feature_dir, patch_feature_dir,
        visual_seg_dir, visual_stitch_dir, config, gpu_id=0):

    print(f'wsi_name: {wsi_name}')
    print(f'patch_coord_dir: {patch_coord_dir}')
    print(f'patch_coord_feature_dir: {patch_coord_feature_dir}')
    print(f'patch_feature_dir: {patch_feature_dir}')
    print(f'visual_seg_dir: {visual_seg_dir}')
    print(f'visual_stitch_dir: {visual_stitch_dir}')
    # print(args)
    slide_id = '.'.join(os.path.basename(wsi_name).split('.')[:-1])

    patch_coord_h5 = os.path.join(patch_coord_dir, slide_id + '.h5')
    patch_coord_feature_h5 = os.path.join(patch_coord_feature_dir, slide_id + '.h5')
    patch_feature_pt = os.path.join(patch_feature_dir, slide_id + '.pt')
    vis_seg_name = os.path.join(visual_seg_dir, slide_id + '.png')
    vis_stitch_name = os.path.join(visual_stitch_dir, slide_id + '.jpg')

    extract_single_wsi_feature(
        wsi_slide, tissue_mask, wsi_mpp, vis_seg_name, slide_id, patch_coord_h5,
        patch_coord_feature_h5, patch_feature_pt,
        vis_stitch_name, config, gpu_id=gpu_id)


def extract_single_wsi_feature_process(
        wsi_slide, tissue_mask, wsi_mpp, vis_seg_name, slide_id, patch_coord_h5,
        patch_coord_feature_h5, patch_feature_pt,
        vis_stitch_name, config, gpu_id=0):
    p = multiprocessing.Process(
        target=extract_single_wsi_feature,
        args=(wsi_slide, tissue_mask, wsi_mpp, vis_seg_name, slide_id, patch_coord_h5,
              patch_coord_feature_h5, patch_feature_pt,
              vis_stitch_name, config, gpu_id))
    p.start()
    p.join()


def filter_patch_with_tissue_mask(tissue_mask, x, y, stride, ratio=0.05):
    tissue_patch = tissue_mask[y:y + stride, x:x + stride]
    tissue_patch_nonzero = cv2.countNonZero(tissue_patch)
    tissue_patch_size = tissue_patch.shape[0] * tissue_patch.shape[1]
    if tissue_patch_nonzero / tissue_patch_size < ratio:
        return True
    return False


class CustomImageDataset(Dataset):
    def __init__(self, slide, tissue_mask, wsi_mpp):
        self.slide = slide
        self.tissue_mask = tissue_mask
        self.target_stride_size = 224
        self.target_mpp = 0.50

        self.mask_ds = self.slide.level_dimensions[0][0] / self.tissue_mask.shape[1]

        self.wsi_mpp = wsi_mpp
        self.wsi_stride = int((self.target_stride_size * self.target_mpp) / self.wsi_mpp)

        self.slice_coords = self.get_slice_coords()
        self.h_patch_num = 0
        self.w_patch_num = 0
        print(len(self.slice_coords))

    def patch_transform_compose(self, image):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        transform_func = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_stride_size),

            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD)
        ])
        input_tensor = transform_func(image)
        return input_tensor

    def get_slice_coords(self):
        wsi_w, wsi_h = self.slide.level_dimensions[0]

        wsi_y_list = list(range(0, wsi_h-self.wsi_stride, self.wsi_stride))
        wsi_x_list = list(range(0, wsi_w-self.wsi_stride, self.wsi_stride))
        self.h_patch_num = len(wsi_y_list)
        self.w_patch_num = len(wsi_x_list)

        slice_coords = []
        for y in wsi_y_list:
            for x in wsi_x_list:
                tissue_x = int(x / self.mask_ds)
                tissue_y = int(y / self.mask_ds)
                tissue_stride = int(self.wsi_stride / self.mask_ds)

                if filter_patch_with_tissue_mask(
                        self.tissue_mask, tissue_x, tissue_y, tissue_stride, ratio=0.1):
                    continue
                slice_coords.append([y, x])
        return slice_coords

    def __len__(self):
        return len(self.slice_coords)

    def __getitem__(self, idx):
        [y, x] = self.slice_coords[idx]

        image_patch = np.array(
            self.slide.read_region(
                (x, y),
                0,
                (self.wsi_stride, self.wsi_stride)).convert('RGB'))

        image_h = image_patch.shape[0]
        image_w = image_patch.shape[1]

        aug_images = self.patch_transform_compose(image_patch)
        aug_images = torch.stack([aug_images])
        label_dict = {
            'image_mpp_ori': self.wsi_mpp,
            'image_mpp': self.target_mpp,
            'image_patch_ori': image_patch,
            'image_h_ori': image_h,
            'image_w_ori': image_w,
            'x': x,
            'y': y,
            'idx': idx
        }

        return aug_images, label_dict


def extract_single_wsi_feature(
        wsi_slide, tissue_mask, wsi_mpp, vis_seg_name, slide_id, patch_coord_h5,
        patch_coord_feature_h5, patch_feature_pt,
        vis_stitch_name, config, gpu_id):

    feature_extract_model_name = config['feature_extract_model']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if feature_extract_model_name == 'resnet50_imagenet_pretrain':
        feature_extract_model = load_resnet_imagenet_model(device)
        feature_size = 1024
    elif feature_extract_model_name == 'resnet50_pathology_moco_pretrain':
        feature_extract_model = load_moco_patho_model(device)
        feature_size = 2048
    elif feature_extract_model_name == 'resnet50_imagenet_moco_pretrain':
        feature_extract_model = load_moco_v2_imagenet_model(device)
        feature_size = 2048
    else:
        raise Exception('feature extract model not define')

    cv2.imwrite(vis_seg_name, tissue_mask)

    print(vis_seg_name)
    print(patch_coord_h5)
    print(patch_coord_feature_h5)
    print(patch_feature_pt)
    print(vis_stitch_name)
    print(slide_id)

    dataset = CustomImageDataset(wsi_slide, tissue_mask, wsi_mpp)
    eval_dataloader = DataLoader(dataset, batch_size=16, num_workers=0) # num_workers

    all_features = np.zeros((len(dataset), 1, feature_size), dtype=np.float32)
    all_coords = np.zeros((len(dataset), 2), dtype=np.float32)

    with torch.no_grad():
        for image, label_dict in tqdm(eval_dataloader):

            x_starts = label_dict['x'].cpu().numpy()
            y_starts = label_dict['y'].cpu().numpy()
            idxs = label_dict['idx'].cpu().numpy()

            left_shape = list(image.shape[2:])
            merge_shape = list(image.shape[:2])

            image = image.view([-1] + left_shape)

            feature = feature_extract_model(image.to(device))
            # print(feature.shape)
            feature = feature.cpu().numpy()

            feature = feature.reshape(merge_shape + [-1])
            for sub_ind, idx in enumerate(idxs):
                all_features[idx] = feature[sub_ind]
                all_coords[idx] = [y_starts[sub_ind], x_starts[sub_ind]]

    save_wsi_features_to_h5(patch_coord_feature_h5, all_features, all_coords)

    return


def extract_tiff_list_feature(tiff_list, save_dir, gpu_id=0, config={}):

    patch_coord_dir, patch_coord_feature_dir, \
        patch_feature_dir, visual_seg_dir, \
        visual_stitch_dir = get_result_dirs(save_dir)

    for tiff_name in tqdm(tiff_list):
        tiff_name = tiff_name.split(',')[0].strip()

        wsi_slide = openslide.open_slide(tiff_name)

        print(tiff_name)
        tissue_mask, _, _, image_thumb = get_wsi_tissue(wsi_slide, gpu_id=gpu_id)
        wsi_w, wsi_h = wsi_slide.level_dimensions[0]
        wsi_mpp = wsi_util.get_wsi_mpp(wsi_slide)
        wsi_level_count = 10

        extract_single_feature(
            tiff_name, wsi_slide, tissue_mask, wsi_mpp, patch_coord_dir,
            patch_coord_feature_dir, patch_feature_dir,
            visual_seg_dir, visual_stitch_dir, config,
            gpu_id=gpu_id)


if __name__ == '__main__':

    if len(sys.argv) != 5:
        print(f'Usage: python3 {sys.argv[0]} tiff_list_name save_dir config_file gpu_id')
        sys.exit(-1)

    tiff_list_name = sys.argv[1]
    save_dir = sys.argv[2]
    config_name = sys.argv[3]
    gpu_id = int(sys.argv[4])
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{gpu_id}'

    config = read_yaml(config_name)

    tiff_list = open(tiff_list_name, 'r').readlines()

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.multiprocessing.set_start_method('spawn')
    extract_tiff_list_feature(tiff_list, save_dir, gpu_id, config)
