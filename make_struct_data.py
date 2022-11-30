# standard library
import os
import sys
import random
# 3rd part packages
# local source
from read_config import read_yaml


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


def split_train_fold(tiff_label_list, fold_num, val_ratio):
    random.shuffle(tiff_label_list)

    all_num = len(tiff_label_list)
    val_num = int(all_num * val_ratio)
    fold_lists = []
    for fold_id in range(fold_num):
        val_start_id = fold_id * val_num
        val_end_id = (fold_id + 1) * val_num
        val_list = tiff_label_list[val_start_id:val_end_id]
        train_list = tiff_label_list[:val_start_id] + tiff_label_list[val_end_id:]

        fold_lists.append([train_list, val_list, val_list])

    return fold_lists


def generate_one_set(save_fold_sub_label_0_dir, save_fold_sub_label_1_dir, tiff_name_label, feature_dir):

    base_dirs = [
        ['patch_coord_feature', 'h5'],
        ['visual_seg', 'png']]

    print(tiff_name_label)
    elems = tiff_name_label.rsplit(',', 1)
    tiff_name = elems[0]
    label = int(elems[1])

    tiff_key = os.path.splitext(os.path.basename(tiff_name))[0]

    for base_dir, base_ext in base_dirs:
        src_file = os.path.join(feature_dir, base_dir, f'{tiff_key}.{base_ext}')
        if label == 0:
            save_fold_sub_label_dir = save_fold_sub_label_0_dir
        else:
            save_fold_sub_label_dir = save_fold_sub_label_1_dir
        link_file = os.path.join(save_fold_sub_label_dir, base_dir, f'{tiff_key}.{base_ext}')
        os.symlink(src_file, link_file)


def generate_one_fold(tiff_label_split_list, save_struct_dir, fold_id, feature_dir, config={}):

    fold_name = f'fold_{fold_id}'
    save_fold_dir = os.path.join(save_struct_dir, fold_name)
    if not os.path.isdir(save_fold_dir):
        os.makedirs(save_fold_dir)

    for tiff_label_line in tiff_label_split_list:
        tiff_label_line = tiff_label_line.strip()
        elems = tiff_label_line.split(',', 1)
        sub_dir = elems[0]
        tiff_name_label = elems[1]
        save_fold_sub_dir = os.path.join(save_fold_dir, sub_dir)
        if not os.path.isdir(save_fold_sub_dir):
            os.makedirs(save_fold_sub_dir)

        save_fold_sub_label_0_dir = os.path.join(save_fold_sub_dir, '0')
        if not os.path.isdir(save_fold_sub_label_0_dir):
            os.makedirs(save_fold_sub_label_0_dir)
        save_fold_sub_label_1_dir = os.path.join(save_fold_sub_dir, '1')
        if not os.path.isdir(save_fold_sub_label_1_dir):
            os.makedirs(save_fold_sub_label_1_dir)

        get_result_dirs(save_fold_sub_label_0_dir)
        get_result_dirs(save_fold_sub_label_1_dir)

        generate_one_set(save_fold_sub_label_0_dir, save_fold_sub_label_1_dir, tiff_name_label, feature_dir)


if __name__ == '__main__':

    if len(sys.argv) != 6:
        print(f'Usage: '
              f'python3 {sys.argv[0]} '
              f'tiff_label_split_list_name save_struct_dir feature_dir fold_id config_file')
        sys.exit(-1)

    tiff_label_split_list_name = sys.argv[1]
    save_struct_dir = sys.argv[2]
    feature_dir = sys.argv[3]
    fold_id = int(sys.argv[4])
    config_name = sys.argv[5]

    config = read_yaml(config_name)

    tiff_label_split_list = open(tiff_label_split_list_name, 'r').readlines()

    generate_one_fold(tiff_label_split_list, save_struct_dir, fold_id, feature_dir, config)
