# standard library
import os
import sys
import random
# 3rd part packages
# local source
from read_config import read_yaml


def split_train_fold(tiff_label_list, fold_num, val_basename_list_name):
    # random.shuffle(tiff_label_list)

    all_num = len(tiff_label_list)
    fold_lists = []
    for fold_id in range(fold_num):

        val_key_list = open(f'{val_basename_list_name}_0{fold_id}_val', 'r').readlines()
        val_key_list = [val_key.strip() for val_key in val_key_list]

        val_list = []
        train_list = []
        for tiff_label_line in tiff_label_list:
            tiff_label_line = tiff_label_line.strip()

            elems = tiff_label_line.split(',')
            tiff_name = elems[0]
            label = int(elems[1])

            tiff_key = os.path.basename(tiff_name).rsplit('_', 2)[0]
            print(tiff_key)

            if tiff_key in val_key_list:
                val_list.append(tiff_label_line)
            else:
                train_list.append(tiff_label_line)

        fold_lists.append([train_list, val_list, val_list])

    return fold_lists


if __name__ == '__main__':

    if len(sys.argv) != 5:
        print(f'Usage: '
              f'python3 {sys.argv[0]} '
              f'tiff_label_list_name fold_num(1/2/3..) val_basename_list_name config_file')
        sys.exit(-1)

    tiff_label_list_name = sys.argv[1]
    fold_num = int(sys.argv[2])
    val_basename_list_name = sys.argv[3]
    config_name = sys.argv[4]

    config = read_yaml(config_name)

    tiff_label_list = open(tiff_label_list_name, 'r').readlines()

    sub_dirs = ['train', 'val', 'test']

    fold_lists = split_train_fold(tiff_label_list, fold_num, val_basename_list_name)

    for fold_id, fold_sub_lists in enumerate(fold_lists):
        fold_name = f'{tiff_label_list_name}.fold_num_{fold_num}.fold_{fold_id + 1}'

        result_file_name = f'{fold_name}_split_train_val_test.txt'

        result_file = open(result_file_name, 'w')

        for sub_dir, sub_list in zip(sub_dirs, fold_sub_lists):
            for tiff_label_line in sub_list:
                tiff_label_line = tiff_label_line.strip()
                line_elems = tiff_label_line.split(',')

                if sub_dir == 'train':
                    start_aug_id = 0
                    end_aug_id = 100
                else:
                    start_aug_id = 0
                    end_aug_id = 1

                for aug_id in range(start_aug_id, end_aug_id):
                    print(f'{sub_dir},{tiff_label_line}\n')
                    print(f'{sub_dir},{line_elems[0][:-5]}_{aug_id}.tiff,{line_elems[1]}\n')
                    result_file.write(f'{sub_dir},{line_elems[0][:-5]}_{aug_id}.tiff,{line_elems[1]}\n')
