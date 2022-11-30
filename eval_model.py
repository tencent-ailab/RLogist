# standard library
import os
import sys
from pathlib import Path
# 3rd part packages
import torch
# local source
from datasets import get_split_dataset
from eval_core import eval
from read_config import read_yaml


if __name__ == '__main__':

    if len(sys.argv) != 5:
        print(f'Usage: python3 {sys.argv[0]} input_dir config_file gpu_id ckpt_name')
        sys.exit(-1)

    sub_sets = ['train', 'val', 'test']
    device = torch.device("cuda")
    use_h5 = True

    root_dir = sys.argv[1]
    config_name = sys.argv[2]
    gpu_id = int(sys.argv[3])
    ckpt_name = sys.argv[4]

    os.environ["CUDA_VISIBLE_DEVICES"] = f'{gpu_id}'

    config = read_yaml(config_name)

    feature_extract_model_name = config['feature_extract_model']

    base_path = Path('')

    if feature_extract_model_name == 'resnet50_imagenet_pretrain':
        save_model_dir = os.path.join(root_dir, 'model_resnet_imagenet')
        pos_feature_coord_name = str(base_path / 'abnormal_feature_resnet50_imagenet_pretrain.h5')
        neg_erase_feature_coord_name = str(base_path / 'normal_feature_resnet50_imagenet_pretrain.h5')
        neg_background_feature_coord_name = str(base_path / 'normal_background_feature_resnet50_imagenet_pretrain.h5')
    elif feature_extract_model_name == 'resnet50_pathology_moco_pretrain':
        save_model_dir = os.path.join(root_dir, 'model_moco_patho')
        pos_feature_coord_name = str(base_path / 'abnormal_feature_resnet50_pathology_moco_pretrain.h5')
        neg_erase_feature_coord_name = str(base_path / 'normal_feature_resnet50_pathology_moco_pretrain.h5')
        neg_background_feature_coord_name = str(base_path /
                                                'normal_background_feature_resnet50_pathology_moco_pretrain.h5')
    elif feature_extract_model_name == 'resnet50_imagenet_moco_pretrain':
        save_model_dir = os.path.join(root_dir, 'model_moco_imagenet')
    else:
        raise Exception('feature extract model not define')

    exp_settings = {
        'drop_out': True,
        'model_size': 'small',
        'model_type': 'clam_sb',
        'n_classes': 2,
        'results_dir': save_model_dir,
        'micro_average': False,
    }

    results_dir = exp_settings['results_dir']
    model_type = exp_settings['model_type']
    model_size = exp_settings['model_size']
    drop_out = exp_settings['drop_out']
    n_classes = exp_settings['n_classes']
    micro_average = exp_settings['micro_average']

    config_name = 'config_template.yaml'
    config = read_yaml(config_name)
    feature_extract_model_name = config['feature_extract_model']

    train_dataset, val_dataset, test_dataset = get_split_dataset(
        root_dir, sub_sets, use_h5, pos_feature_coord_name, neg_erase_feature_coord_name,
        neg_background_feature_coord_name)

    model_train, patient_results_train, train_error, train_auc, train_df = eval(
        train_dataset, ckpt_name, dropout=drop_out,
        n_classes=n_classes, model_size=model_size,
        model_type=model_type, device=device, micro_average=micro_average,
        feature_extract_model_name=feature_extract_model_name)

    model_test, patient_results_test, test_error, test_auc, test_df = eval(
        test_dataset, ckpt_name, dropout=drop_out,
        n_classes=n_classes, model_size=model_size,
        model_type=model_type, device=device, micro_average=micro_average,
        feature_extract_model_name=feature_extract_model_name)

    print(patient_results_train, train_error, train_auc)
    print(patient_results_test, test_error, test_auc)

    print(train_df)
    print('train_error: ', train_error)
    print('train_auc: ', train_auc)

    print(test_df)
    print('test_error: ', test_error)
    print('test_auc: ', test_auc)
