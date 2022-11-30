# standard library
import os
import sys
from pathlib import Path
# 3rd part packages
import torch
import numpy as np
import random
# local source
from datasets import get_split_dataset
from train_core import train
from read_config import read_yaml


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f'Usage: python3 {sys.argv[0]} input_dir config_file gpu_id')
        sys.exit(-1)

    root_dir = sys.argv[1]

    config_name = sys.argv[2]

    gpu_id = int(sys.argv[3])

    os.environ["CUDA_VISIBLE_DEVICES"] = f'{gpu_id}'

    init_seeds(seed=40)

    config = read_yaml(config_name)

    loss_func = config['loss_func']
    learning_rate = config['learning_rate']
    lr_scheduler = config['lr_scheduler']
    mil_model = config['mil_model']
    num_class = config['class_num']
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

    if mil_model == 'CLAM_SB':
        model_type = 'clam_sb'
    elif mil_model == 'CLAM_MB':
        model_type = 'clam_mb'
    else:
        raise Exception('mil model type not define')

    if loss_func == 'CrossEntropyLoss':
        loss_type = 'ce'
    elif loss_func == 'svm':
        loss_type = 'svm'
    else:
        raise Exception('loss func type not define')

    sub_sets = ['train', 'val', 'test']
    fold = 1
    device = torch.device("cuda")
    use_h5 = True

    exp_settings = {
        'B': 8,
        'bag_loss': loss_type,
        'bag_weight': 0.7,
        'drop_out': True,
        'early_stopping': True,
        'inst_loss': 'svm',
        'log_data': True,
        'lr': learning_rate,
        'max_epochs': 200,
        'model_size': 'small',
        'model_type': model_type,
        'n_classes': num_class,
        'no_inst_cluster': False,
        'opt': 'adam',
        'reg': 1e-05,
        'results_dir': save_model_dir,
        'subtyping': False,
        'testing': False,
        'weighted_sample': True,
    }

    B = exp_settings['B']
    bag_loss = exp_settings['bag_loss']
    bag_weight = exp_settings['bag_weight']
    drop_out = exp_settings['drop_out']
    early_stopping = exp_settings['early_stopping']
    inst_loss = exp_settings['inst_loss']
    log_data = exp_settings['log_data']
    lr = exp_settings['lr']
    max_epochs = exp_settings['max_epochs']
    model_size = exp_settings['model_size']
    model_type = exp_settings['model_type']
    n_classes = exp_settings['n_classes']
    no_inst_cluster = exp_settings['no_inst_cluster']
    opt = exp_settings['opt']
    reg = exp_settings['reg']
    results_dir = exp_settings['results_dir']
    subtyping = exp_settings['subtyping']
    testing = exp_settings['testing']
    weighted_sample = exp_settings['weighted_sample']

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    exp_setting_writer = open(os.path.join(results_dir, 'exp_setting.txt'), 'w')
    for key, value in exp_settings.items():
        exp_setting_writer.write(f'{key} = {value}\n')

    train_dataset, val_dataset, test_dataset = get_split_dataset(
        root_dir, sub_sets, use_h5, pos_feature_coord_name, neg_erase_feature_coord_name,
        neg_background_feature_coord_name)

    datasets = (train_dataset, val_dataset, test_dataset)

    results, test_auc, val_auc, test_acc, val_acc = train(
        feature_extract_model_name,
        datasets, fold, device, B=B, bag_loss=bag_loss,
        bag_weight=bag_weight, drop_out=drop_out, early_stopping=early_stopping,
        inst_loss=inst_loss, log_data=log_data,
        lr=lr, max_epochs=max_epochs, model_size=model_size, model_type=model_type,
        n_classes=n_classes, no_inst_cluster=no_inst_cluster, opt=opt,
        reg=reg, results_dir=results_dir,
        subtyping=subtyping, testing=testing, weighted_sample=weighted_sample)

    exp_result_writer = open(os.path.join(results_dir, 'exp_result.txt'), 'w')
    result_show = ''

    exp_result_writer.write(f'val_auc = {val_auc:.4f}\n')
    exp_result_writer.write(f'val_acc = {val_acc:.4f}\n')
    exp_result_writer.write(f'test_auc = {test_auc:.4f}\n')
    exp_result_writer.write(f'test_acc = {test_acc:.4f}\n')

    for slide_id, slide_result in results.items():
        exp_result_writer.write(f'slide_id = {slide_result["slide_id"]}, '
                                f'prob = {str(slide_result["prob"].tolist()): <25}, '
                                f'label = {slide_result["label"]}\n')
        result_show += f'slide_id = {slide_result["slide_id"]}, ' \
                       f'prob = {str(slide_result["prob"].tolist()): <25}, ' \
                       f'label = {slide_result["label"]}\n'
    result_show += f'val_auc = {val_auc:.4f}\n'
    result_show += f'val_acc = {val_acc:.4f}\n'
    result_show += f'test_auc = {test_auc:.4f}\n'
    result_show += f'test_acc = {test_acc:.4f}\n'
    print(result_show)
