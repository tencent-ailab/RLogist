# standard library
from collections import OrderedDict
import os
import sys
# 3rd part packages
import torch
import torch.nn as nn
import torchvision.models as models
# local source
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(file_dir, "../../"))

from models.feature_model.resnet_imagenet import resnet50_baseline
from models.feature_model.moco_patho_ys import MoCo as MoCo_patho
from models.feature_model.moco_v2_imagenet import MoCo as MoCo_imagenet


def load_resnet_imagenet_model(device, dense=False):
    print('loading resnet_imagenet pretrained checkpoint')

    model_path = os.path.join(
        file_dir, "./resnet50-19c8e357.pth")
    model = resnet50_baseline(pretrained=True, model_path=model_path, dense=dense)
    model = model.to(device, non_blocking=True)

    # print_network(model)

    model.eval()
    return model


def load_moco_patho_model(device):
    def create_moco_ys():
        net = MoCo_patho(models.__dict__['resnet50'], dim=128)

        model_path = os.path.join(
            file_dir, "./moco_patho_checkpoint_0039.pth.tar")

        pretext_model = torch.load(model_path, map_location='cpu')['state_dict']
        td = OrderedDict()
        for key, value in pretext_model.items():
            print(key)
            print(value.shape)
            k = key[7:]
            td[k] = value
        msg = net.load_state_dict(td)

        net.encoder_q.fc = nn.Identity()
        net.encoder_q.instDis = nn.Identity()
        net.encoder_q.groupDis = nn.Identity()
        print(msg)

        net.avgpool = nn.AvgPool2d(14, stride=1, padding=7)

        return net
    print('loading moco_patho pretrained checkpoint')

    model = create_moco_ys()

    model = model.to(device, non_blocking=True)

    # print_network(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()
    return model


def load_moco_v2_imagenet_model(device):
    def create_moco_v2():
        model = MoCo_imagenet(models.__dict__['resnet50'], dim=128)

        model_path = os.path.join(
            file_dir, "./moco_v2_800ep_pretrain.pth.tar")

        print("=> loading checkpoint '{}'".format(model_path))

        pretext_model = torch.load(model_path, map_location='cpu')['state_dict']
        td = OrderedDict()
        for key, value in pretext_model.items():
            print(key)
            print(value.shape)
            k = key[7:]
            td[k] = value
        msg = model.load_state_dict(td, strict=False)

        model.encoder_q.fc = nn.Identity()
        model.encoder_q.instDis = nn.Identity()
        model.encoder_q.groupDis = nn.Identity()

        # msg = model.load_state_dict(checkpoint['state_dict'])
        print(msg)
        return model

    print('loading moco_v2 imagenet pretrained checkpoint')

    model = create_moco_v2()

    model = model.to(device, non_blocking=True)

    # print_network(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()
    return model


def main():
    print('main')
    device_id = 0
    device = torch.device(f'cuda:{device_id}')
    # feature_model = load_moco_patho_model(device)
    # feature_model = load_resnet_imagenet_model(device)
    feature_model = load_moco_v2_imagenet_model(device)

    rand_input = torch.rand(1, 3, 256, 256).cuda()
    print(rand_input.shape)

    with torch.no_grad():
        feature = feature_model(rand_input.to(device, non_blocking=True))

    print(feature.shape)


if __name__ == '__main__':
    main()
