import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
# from models.model_clam_v3 import CLAM_SB, CLAM_MB
from models.model_clam import CLAM_SB, CLAM_MB
import pdb
import os
import pandas as pd
from utils.utils import get_simple_loader, get_split_loader, print_network, get_optim, calculate_error
from train_core import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


def initiate_model(ckpt_path, dropout=True, n_classes=2, model_size='small', model_type='clam_sb',
                   feature_extract_model_name=''):

    if model_type == 'clam_sb':
        model = CLAM_SB(dropout=dropout, n_classes=n_classes, size_arg=model_size,
                        feature_extract_model_name=feature_extract_model_name)
    elif model_type == 'clam_mb':
        model = CLAM_MB(dropout=dropout, n_classes=n_classes, size_arg=model_size,
                        feature_extract_model_name=feature_extract_model_name)
    else: # args.model_type == 'mil'
        if n_classes > 2:
            model = MIL_fc_mc(dropout=dropout, n_classes=n_classes, size_arg=model_size)
        else:
            model = MIL_fc(dropout=dropout, n_classes=n_classes, size_arg=model_size)

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''): ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    model.relocate()
    model.eval()
    return model


def eval(
        dataset, ckpt_path, dropout=True,
        n_classes=2, model_size='small',
        model_type='clam_sb', device=None, micro_average=False,
        feature_extract_model_name=''):
    model = initiate_model(
        ckpt_path, dropout=dropout, n_classes=n_classes,
        model_size=model_size, model_type=model_type,
        feature_extract_model_name=feature_extract_model_name)

    print('Init Loaders')
    loader = get_simple_loader(dataset, device=device)
    patient_results, test_error, auc, df, _ = summary(
        model, loader, n_classes=n_classes, device=device, micro_average=micro_average)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df


def summary(model, loader, n_classes=2, device=None, micro_average=False):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label, neg_feature, pos_feature) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids[batch_idx]
        with torch.no_grad():
            # logits, Y_prob, Y_hat, _, results_dict = model(data)
            logits, Y_prob, Y_hat, A_raw, A, A_sigmoid, \
                y_prob_instance, instance_dict, = model(data)

        acc_logger.log(Y_hat, label)

        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})

        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else:
        if n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
            for class_idx in range(n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'labels': all_labels, 'Y_hat': all_preds}
    for c in range(n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger
