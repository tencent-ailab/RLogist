"""
module docstring.
"""

# standard library
import os
import sys
# 3rd part packages
import torch
# local source
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(file_dir, "../"))


def load_slide_model(
        ckpt_path='', drop_out=True, model_size='small',
        model_type='clam_sb', n_classes=2, feature_extract_model_name='',
        sigmoid_bias=2.0):
    from models.model_clam import CLAM_MB, CLAM_SB

    def initiate_model(
            ckpt_path, dropout=True, n_classes=2,
            model_size='small', model_type='clam_sb', sigmoid_bias=2.0):

        if model_type == 'clam_sb':
            model = CLAM_SB(dropout=dropout, n_classes=n_classes, size_arg=model_size,
                            feature_extract_model_name=feature_extract_model_name,
                            sigmoid_bias=sigmoid_bias)
        elif model_type == 'clam_mb':
            model = CLAM_MB(dropout=dropout, n_classes=n_classes, size_arg=model_size,
                            feature_extract_model_name=feature_extract_model_name,
                            sigmoid_bias=sigmoid_bias)

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

    slide_model = initiate_model(
        ckpt_path, dropout=drop_out, n_classes=n_classes,
        model_size=model_size, model_type=model_type,
        sigmoid_bias=sigmoid_bias)

    return slide_model


def load_dense_slide_model(
        ckpt_path='', drop_out=True, model_size='small',
        model_type='clam_sb', n_classes=2, feature_extract_model_name='',
        sigmoid_bias=2.0):
    # from models.model_clam_dense import CLAM_MB, CLAM_SB
    # from models.model_clam_v3_dense import CLAM_MB, CLAM_SB
    from models.model_clam_v4_dense import CLAM_MB, CLAM_SB

    def initiate_model(
            ckpt_path, dropout=True, n_classes=2,
            model_size='small', model_type='clam_sb', sigmoid_bias=2.0):

        if model_type == 'clam_sb':
            model = CLAM_SB(dropout=dropout, n_classes=n_classes, size_arg=model_size,
                            feature_extract_model_name=feature_extract_model_name,
                            sigmoid_bias=sigmoid_bias)
        elif model_type == 'clam_mb':
            model = CLAM_MB(dropout=dropout, n_classes=n_classes, size_arg=model_size,
                            feature_extract_model_name=feature_extract_model_name,
                            sigmoid_bias=sigmoid_bias)

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

    slide_model = initiate_model(
        ckpt_path, dropout=drop_out, n_classes=n_classes,
        model_size=model_size, model_type=model_type,
        sigmoid_bias=sigmoid_bias)

    return slide_model


def main():
    print('main')


if __name__ == '__main__':
    main()
