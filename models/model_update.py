import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""


# model single branch
class CLAM_SB(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, feature_extract_model_name='',
                 sigmoid_bias=2.0):
        super(CLAM_SB, self).__init__()

        if feature_extract_model_name == 'resnet50_imagenet_pretrain':
            self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        elif feature_extract_model_name == 'resnet50_pathology_moco_pretrain':
            self.size_dict = {"small": [2048, 512, 256], "big": [2048, 512, 384]}
        elif feature_extract_model_name == 'resnet50_imagenet_moco_pretrain':
            self.size_dict = {"small": [2048, 512, 256], "big": [2048, 512, 384]}
        else:
            raise Exception('feature extract model not define')

        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        # self.classifiers = nn.Linear(1024, n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        print('k_sample:', k_sample)
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        self.patch_loss_atten_fn = nn.MSELoss(reduction='mean')
        self.patch_loss_prob_fn = nn.CrossEntropyLoss()

        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device, dtype=torch.long).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device, dtype=torch.long).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def select_feature(self, feature):
        max_index = feature.shape[0]
        select_num = 5000
        if max_index > select_num:
            select_index = np.random.choice(
                range(feature.shape[0]), size=(select_num, 1), replace=False)

            select_feature = np.take_along_axis(feature, select_index, axis=0)
        else:
            select_feature = feature

        return select_feature

    def forward(
            self, h, label=None,
            neg_feature=None, pos_feature=None,
            instance_eval=False, return_features=False, attention_only=False, val_flag=False):
        device = h.device

        # print(f'h.shape: {h.shape}')

        atten_patch_raw, feature_patch = self.attention_net(h)
        # print(f'atten_patch_raw.shape: {atten_patch_raw.shape}')
        atten_patch_raw = torch.transpose(atten_patch_raw, 1, 0)  # KxN
        atten_patch_sigmoid = F.sigmoid(atten_patch_raw)
        # print(f'atten_patch_sigmoid.shape: {atten_patch_sigmoid.shape}')
        # print(f'feature_patch.shape: {feature_patch.shape}')

        feature_patch_logits = self.classifiers(feature_patch)
        # print(f'feature_patch_logits.shape: {feature_patch_logits.shape}')

        if len(feature_patch_logits.shape) == 4:
            feature_patch_prob = F.softmax(feature_patch_logits, dim=3)
        elif len(feature_patch_logits.shape) == 2:
            feature_patch_prob = F.softmax(feature_patch_logits, dim=1)
        else:
            raise Exception('feature_patch_logits.shape is not correct')

        atten_raw = atten_patch_raw
        atten_sigmoid = atten_patch_sigmoid
        instance_logits = feature_patch_logits
        instance_probs = feature_patch_prob

        return atten_raw, atten_sigmoid, instance_logits, instance_probs


# model multi branch
class CLAM_MB(CLAM_SB):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, feature_extract_model_name='',
                 sigmoid_bias=2.0):
        nn.Module.__init__(self)

        if feature_extract_model_name == 'resnet50_imagenet_pretrain':
            self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        elif feature_extract_model_name == 'resnet50_pathology_moco_pretrain':
            self.size_dict = {"small": [2048, 512, 256], "big": [2048, 512, 384]}
        elif feature_extract_model_name == 'resnet50_imagenet_moco_pretrain':
            self.size_dict = {"small": [2048, 512, 256], "big": [2048, 512, 384]}
        else:
            raise Exception('feature extract model not define')

        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)]  # use an indepdent linear layer to
        # predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict
