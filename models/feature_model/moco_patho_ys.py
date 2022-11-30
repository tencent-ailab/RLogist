import torch
import numpy as np
import torch.nn as nn

import torchvision.models as models


class MoCo(nn.Module):
    def __init__(
            self, base_encoder, dim=128, K=65536, m=0.999, T=0.07,
            mlp=True, mgd=False, descriptors=None, p=3,
            normalize=False, attention=False, mpp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.normalize = normalize

        # create the encoders
        # num_classes is the output fc dimension
        if attention:
          self.encoder_q = base_encoder(num_classes=dim, attention=attention)
          self.encoder_k = base_encoder(num_classes=dim, attention=attention)
        else:
          self.encoder_q = base_encoder(num_classes=dim)
          self.encoder_k = base_encoder(num_classes=dim)
        # if mgd and descriptors is not None:
        #     dim_mlp = self.encoder_q.fc.weight.shape[1]
        #     self.encoder_q.avgpool = MGD(dim_mlp, dim_mlp, descriptors=descriptors, p=p)
        #     self.encoder_k.avgpool = MGD(dim_mlp, dim_mlp, descriptors=descriptors, p=p)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        if mpp:
            self.mpp_predictor_q = nn.Linear(dim, 2)
            self.mpp_predictor_k = nn.Linear(dim, 2)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, im_q):
        q = self.encoder_q(im_q)  # querys: NxC
        if self.normalize:
            q = nn.functional.normalize(q, dim=1)
        # print(q.size())
        return q


if __name__ == '__main__':
    pass




