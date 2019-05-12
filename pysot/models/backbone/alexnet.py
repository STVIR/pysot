from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import torch
import torch.nn as nn

class AlexNet(nn.Module):
    configs = [3, 96, 256, 384, 384, 256]
    def __init__(self, width_mult=1):
        configs = list(map(lambda x: 3 if x==3 else int(x*width_mult), AlexNet.configs))
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(configs[0], configs[1] , kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
        )
        self.feature_size = configs[5]

    def forward(self, x):
        x = self.features(x)
        return x

def alexnet(**kwargs):
    return AlexNet(**kwargs)
