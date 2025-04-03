from __future__ import absolute_import, division, print_function, unicode_literals

from pysot.models.backbone.alexnet import alexnet, alexnetlegacy
from pysot.models.backbone.mobile_v2 import mobilenetv2
from pysot.models.backbone.resnet_atrous import resnet18, resnet34, resnet50

BACKBONES = {
    "alexnetlegacy": alexnetlegacy,
    "mobilenetv2": mobilenetv2,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "alexnet": alexnet,
}


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
