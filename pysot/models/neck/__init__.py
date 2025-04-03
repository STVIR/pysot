from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.models.neck.neck import AdjustAllLayer, AdjustLayer

NECKS = {"AdjustLayer": AdjustLayer, "AdjustAllLayer": AdjustAllLayer}


def get_neck(name, **kwargs):
    return NECKS[name](**kwargs)
