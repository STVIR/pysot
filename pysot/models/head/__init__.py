from __future__ import absolute_import, division, print_function, unicode_literals

from pysot.models.head.mask import MaskCorr, Refine
from pysot.models.head.rpn import DepthwiseRPN, MultiRPN, UPChannelRPN

RPNS = {
    "UPChannelRPN": UPChannelRPN,
    "DepthwiseRPN": DepthwiseRPN,
    "MultiRPN": MultiRPN,
}

MASKS = {
    "MaskCorr": MaskCorr,
}

REFINE = {
    "Refine": Refine,
}


def get_rpn_head(name, **kwargs):
    return RPNS[name](**kwargs)


def get_mask_head(name, **kwargs):
    return MASKS[name](**kwargs)


def get_refine_head(name):
    return REFINE[name]()
