# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "siamrpn_r50_l234_dwxcorr"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'pysot.models.backbone.resnet_atrous.resnet50'  


# Layers to use for alexnet we only use last layer (-1)
__C.BACKBONE.LAYERS = (2, 3, 4)

# Channels
__C.BACKBONE.CHANNELS = (128*4, 256*4, 512*4)

# Width mult
__C.BACKBONE.WIDTH_MULT = 1.4


# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

# Adjust layer type
__C.ADJUST.TYPE = "pysot.models.neck.neck.AdjustAllLayer"

# Adjust number of channels
__C.ADJUST.ADJUST_CHANNEL = (256, 256, 256)

# ------------------------------------------------------------------------ #
# RPN options
# ------------------------------------------------------------------------ #
__C.RPN = CN()

# RPN type
__C.RPN.TYPE = 'pysot.models.head.rpn.MultiRPN'

# Whether to weighted aggregate
__C.RPN.WEIGHTED = False


# ------------------------------------------------------------------------ #
# mask options
# ------------------------------------------------------------------------ #
__C.MASK = CN()

# Whether to use mask generate segmentation
__C.MASK.MASK = False

# Mask type
__C.MASK.MASK_TYPE = "pysot.models.head.mask.MaskCorr"

# Mask out channels
__C.MASK.OUT_CHANNELS = 63 ** 2

# Mask refine
__C.MASK.REFINE = False

# Refine type
__C.MASK.REFINE_TYPE = "pysot.models.head.mask.Refine"

# ------------------------------------------------------------------------ #
# Anchor options
# ------------------------------------------------------------------------ #
__C.ANCHOR = CN()

# Anchor stride
__C.ANCHOR.STRIDE = 8

# Anchor ratios
__C.ANCHOR.RATIOS = [0.33, 0.5, 1, 2, 3]

# Anchor scales
__C.ANCHOR.SCALES = [8]

# Anchor number
__C.ANCHOR.ANCHOR_NUM = len(__C.ANCHOR.RATIOS) * len(__C.ANCHOR.SCALES)


# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'pysot.tracker.siamrpn_tracker.SiamRPNTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Base size
__C.TRACK.BASE_SIZE = 8

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

# Long term lost search size
__C.TRACK.LOST_INSTANCE_SIZE = 831

# Long term confidence low
__C.TRACK.CONFIDENCE_LOW = 0.85

# Long term confidence high
__C.TRACK.CONFIDENCE_HIGH = 0.998

# Mask threshold
__C.TRACK.MASK_THERSHOLD = 0.30

# Mask output size
__C.TRACK.MASK_OUTPUT_SIZE = 127
