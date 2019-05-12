# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        
        # build backbone
        module_name, cls_name = cfg.BACKBONE.TYPE.rsplit('.', 1)
        module = importlib.import_module(module_name)
        if cls_name.startswith('alexnet'):
            self.backbone = getattr(module, cls_name)(width_mult=cfg.BACKBONE.WIDTH_MULT)
        elif cls_name.startswith('mobile'):
            self.backbone = getattr(module, cls_name)(width_mult=cfg.BACKBONE.WIDTH_MULT,
                    used_layers=cfg.BACKBONE.LAYERS)
        else: 
            self.backbone = getattr(module, cls_name)(used_layers=cfg.BACKBONE.LAYERS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            module_name, cls_name = cfg.ADJUST.TYPE.rsplit('.', 1)
            module = importlib.import_module(module_name)
            module = getattr(module, cls_name)
            self.neck = module(cfg.BACKBONE.CHANNELS, cfg.ADJUST.ADJUST_CHANNEL)
            
        # build rpn head
        module_name, cls_name = cfg.RPN.TYPE.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        if cfg.ADJUST.ADJUST:
            channels = cfg.ADJUST.ADJUST_CHANNEL
        else:
            channels = cfg.BACKBONE.CHANNELS
        if len(channels) == 1:
            channels = channels[0]

        if cfg.RPN.WEIGHTED:
            self.rpn_head = cls(cfg.ANCHOR.ANCHOR_NUM, channels, True)
        else:
            self.rpn_head = cls(cfg.ANCHOR.ANCHOR_NUM, channels)

        # build mask head
        if cfg.MASK.MASK:
            module_name, cls_name = cfg.MASK.MASK_TYPE.rsplit('.', 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, cls_name)
            self.mask_head = cls(cfg.ADJUST.ADJUST_CHANNEL[0],
                                 cfg.ADJUST.ADJUST_CHANNEL[0],
                                 cfg.MASK.OUT_CHANNELS)
            if cfg.MASK.REFINE:
                module_name, cls_name = cfg.MASK.REFINE_TYPE.rsplit('.', 1)
                module = importlib.import_module(module_name)
                cls = getattr(module, cls_name)
                self.refine_head = cls()

    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }
    
    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)
