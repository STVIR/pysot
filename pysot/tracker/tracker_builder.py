# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import importlib

from pysot.core.config import cfg

def build_tracker(model):
    module_name, cls_name = cfg.TRACK.TYPE.rsplit('.', 1)
    module = importlib.import_module(module_name)
    tracker = getattr(module, cls_name)(model)
    return tracker
