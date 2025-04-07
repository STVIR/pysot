"""
Helper functions for the SiamMask model.
https://drive.google.com/drive/folders/1YbPUQVTYw_slAvk_DchvRY-7B6rnSXP9
"""

from collections import OrderedDict

import torch

model = torch.load("SiamMask_VOT_LD.pth", map_location=lambda storage, loc: storage)

new_model = OrderedDict()

for k, v in model.items():
    if k.startswith("features.features"):
        k = k.replace("features.features", "backbone")
    elif k.startswith("features"):
        k = k.replace("features", "neck")
    elif k.startswith("rpn_model"):
        k = k.replace("rpn_model", "rpn_head")
    elif k.startswith("mask_model"):
        k = k.replace("mask_model.mask", "mask_head")
    elif k.startswith("refine_model"):
        k = k.replace("refine_model", "refine_head")
    new_model[k] = v

torch.save(new_model, "model.pth")
