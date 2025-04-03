from __future__ import absolute_import, division, print_function, unicode_literals

from pysot.core.config import cfg
from pysot.tracker.siammask_tracker import SiamMaskTracker
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.tracker.siamrpnlt_tracker import SiamRPNLTTracker

TRACKS = {
    "SiamRPNTracker": SiamRPNTracker,
    "SiamMaskTracker": SiamMaskTracker,
    "SiamRPNLTTracker": SiamRPNLTTracker,
}


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
