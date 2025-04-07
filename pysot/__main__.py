from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

import cv2
import numpy as np
import torch
from pathlib import Path
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.cfg.mapping import ModelConfigBuilder
from pysot.utils.download_utils import download_file, download_youtube
from pysot.utils.log_helper import setup_logger
from pysot.utils.video_utils import get_frames

logger = setup_logger("pysot")

torch.set_num_threads(1)
ROOT_DIR = Path(__file__).resolve().parent.parent


parser = argparse.ArgumentParser(description="Tracking demo")
parser.add_argument("--model_name", type=str, help="model name")
parser.add_argument("--video_name", default="", type=str, help="videos or image files")
args = parser.parse_args()


def main():
    model_metadata = ModelConfigBuilder(args.model_name)
    model_metadata = model_metadata.get_metadata()

    model_weight_url = model_metadata.get("weights")
    model_weight_path = model_metadata.get("dir") / "model.pth"
    try:
        if not model_weight_path.exists():
            download_file(url=model_weight_url, dest=str(model_weight_path))
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")

    model_config = model_metadata.get("config")

    # load config
    cfg.merge_from_file(model_config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device("cuda" if cfg.CUDA else "cpu")

    # create model
    model = ModelBuilder()

    # load model)
    model.load_state_dict(
        torch.load(model_weight_path, map_location=device),
    )
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True

    video_name = args.video_name
    video_placeholder = ROOT_DIR / cfg.DATASET.TEMP
    video_placeholder.mkdir(parents=True, exist_ok=True)
    if video_name:
        if video_name.startswith("http") and "youtube" in video_name:
            try:
                status, video_name = download_youtube(video_name, video_placeholder)
            except Exception as e:
                logger.error(f"Failed to download YouTube video: {e}")
        else:
            video_path = Path(video_name)
            if not video_path.is_absolute():
                video_path = ROOT_DIR / video_path
                video_name = str(video_path)
    else:
        video_name = "webcam"

    video_name = str(ROOT_DIR / cfg.DATASET.TEMP / video_name)

    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for frame in get_frames(video_name):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame)
            if "polygon" in outputs:
                polygon = np.array(outputs["polygon"]).astype(np.int32)
                cv2.polylines(
                    frame, [polygon.reshape((-1, 1, 2))], True, (0, 255, 0), 3
                )
                mask = (outputs["mask"] > cfg.TRACK.MASK_THERSHOLD) * 255
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs["bbox"]))
                cv2.rectangle(
                    frame,
                    (bbox[0], bbox[1]),
                    (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    (0, 255, 0),
                    3,
                )
            cv2.imshow(video_name, frame)
            cv2.waitKey(40)


if __name__ == "__main__":
    main()
