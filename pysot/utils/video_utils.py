import os
from glob import glob

import cv2


def get_frames(video_name: str = ""):
    match video_name:
        case "":
            print("math 1")
            cap = cv2.VideoCapture(0)
            # warmup
            for _ in range(5):
                cap.read()
            while True:
                ret, frame = cap.read()
                if ret:
                    yield frame
                else:
                    break
        case str():
            print("math 2")
            print(type(video_name))
            if video_name.endswith(".avi") or video_name.endswith(".mp4"):
                print("match 2.1", video_name)
                cap = cv2.VideoCapture(video_name)
                while True:
                    ret, frame = cap.read()
                    print(frame)
                    if ret:
                        print(frame.shape)
                        yield frame
                    else:
                        break
            else:
                print("math 2.2")
                images = glob(os.path.join(video_name, "*.jp*"))
                images = sorted(
                    images, key=lambda x: int(x.split("/")[-1].split(".")[0])
                )
                for img in images:
                    frame = cv2.imread(img)
                    yield frame
        case _:
            raise ValueError(f"Invalid video name: {video_name}")
