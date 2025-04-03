import os
import cv2
from glob import glob
from pytubefix import YouTube
from pytubefix.cli import on_progress


def download_youtube_video(url: str, dest: str):
    yt = YouTube(url, on_progress_callback=on_progress)
    ys = yt.streams.get_highest_resolution()
    ys.download(output_path=dest)


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith("avi") or video_name.endswith("mp4"):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, "*.jp*"))
        images = sorted(images, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame
