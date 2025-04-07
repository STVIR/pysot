import gdown
from pytubefix import YouTube
from pytubefix.cli import on_progress
from pysot.utils.log_helper import setup_logger

logger = setup_logger("pysot")


def download_youtube(url: str, dest: str) -> bool:
    try:
        yt = YouTube(url, on_progress_callback=on_progress)
        ys = yt.streams.get_highest_resolution()
        ys.download(output_path=dest)
        logger.info(
            f"YouTube video downloaded successfully: {dest / ys.default_filename}"
        )
        return True, dest / ys.default_filename
    except Exception as e:
        logger.error(f"Error downloading YouTube video: {e}")
        return False, dest / ys.default_filename


def download_file(url: str, dest: str, fuzzy: bool = True, quiet: bool = False) -> bool:
    try:
        gdown.download(url, dest, fuzzy=fuzzy, quiet=quiet)
        logger.info(f"File downloaded successfully: {dest}")
        return True, dest
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return False, str(e)
