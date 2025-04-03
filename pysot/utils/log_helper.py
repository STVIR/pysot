from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os
import sys
import colorlog


if hasattr(sys, "frozen"):  # support for py2exe
    _srcfile = "logging%s__init__%s" % (os.sep, __file__[-4:])
elif __file__[-4:].lower() in [".pyc", ".pyo"]:
    _srcfile = __file__[:-4] + ".py"
else:
    _srcfile = __file__
_srcfile = os.path.normcase(_srcfile)


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Remove all existing handlers
    logger.handlers.clear()

    handler = logging.StreamHandler()

    # Define color scheme for different log levels
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def print_speed(i, i_time, n):
    """print_speed(index, index_time, total_iteration)"""
    logger = logging.getLogger("global")
    average_time = i_time
    remaining_time = (n - i) * average_time
    remaining_day = math.floor(remaining_time / 86400)
    remaining_hour = math.floor(remaining_time / 3600 - remaining_day * 24)
    remaining_min = math.floor(
        remaining_time / 60 - remaining_day * 1440 - remaining_hour * 60
    )
    logger.info(
        "Progress: %d / %d [%d%%], Speed: %.3f s/iter, ETA %d:%02d:%02d (D:H:M)\n"
        % (
            i,
            n,
            i / n * 100,
            average_time,
            remaining_day,
            remaining_hour,
            remaining_min,
        )
    )


def find_caller():
    def current_frame():
        try:
            raise Exception
        except:
            return sys.exc_info()[2].tb_frame.f_back

    f = current_frame()
    if f is not None:
        f = f.f_back
    rv = "(unknown file)", 0, "(unknown function)"
    while hasattr(f, "f_code"):
        co = f.f_code
        filename = os.path.normcase(co.co_filename)
        rv = (co.co_filename, f.f_lineno, co.co_name)
        if filename == _srcfile:
            f = f.f_back
            continue
        break
    rv = list(rv)
    rv[0] = os.path.basename(rv[0])
    return rv


class LogOnce:
    def __init__(self):
        self.logged = set()
        self.logger = setup_logger("LogOnce")

    def log(self, strings):
        fn, lineno, caller = find_caller()
        key = (fn, lineno, caller, strings)
        if key in self.logged:
            return
        self.logged.add(key)
        message = "{filename:s}<{caller}>#{lineno:3d}] {strings}".format(
            filename=fn, lineno=lineno, strings=strings, caller=caller
        )
        self.logger.info(message)


once_logger = LogOnce()


def log_once(strings):
    once_logger.log(strings)
