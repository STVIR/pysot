# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging
import math
import sys


if hasattr(sys, 'frozen'):  # support for py2exe
    _srcfile = "logging%s__init__%s" % (os.sep, __file__[-4:])
elif __file__[-4:].lower() in ['.pyc', '.pyo']:
    _srcfile = __file__[:-4] + '.py'
else:
    _srcfile = __file__
_srcfile = os.path.normcase(_srcfile)

logs = set()


class Filter:
    def __init__(self, flag):
        self.flag = flag

    def filter(self, x):
        return self.flag


class Dummy:
    def __init__(self, *arg, **kwargs):
        pass

    def __getattr__(self, arg):
        def dummy(*args, **kwargs): pass
        return dummy


def get_format(logger, level):
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])

        if level == logging.INFO:
            logger.addFilter(Filter(rank == 0))
    else:
        rank = 0
    format_str = '[%(asctime)s-rk{}-%(filename)s#%(lineno)3d] %(message)s'.format(rank)
    formatter = logging.Formatter(format_str)
    return formatter


def get_format_custom(logger, level):
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        if level == logging.INFO:
            logger.addFilter(Filter(rank == 0))
    else:
        rank = 0
    format_str = '[%(asctime)s-rk{}-%(message)s'.format(rank)
    formatter = logging.Formatter(format_str)
    return formatter


def init_log(name, level=logging.INFO, format_func=get_format):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = format_func(logger, level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def add_file_handler(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(get_format(logger, level))
    logger.addHandler(fh)


init_log('global')


def print_speed(i, i_time, n):
    """print_speed(index, index_time, total_iteration)"""
    logger = logging.getLogger('global')
    average_time = i_time
    remaining_time = (n - i) * average_time
    remaining_day = math.floor(remaining_time / 86400)
    remaining_hour = math.floor(remaining_time / 3600 -
                                remaining_day * 24)
    remaining_min = math.floor(remaining_time / 60 -
                               remaining_day * 1440 -
                               remaining_hour * 60)
    logger.info('Progress: %d / %d [%d%%], Speed: %.3f s/iter, ETA %d:%02d:%02d (D:H:M)\n' %
                (i, n, i / n * 100,
                 average_time,
                 remaining_day, remaining_hour, remaining_min))


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
        self.logger = init_log('log_once', format_func=get_format_custom)

    def log(self, strings):
        fn, lineno, caller = find_caller()
        key = (fn, lineno, caller, strings)
        if key in self.logged:
            return
        self.logged.add(key)
        message = "{filename:s}<{caller}>#{lineno:3d}] {strings}".format(
                filename=fn, lineno=lineno, strings=strings, caller=caller)
        self.logger.info(message)


once_logger = LogOnce()


def log_once(strings):
    once_logger.log(strings)


def main():
    for i, lvl in enumerate([logging.DEBUG, logging.INFO,
                             logging.WARNING, logging.ERROR,
                             logging.CRITICAL]):
        log_name = str(lvl)
        init_log(log_name, lvl)
        logger = logging.getLogger(log_name)
        print('****cur lvl:{}'.format(lvl))
        logger.debug('debug')
        logger.info('info')
        logger.warning('warning')
        logger.error('error')
        logger.critical('critiacal')


if __name__ == '__main__':
    main()
    for i in range(10):
        log_once('xxx')
