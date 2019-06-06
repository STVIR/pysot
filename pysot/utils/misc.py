# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from colorama import Fore, Style


__all__ = ['commit', 'describe']


def _exec(cmd):
    f = os.popen(cmd, 'r', 1)
    return f.read().strip()


def _bold(s):
    return "\033[1m%s\033[0m" % s


def _color(s):
    return f'{Fore.RED}{s}{Style.RESET_ALL}'


def _describe(model, lines=None, spaces=0):
    head = " " * spaces
    for name, p in model.named_parameters():
        if '.' in name:
            continue
        if p.requires_grad:
            name = _color(name)
        line = "{head}- {name}".format(head=head, name=name)
        lines.append(line)

    for name, m in model.named_children():
        space_num = len(name) + spaces + 1
        if m.training:
            name = _color(name)
        line = "{head}.{name} ({type})".format(
                head=head,
                name=name,
                type=m.__class__.__name__)
        lines.append(line)
        _describe(m, lines, space_num)


def commit():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    cmd = "cd {}; git log | head -n1 | awk '{{print $2}}'".format(root)
    commit = _exec(cmd)
    cmd = "cd {}; git log --oneline | head -n1".format(root)
    commit_log = _exec(cmd)
    return "commit : {}\n  log  : {}".format(commit, commit_log)


def describe(net, name=None):
    num = 0
    lines = []
    if name is not None:
        lines.append(name)
        num = len(name)
    _describe(net, lines, num)
    return "\n".join(lines)
