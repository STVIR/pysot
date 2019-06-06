# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class Meter(object):
    def __init__(self, name, val, avg):
        self.name = name
        self.val = val
        self.avg = avg

    def __repr__(self):
        return "{name}: {val:.6f} ({avg:.6f})".format(
            name=self.name, val=self.val, avg=self.avg
        )

    def __format__(self, *tuples, **kwargs):
        return self.__repr__()


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, num=100):
        self.num = num
        self.reset()

    def reset(self):
        self.val = {}
        self.sum = {}
        self.count = {}
        self.history = {}

    def update(self, batch=1, **kwargs):
        val = {}
        for k in kwargs:
            val[k] = kwargs[k] / float(batch)
        self.val.update(val)
        for k in kwargs:
            if k not in self.sum:
                self.sum[k] = 0
                self.count[k] = 0
                self.history[k] = []
            self.sum[k] += kwargs[k]
            self.count[k] += batch
            for _ in range(batch):
                self.history[k].append(val[k])

            if self.num <= 0:
                # < 0, average all
                self.history[k] = []

                # == 0: no average
                if self.num == 0:
                    self.sum[k] = self.val[k]
                    self.count[k] = 1

            elif len(self.history[k]) > self.num:
                pop_num = len(self.history[k]) - self.num
                for _ in range(pop_num):
                    self.sum[k] -= self.history[k][0]
                    del self.history[k][0]
                    self.count[k] -= 1

    def __repr__(self):
        s = ''
        for k in self.sum:
            s += self.format_str(k)
        return s

    def format_str(self, attr):
        return "{name}: {val:.6f} ({avg:.6f}) ".format(
                    name=attr,
                    val=float(self.val[attr]),
                    avg=float(self.sum[attr]) / self.count[attr])

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return super(AverageMeter, self).__getattr__(attr)
        if attr not in self.sum:
            print("invalid key '{}'".format(attr))
            return Meter(attr, 0, 0)
        return Meter(attr, self.val[attr], self.avg(attr))

    def avg(self, attr):
        return float(self.sum[attr]) / self.count[attr]


if __name__ == '__main__':
    avg1 = AverageMeter(10)
    avg2 = AverageMeter(0)
    avg3 = AverageMeter(-1)

    for i in range(20):
        avg1.update(s=i)
        avg2.update(s=i)
        avg3.update(s=i)

        print('iter {}'.format(i))
        print(avg1.s)
        print(avg2.s)
        print(avg3.s)
