from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn


def conv_bn(inp, oup, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        self.use_res_connect = self.stride == 1 and inp == oup

        padding = 2 - stride
        if dilation > 1:
            padding = dilation

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3,
                      stride, padding, dilation=dilation,
                      groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Sequential):
    def __init__(self, width_mult=1.0, used_layers=[3, 5, 7]):
        super(MobileNetV2, self).__init__()

        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1, 1],
            [6, 24, 2, 2, 1],
            [6, 32, 3, 2, 1],
            [6, 64, 4, 2, 1],
            [6, 96, 3, 1, 1],
            [6, 160, 3, 2, 1],
            [6, 320, 1, 1, 1],
        ]
        # 0,2,3,4,6

        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1, 1],
            [6, 24, 2, 2, 1],
            [6, 32, 3, 2, 1],
            [6, 64, 4, 1, 2],
            [6, 96, 3, 1, 2],
            [6, 160, 3, 1, 4],
            [6, 320, 1, 1, 4],
        ]

        self.channels = [24, 32, 96, 320]
        self.channels = [int(c * width_mult) for c in self.channels]

        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) \
            if width_mult > 1.0 else 1280

        self.add_module('layer0', conv_bn(3, input_channel, 2, 0))

        last_dilation = 1

        self.used_layers = used_layers

        for idx, (t, c, n, s, d) in \
                enumerate(self.interverted_residual_setting, start=1):
            output_channel = int(c * width_mult)

            layers = []

            for i in range(n):
                if i == 0:
                    if d == last_dilation:
                        dd = d
                    else:
                        dd = max(d // 2, 1)
                    layers.append(InvertedResidual(input_channel,
                                                   output_channel, s, t, dd))
                else:
                    layers.append(InvertedResidual(input_channel,
                                                   output_channel, 1, t, d))
                input_channel = output_channel

            last_dilation = d

            self.add_module('layer%d' % (idx), nn.Sequential(*layers))

    def forward(self, x):
        outputs = []
        for idx in range(8):
            name = "layer%d" % idx
            x = getattr(self, name)(x)
            outputs.append(x)
        p0, p1, p2, p3, p4 = [outputs[i] for i in [1, 2, 3, 5, 7]]
        out = [outputs[i] for i in self.used_layers]
        if len(out) == 1:
            return out[0]
        return out


def mobilenetv2(**kwargs):
    model = MobileNetV2(**kwargs)
    return model


if __name__ == '__main__':
    net = mobilenetv2()

    print(net)

    from torch.autograd import Variable
    tensor = Variable(torch.Tensor(1, 3, 255, 255)).cuda()

    net = net.cuda()

    out = net(tensor)

    for i, p in enumerate(out):
        print(i, p.size())
