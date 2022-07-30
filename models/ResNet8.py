
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import math

# from torchsummary import summary

'''
taken from https://github.com/pytorch/pytorch/issues/67551#issuecomment-954972351
pytorch does not support padding='same' if Conv2d has stride other than 1
therefore use helperfunction to calculate padding
'''
class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ResNet8(nn.Module):
    def __init__(self, input_dim, output_dim, f=0.25):
        super(ResNet8, self).__init__()
        self.f = f
        # kaiming he norm used as default by pytorch

        # first residual block
        self.x1 = nn.Sequential(
            Conv2dSame(in_channels=input_dim, out_channels=int(32*f), kernel_size=(5, 5), stride=(2, 2)),
            nn.MaxPool2d((3, 3), stride=(2, 2))
        )
        self.x2 = nn.Sequential(
            nn.ReLU(),
            Conv2dSame(in_channels=int(32*f), out_channels=int(32*f), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            Conv2dSame(in_channels=int(32*f), out_channels=int(32*f), kernel_size=(3, 3), stride=(1, 1))
        )
        self.x1_ = Conv2dSame(in_channels=int(32*f), out_channels=int(32*f), kernel_size=(1, 1), stride=(2, 2))

        # second residual block
        self.x4 = nn.Sequential(
            nn.ReLU(),
            Conv2dSame(in_channels=int(32*f), out_channels=int(64*f), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            Conv2dSame(in_channels=int(64*f), out_channels=int(64*f), kernel_size=(3, 3), stride=(1, 1))
        )

        self.x3 = Conv2dSame(in_channels=int(32*f), out_channels=int(64*f), kernel_size=(1, 1), stride=(2, 2))


        # third residual block

        self.x6 = nn.Sequential(
            nn.ReLU(),
            Conv2dSame(in_channels=int(64*f), out_channels=int(128*f), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            Conv2dSame(in_channels=int(128*f), out_channels=int(128*f), kernel_size=(3, 3), stride=(1, 1))
        )

        self.x5 = Conv2dSame(in_channels=int(64*f), out_channels=int(128*f), kernel_size=(1, 1), stride=(2, 2))

        self.x7 = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.dense = nn.Linear(in_features=int(2240*4*f), out_features=256)
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=output_dim)
        )


    def forward(self, t):
        # first residual block
        t1 = self.x1(t)
        t2 = self.x2(t1)
        t1_ = self.x1_(t1)
        t3 = t2 + t1_

        # second residual block
        t4 = self.x4(t3)
        t3_ = self.x3(t3)
        t5 = t3_ + t4

        # third resudual block
        t6 = self.x6(t5)
        t5_ = self.x5(t5)
        t7 = t5_ + t6

        t8 = self.x7(t7)
        td = self.dense(t8)
        to = self.out(td)

        return to

if __name__ == '__main__':
    rn8 = ResNet8(3, 4)
    # summary(rn8, input_size=(3, 200,300), device='cpu')