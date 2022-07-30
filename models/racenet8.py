from cmath import tanh
import torch
from .ResNet8 import ResNet8


class RaceNet8(ResNet8):
    def __init__(self, input_dim, output_dim, f=0.25, num_recurrent_layer=1):
        super().__init__(input_dim, output_dim, f)
        self.recurrent = torch.nn.RNN(
            input_size = int(2240*4*f),
            hidden_size = 256,
            num_layers = num_recurrent_layer,
            nonlinearity = 'tanh',
            dropout = 0.5
        )
        
    def forward(self, t, h_last):
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
        t8 = torch.unsqueeze(t8, dim=1)
        td , h_now = self.recurrent(t8, h_last)
        to = self.out(td)
        return to, h_now