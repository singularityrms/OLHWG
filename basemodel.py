import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
print(device)
###  时间编码  #####
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


### noise schedule ###
def cosine_schedule(initial_scale, final_scale, total_steps):
    scales = []
    for step in range(total_steps):
        proportion = step / total_steps
        decay = 0.5 * (1 + torch.cos(torch.tensor(np.pi * proportion)))
        scale = initial_scale * (1 - decay) + final_scale * decay
        scales.append(scale)
    return torch.tensor(scales)


### 网络模型   #########
class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1):
        super().__init__()
        self.pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, dilation=dilation, stride=stride)
        #self.dp = nn.Dropout(p=0.02)
                              
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = F.pad(x, [0, self.pad])
        x = self.conv(x)
        return x


class TransConv1d(nn.Module):
    #还有一点瑕疵  关于pad长度
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1):
        super().__init__()
        self.pad = int(((2*dilation - 1)*stride - dilation + 1)/2)
        self.transconv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=kernel_size, stride=stride, padding=self.pad)
        #self.dp = nn.Dropout(p=0.02)

        nn.init.xavier_uniform_(self.transconv.weight)
        
    def forward(self, x):
        x = self.transconv(x)
        return x


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size=3):
        # residual in == out stride = 1
        super().__init__()
        self.layers = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation),
            nn.ELU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if stride != 1:
            kernel = stride * 2
        else:
            kernel = 1

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=5),
            nn.ELU(),
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride)
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if stride != 1:
            kernel = stride * 2
        else:
            kernel = 1
        self.layers = nn.Sequential(
            TransConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=5)
        )

    def forward(self, x):
        return self.layers(x)