import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F


# NFV 5x5
class CNN2D(nn.Module):
    def __init__(self, qmax):
        # register dx and dt at parent class level
        super().__init__()
        # 5 input channels (past time steps)
        # 16 output channels
        # 2 kernel size
        self.qmax = qmax
        self.conv1 = nn.Conv1d(5, 16, kernel_size=6, stride=1)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(16, 16, kernel_size=1, stride=1)
        self.conv4 = nn.Conv1d(16, 16, kernel_size=1, stride=1)
        self.conv5 = nn.Conv1d(16, 16, kernel_size=1, stride=1)
        self.conv6 = nn.Conv1d(16, 1, kernel_size=1, stride=1)

    def forward(self, rho, *args):  # rho: BTX (with T=C=5 and X flipped)
        x = F.tanh(self.conv1(rho))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = F.tanh(self.conv4(x))
        x = F.tanh(self.conv5(x))
        flow = self.conv6(x)
        return torch.clamp(flow.squeeze(1), 0, self.qmax)

    def load_checkpoint(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device))
