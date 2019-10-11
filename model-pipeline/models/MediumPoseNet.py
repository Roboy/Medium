import torch
import torch.nn as nn
import torch.nn.functional as F
# from config import config as cfg
from torch.autograd import Variable


class RF_Pose_RPN(nn.Module):
    def __init__(self, in_channels=20, out_channels=13):
        super(RF_Pose_RPN, self).__init__()

        self.in_channels = in_channels

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, 1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, 1),
            nn.BatchNorm3d(32),
            nn.Dropout3d(p=0.2, inplace=False),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.Dropout3d(p=0.2, inplace=False),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, 3, (1, 1, 2), (0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 5, (2, 1), 1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(p=0.2, inplace=False),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256,
                       64,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,
                       out_channels,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       bias=True),
            nn.Dropout2d(p=0.2, inplace=False),
            nn.ReLU()
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.reshape(x, (-1, 256, 21, 56))
        x = self.dconv1(x)
        out = self.conv4(x)

        return out


class RF_Pose2D_funky(nn.Module):
    def __init__(self, in_channels=20, out_channels=13):
        super(RF_Pose2D_funky, self).__init__()

        self.in_channels = in_channels

        # expects two inputs 116 x 7 and 116 x 25

        self.conv_hor = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=(1, 2),
                      padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.2, inplace=False),
            nn.ReLU()
        )

        self.conv_ver = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=(1, 2),
                      padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.2, inplace=False),
            nn.ReLU(),
            nn.Upsample(size=(24, 58))
        )

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=13,
                               kernel_size=3,
                               stride=(2, 1),
                               padding=(3, 1),
                               ),
            nn.BatchNorm2d(13),
            nn.Dropout2d(p=0.2, inplace=False),
            nn.ReLU()
        )

        self.layer1 = self.make_layer(ResidualBlock, 13, 13, 2)
        self.layer2 = self.make_layer(ResidualBlock, 13, 13, 2)

    def make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x_1, x_2):
        x_1 = self.conv_hor(x_1)
        x_2 = self.conv_ver(x_2)

        x = torch.cat((x_1, x_2), 1)

        x = self.dconv1(x)

        x = self.layer1(x)
        x = self.layer2(x)

        return x


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.softmax(out)
        return out
