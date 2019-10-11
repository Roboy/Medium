import torch
import torch.nn as nn

class RF_Pose3D_130100(nn.Module):
    def __init__(self, in_channels=20, out_channels = 13):
        super(RF_Pose3D_130100, self).__init__()

        self.in_channels = in_channels

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, 1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, 1, (0,0,0)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, 1, (1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, 1, (1,1,1)),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, 3, (1, 1, 2), (0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, (2, 1), 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.upsample = nn.Sequential(
            nn.Upsample(size=(90, 120)),
            nn.ReLU()
        )

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 6, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.output = nn.Sequential(
            nn.Conv2d(32, 13, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.reshape(x, (-1, 256, 21, 56))

        x = self.dconv1(x)
        x = self.upsample(x)
        x = self.dconv2(x)
        x = self.output(x)

        return x


class RF_Pose3D_130100_Dropout(nn.Module):
    def __init__(self, in_channels=20, out_channels = 13):
        super(RF_Pose3D_130100_Dropout, self).__init__()

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
            nn.Conv3d(128, 256, 3, (1, 1, 2), (0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, (2, 1), 1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.2, inplace=False),
            nn.ReLU()
        )

        self.upsample = nn.Sequential(
            nn.Upsample(size=(94, 124)),
            nn.ReLU()
        )

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 6, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.2, inplace=False),
            nn.ReLU()
        )

        self.output = nn.Sequential(
            nn.Conv2d(32, 13, 5, 1),
            nn.ReLU()
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.reshape(x, (-1, 256, 21, 56))

        x = self.dconv1(x)
        x = self.upsample(x)
        x = self.dconv2(x)
        x = self.output(x)

        return x
