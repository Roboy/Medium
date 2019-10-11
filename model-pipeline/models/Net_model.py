import torch
import torch.nn as nn


class RF_Pose3D_130100(nn.Module):
    def __init__(self, in_channels=20, out_channels=13):
        super(RF_Pose3D_130100, self).__init__()

        self.in_channels = in_channels

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, 1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, 1, (0, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, 1, (1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, 1, (1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, 3, (1, 1, 2), (0, 1, 1)),
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
    def __init__(self, in_channels=20, out_channels=13):
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
            nn.Conv3d(128, 256, 3, (1, 1, 2), (0, 1, 1)),
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
            nn.Conv2d(32, out_channels, 5, 1),
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


# Region Proposal Network
class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        self.block_1 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_1 += [Conv2d(128, 128, 3, 1, 1) for _ in range(3)]
        self.block_1 = nn.Sequential(*self.block_1)

        self.block_2 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_2 += [Conv2d(128, 128, 3, 1, 1) for _ in range(5)]
        self.block_2 = nn.Sequential(*self.block_2)

        self.block_3 = [Conv2d(128, 256, 3, 2, 1)]
        self.block_3 += [nn.Conv2d(256, 256, 3, 1, 1) for _ in range(5)]
        self.block_3 = nn.Sequential(*self.block_3)

        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 4, 0),nn.BatchNorm2d(256))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(128, 256, 2, 2, 0),nn.BatchNorm2d(256))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(128, 256, 1, 1, 0),nn.BatchNorm2d(256))

        self.score_head = Conv2d(768, cfg.anchors_per_position, 1, 1, 0, activation=False, batch_norm=False)
        self.reg_head = Conv2d(768, 7 * cfg.anchors_per_position, 1, 1, 0, activation=False, batch_norm=False)

    def forward(self,x):
        x = self.block_1(x)
        x_skip_1 = x
        x = self.block_2(x)
        x_skip_2 = x
        x = self.block_3(x)
        x_0 = self.deconv_1(x)
        x_1 = self.deconv_2(x_skip_2)
        x_2 = self.deconv_3(x_skip_1)
        x = torch.cat((x_0,x_1,x_2),1)
        return self.score_head(x),self.reg_head(x)


class RF_Pose3D_25116(nn.Module):
    def __init__(self, in_channels=20, out_channels=13):
        super(RF_Pose3D_25116, self).__init__()

        self.in_channels = in_channels

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, 1),
            # nn.ZeroPad2d((2, 2, 0, 0)),  # left,right,top,bottom #25
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, 1, 1),
            # nn.ZeroPad2d((2, 2, 0, 0)),  # left,right,top,bottom #25
            nn.BatchNorm3d(16),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, 3, 1),
            # nn.ZeroPad2d((2, 2, 0, 0)),  # left,right,top,bottom #25
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
            # nn.ZeroPad2d((2, 2, 0, 0)),  # left,right,top,bottom #25
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, 3, 1),
            # nn.ZeroPad2d((2, 2, 0, 0)),  # left,right,top,bottom #25
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5, 1, 1),
            # nn.MaxPool2d((1, 3), 1),
            nn.PReLU()
        )

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 5, 1),
            # nn.MaxPool2d((1, 3), 1),
            nn.PReLU()
        )

        self.output = nn.Sequential(
            nn.ConvTranspose2d(16, 13, 1, 1),
            # nn.MaxPool2d((1, 3), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.reshape(x, (-1, 64, 110, 19))
        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.output(x)
        return x


class RF_Pose(nn.Module):
    def __init__(self, filters_encode, filters_decode, in_channels=30, out_channels=13):
        super(RF_Pose, self).__init__()

        self.in_channels = in_channels
        self.conv = []

        for idx, filter_encode in enumerate(filters_encode):
            if idx == 0:
                self.inlayer = nn.Sequential(
                    nn.Conv2d(in_channels=8,
                              out_channels=filter_encode,
                              kernel_size=(1, 3, 3),
                              stride=1,
                              # padding=(2, 2, 2, 2) #left,right,top,bottom
                              ),
                    nn.BatchNorm2d(filter_encode),
                    nn.ReLU(),
                )
            else:
                self.conv[idx - 1] = nn.Sequential(
                    nn.Conv2d(in_channels=filters_encode[idx - 1],
                              out_channels=filter_encode,
                              kernel_size=(1, 3, 3),
                              stride=1,
                              # padding=(2, 2, 2, 2) #left,right,top,bottom
                              ),
                    nn.BatchNorm2d(filter_encode),
                    nn.ReLU(),
                )

        for idx, filter_decode in enumerate(filters_decode):
            if idx == 0:
                self.dconv[idx] = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=filter_encode,
                                       out_channels=filter_decode,
                                       kernel_size=(1, 5, 5),
                                       stride=1,
                                       ),
                    nn.PReLU(),
                )
            else:
                self.dconv[idx] = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=filters_decode[idx - 1],
                                       out_channels=filter_decode,
                                       kernel_size=(1, 5, 5),
                                       stride=1,
                                       ),
                    nn.PReLU(),
                )
        self.outlayer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=filter_decode,
                               out_channels=out_channels,
                               kernel_size=(5, 5),
                               stride=1,
                               ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.inlayer(x)
        for encode in self.conv:
            x = encode(x)
        for decode in self.dconv:
            x = decode(x)
        x = self.outlayer(x)

        return x



