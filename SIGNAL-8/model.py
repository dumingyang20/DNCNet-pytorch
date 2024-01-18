import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# num_class = 8  # complex data
# num_class = 11  # pkl file
num_class = 24  # hdf5 file


class DNCNet(nn.Module):
    def __init__(self):
        super(DNCNet, self).__init__()
        self.fcn = FCN()
        self.unet = UNet()

    def forward(self, x):
        noise_level = self.fcn(x)
        concat_img = torch.cat([x, noise_level], dim=1)
        out = self.unet(concat_img) + x
        # out = self.unet(concat_img)
        # noise_level = adain(noise_level)
        # out = adain(out)
        return noise_level, out


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(2, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.outc = nn.Sequential(
            nn.Conv1d(32, 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        conv1 = self.inc(x)
        conv2 = self.conv(conv1)
        conv3 = self.conv(conv2)
        conv4 = self.conv(conv3)
        conv5 = self.outc(conv4)
        return conv5


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.inc = nn.Sequential(
            single_conv(4, 64),
            single_conv(64, 64)
        )

        self.down1 = nn.AvgPool1d(2)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.down2 = nn.AvgPool1d(2)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.outc = outconv(64, 2)

    def forward(self, x):
        inx = self.inc(x)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):  # up sampling
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch//2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[1] - x1.size()[1]
        diffX = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class fixed_loss(nn.Module):
    """
    loss = output - noisy_signal - noise
    """
    def __init__(self):
        super().__init__()

    def forward(self, noise_free, out_image, est_noise):
        h_x = est_noise.size()[1]
        w_x = est_noise.size()[2]
        count_h = self._tensor_size(est_noise[:, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, 1:])
        h_tv = torch.pow((est_noise[:, 1:, :] - est_noise[:, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((est_noise[:, :, 1:] - est_noise[:, :, :w_x-1]), 2).sum()

        asymloss = torch.mean(torch.mul(torch.abs(0.3 - F.relu(noise_free - est_noise)),
                                        torch.pow(est_noise - noise_free, 2)))

        denoised_loss = torch.mean(torch.pow((out_image - noise_free), 2))

        smooth_loss = h_tv / count_h + w_tv / count_w
        loss = denoised_loss
        # loss = denoised_loss + 0.05 * smooth_loss + 0.5 * asymloss
        return loss

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]


class LeNet(nn.Module):  # input size: 2*1000
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, stride=1, kernel_size=11, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, stride=1, kernel_size=7, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64 * 247, out_features=1024, bias=True)  # for length=1000
        # self.fc1 = nn.Linear(in_features=64 * 29, out_features=1024, bias=True)  # for length=128 (RadioML, I/Q)
        # self.fc1 = nn.Linear(in_features=64 * 253, out_features=1024, bias=True)  # for length=1024 (RadioML, I/Q)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_class, bias=True)
        self.drop = nn.Dropout(p=0.5, inplace=False)

        # fix the classifier
        for param in self.parameters():
            param.requeires_grad = False

    def forward(self, x):
        y1 = F.relu(self.conv1(x))
        y1 = self.pool(y1)
        y2 = F.relu(self.conv2(y1))
        y2 = self.pool(y2)
        y3 = y2.view(y2.shape[0], -1)
        y3 = F.relu(self.fc1(y3))
        y3 = self.drop(y3)
        out = F.softmax(self.fc2(y3), dim=1)
        return out


# prepare for building the ResNet
class BasicBlock(nn.Module):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm1d(planes)  # the number of output channels
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        # in_planes should has the same channels with planes after expansion
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride,
                                                    bias=False),
                                          nn.BatchNorm1d(self.expansion*planes))

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv1(x))
        # out = self.bn2(self.conv2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """for resnet more than 34 layers"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):  # input size: 2*1024
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm1d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * 32, num_class)  # for ResNet(BasicBlock, [2, 2, 2, 2]), length=1024
        # self.linear = nn.Linear(128 * 128, num_class)  # for ResNet(BasicBlock, [2, 2]), length=1024

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool1d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
    # return ResNet(BasicBlock, [2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


class VGG(nn.Module):  # input size: 2*1000 when deal with complex data
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1_1 = nn.Conv1d(in_channels=2, out_channels=64, stride=1, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv1d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv1d(in_channels=64, out_channels=128, stride=1, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv1d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv1d(in_channels=128, out_channels=256, stride=1, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv1d(in_channels=256, out_channels=256, stride=1, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv1d(in_channels=256, out_channels=256, stride=1, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv1d(in_channels=256, out_channels=512, stride=1, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv1d(in_channels=512, out_channels=512, stride=1, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv1d(in_channels=512, out_channels=512, stride=1, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv1d(in_channels=512, out_channels=512, stride=1, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv1d(in_channels=512, out_channels=512, stride=1, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv1d(in_channels=512, out_channels=512, stride=1, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # self.fc1 = nn.Linear(in_features=512 * 31, out_features=4096, bias=True)  # for length=1000
        self.fc1 = nn.Linear(in_features=512 * 32, out_features=4096, bias=True)  # for length=1024
        self.fc2 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_class, bias=True)

        # fix the classifier
        # for param in self.parameters():
        #     param.requeires_grad = False

    def forward(self, x):
        y1 = F.relu(self.conv1_1(x))   # output size: 64*900
        y1 = F.relu(self.conv1_2(y1))  # output size: 64*900
        y1 = self.pool(y1)             # output size: 64*450

        y2 = F.relu(self.conv2_1(y1))  # output size: 128*450
        y2 = F.relu(self.conv2_2(y2))  # output size: 128*450
        y2 = self.pool(y2)             # output size: 128*225

        y3 = F.relu(self.conv3_1(y2))  # output size: 256*225
        y3 = F.relu(self.conv3_2(y3))  # output size: 256*225
        y3 = F.relu(self.conv3_3(y3))  # output size: 256*225
        y3 = self.pool(y3)             # output size: 256*112

        y4 = F.relu(self.conv4_1(y3))  # output size: 512*112
        y4 = F.relu(self.conv4_2(y4))  # output size: 512*112
        y4 = F.relu(self.conv4_3(y4))  # output size: 512*112
        y4 = self.pool(y4)             # output size: 512*56

        y5 = F.relu(self.conv5_1(y4))  # output size: 512*56
        y5 = F.relu(self.conv5_2(y5))  # output size: 512*56
        y5 = F.relu(self.conv5_3(y5))  # output size: 512*56
        y5 = self.pool(y5)             # output size: 512*28

        y = y5.view(y5.shape[0], -1)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        out = self.fc3(y)
        return out


