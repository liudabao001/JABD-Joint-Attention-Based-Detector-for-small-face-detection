import os

import torch.nn as nn
import torch

import torchvision.models as models
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,  # 残差块的选择：如果定义ResNet18/34时，就选择基础模块(BasicBlock)，如果定义ResNet50/101/152，就使用瓶颈模块(Bottleneck)
                 blocks_num,  # 定义所使用的残差块的数量，它是一个列表参数
                 num_classes=1000,  # 网络的分类个数
                 ):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 下面的 self.layer1，self.layer2，self.layer3，self.layer4分别是不同的模块，即对应着上面表格中的conv1，conv2_x，conv3_x，conv4_x，conv5_x中的残差结构
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 256, blocks_num[3], stride=2)
        self.layer5 = self._make_layer(block, 512, blocks_num[4], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化（即全局平均池化），它会将输入特征图池化成1x1大小
        # 因为前边经过自适应平均池化后特征图大小变为1x1，并且有512 * block.expansion个通道，所以展平后的维度是512 * block.expansion，所以下面的全连接层的输入维度是512 * block.expansion
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 对卷积层的参数进行初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # 自适应平均池化（即全局平均池化），它会将输入特征图池化成1x1大小
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes=1000):
    # 预训练权重下载链接：  https://download.pytorch.org/models/resnet18-5c106cde.pth
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes=1000):
    # 预训练权重下载链接： https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes=1000):
    # 预训练权重下载链接： https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes=1000, include_top=True):
    # 预训练权重下载链接： https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def resnet152(num_classes=1000):
    # 预训练权重下载链接： https://download.pytorch.org/models/resnet152-b121ed2d.pth
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

def resnet152_self(num_classes=1000):
    # 预训练权重下载链接： https://download.pytorch.org/models/resnet152-b121ed2d.pth
    return ResNet(Bottleneck, [3, 8, 18,18, 3], num_classes=num_classes)
# device = torch.device('cuda')
# net = resnet34()
# # load pretrain weights
# # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
# model_weight_path = "./resnet34-pre.pth"  # 加载resnet的预训练模型
# assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
# net.load_state_dict(torch.load(model_weight_path, map_location=device))
# net.to(device if torch.cuda.is_available() else 'cpu')
# print(net.to(device))  # 输出模型结构

# model = resnet152_self()
# print(model)
# state_dict = model.state_dict()
# with open ('resnet152_self.txt','w') as f:
#     f.write((str)model)

Resnet152 = resnet152_self()
import torchvision.models as models


# import torch
# import torchvision.models as models
#
# # Load resnet152 model
# model = resnet152_self()
#
# # Save the model's architecture
# with open('resnet152_model_structure.txt', 'w') as f:
#     f.write(str(model))
