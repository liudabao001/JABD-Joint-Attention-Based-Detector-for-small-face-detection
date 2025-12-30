import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from torchvision import models

from nets.layers import  SSH
from nets.mobilenet025 import MobileNetV1


def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


# ---------------------------------------------------#
#   卷积块
#   Conv2D + BatchNormalization
# ---------------------------------------------------#
def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )
class FPN(nn.Module):
    def  __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

        self.Nlm = NLM(256)

    def forward(self, inputs):
        #-------------------------------------------#
        #   获得三个shape的有效特征层
        #   分别是C3  80, 80, 64
        #         C4  40, 40, 128
        #         C5  20, 20, 256
        #-------------------------------------------#
        # inputs = list(inputs.values())

        #-------------------------------------------#
        #   获得三个shape的有效特征层
        #   分别是output1  80, 80, 64
        #         output2  40, 40, 64
        #         output3  20, 20, 64
        #-------------------------------------------#
        output1 = self.output1(inputs[0])
        output2 = self.output2(inputs[1])
        output3 = self.output3(inputs[2])

        #-------------------------------------------#
        #   output3上采样和output2特征融合
        #   output2  40, 40, 64
        #-------------------------------------------#
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        up3 = self.Nlm(up3)
        output2 = output2 + up3
        output2 = self.merge2(output2)

        #-------------------------------------------#
        #   output2上采样和output1特征融合
        #   output1  80, 80, 64
        #-------------------------------------------#
        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        up2 = self.Nlm(up2)
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out

# ---------------------------------------------------#
#   种类预测（是否包含人脸）
# ---------------------------------------------------#
class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


# ---------------------------------------------------#
#   预测框预测
# ---------------------------------------------------#
class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


class IOUHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(IOUHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 1)
class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


class NLM(nn.Module):
    def __init__(self, in_channels, scale=1, psp_size=(1, 4, 8, 12), ch=4):  # psp_size=(1, 3, 6, 8)
        super(NLM, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.ch = ch
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_query = nn.Conv2d(in_channels=self.in_channels, out_channels=self.ch, kernel_size=1)
        self.f_key = nn.Conv2d(in_channels=self.in_channels, out_channels=self.ch, kernel_size=1)
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.ch, kernel_size=1)

        self.psp = PSPModule(psp_size)

        self.W = nn.Conv2d(in_channels=self.ch, out_channels=self.in_channels, kernel_size=1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        # query: Nx1xHxW -> Nx1xHW
        query = self.f_query(x).view(batch_size, self.ch, -1)
        # Nx1xHW -> NxHWx1
        query = query.permute(0, 2, 1)

        # key：Nx1xS
        key = self.f_key(x)
        key = self.psp(key)

        # value: Nx1xHW -> Nx1xS （S = 110）
        value = self.psp(self.f_value(x))
        # Nx1xS -> NxSx1 （S = 110）
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (1 ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.ch, h, w)
        context = self.W(context)
        context += x
        return context


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        # y = y*F.relu6(y+3,inplace=True)/6
        return x * y.expand_as(x)
# ---------------------------------------------------#
#   人脸关键点预测
# ---------------------------------------------------#
class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, cfg=None, pretrained=False, mode='train'):
        super(RetinaFace, self).__init__()
        backbone = None
        # -------------------------------------------#
        #   选择使用mobilenet0.25、resnet50作为主干
        # -------------------------------------------#
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if pretrained:
                checkpoint = torch.load("./model_data/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]
                    new_state_dict[name] = v
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            backbone = models.resnet50(pretrained=pretrained)
        elif cfg['name'] == 'Resnet152':
            backbone = models.resnet152(pretrained=pretrained)

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])

        # -------------------------------------------#
        #   获得每个初步有效特征层的通道数
        # -------------------------------------------#
        in_channels_list = [cfg['in_channel'] * 2, cfg['in_channel'] * 4, cfg['in_channel'] * 8]
        # -------------------------------------------#
        #   利用初步有效特征层构建特征金字塔
        # -------------------------------------------#
        self.fpn = FPN(in_channels_list, cfg['out_channel'])
        # -------------------------------------------#
        #   利用ssh模块提高模型感受野
        # -------------------------------------------#
        self.ssh1 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh2 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh3 = SSH(cfg['out_channel'], cfg['out_channel'])

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.IouHead = self._make_IOU_head(fpn_num=3, inchannels=cfg['out_channel'])

        self.Nlm = NLM(256)
        self.eca_64 = eca_block(512)
        self.eca_128 = eca_block(1024)
        self.eca_256 = eca_block(2048)
        self.eca_fpn = eca_block(256)

        self.mode = mode

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_IOU_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        iouhead = nn.ModuleList()
        # for i in range(fpn_num):
        #     bboxhead.append(BboxHead(inchannels,anchor_num))
        iouhead.append(BboxHead(inchannels, anchor_num))
        iouhead.append(BboxHead(inchannels, anchor_num))
        iouhead.append(IOUHead(inchannels, anchor_num))
        return iouhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        # -------------------------------------------#
        #   获得三个shape的有效特征层
        #   分别是C3  80, 80, 64
        #         C4  40, 40, 128
        #         C5  20, 20, 256
        # -------------------------------------------#
        out = self.body.forward(inputs)
        inputs = list(out.values())
        # dropout
        out1 = F.dropout(inputs[0])
        out2 = F.dropout(inputs[1])
        out3 = F.dropout(inputs[2])

        # out1 = self.eca_64(inputs[0])
        out1 = self.eca_64(out1)
        out2 = self.eca_128(out2)
        out3 = self.eca_256(out3)
        out = [out1, out2, out3]
        # print(out)

        # -------------------------------------------#
        #   获得三个shape的有效特征层
        #   分别是output1  80, 80, 64
        #         output2  40, 40, 64
        #         output3  20, 20, 64
        # -------------------------------------------#
        fpn = self.fpn.forward(out)
        # fpn[0]  (1,256,160,160)
        # fpn[1]  (1,256,80,80)
        # fpn[2]  (1,256,40,40)
        # fpn_Nonlocal1 = self.Nlm(fpn[0])
        # fpn_Nonlocal2 = self.Nlm(fpn[1])
        # fpn_Nonlocal3 = self.Nlm(fpn[2])
        feature1 = self.ssh1(self.eca_fpn(fpn[0]))
        feature2 = self.ssh2(self.eca_fpn(fpn[1]))
        feature3 = self.ssh3(self.eca_fpn(fpn[2]))
        features = [feature1, feature2, feature3]

        # -------------------------------------------#
        #   将所有结果进行堆叠
        # -------------------------------------------#
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        # Iou_re = torch.cat([self.IouHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.mode == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output


