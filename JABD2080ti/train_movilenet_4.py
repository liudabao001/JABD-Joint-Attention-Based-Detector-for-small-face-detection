import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models._utils as _utils
# from nets.resnet_pytorch_r import resnet50
import math
from torchvision import models
import torch.nn.functional as F
# from nets.retinaface_r import RetinaFace
# from nets.retinaface_NonLocal import RetinaFace
# from nets.retinaface50_self import RetinaFace
# from nets.retinaface_att import RetinaFace
# from nets.retinaface_152 import RetinaFace
# from nets.retinaface_backbone_att import RetinaFace
# from nets.retinaface_backbone_fpn_att import RetinaFace
from nets.retinaface_training import MultiBoxLoss, weights_init
from utils.anchors import Anchors
from utils.callbacks import LossHistory
from utils.config import cfg_mnet, cfg_mnet_4,cfg_re50, cfg_re50_self, cfg_re152, cfg_re101
from utils.dataloader import DataGenerator, detection_collate
from nets.mobilenetV3 import MobileNetV3_Large_4
# from utils.utils_fit_change import fit_one_epoch  # save in logs
import torch
import math
from tqdm import tqdm
from utils.utils import get_lr
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"

if __name__ == "__main__":
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = True
    # --------------------------------#
    #   获得训练用的人脸标签与坐标
    # --------------------------------#
    training_dataset_path = './data/widerface/train/label.txt'
    # -------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet或者resnet50
    # -------------------------------#
    backbone = "mobilenetV3"
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    # --------------------------------------------------------------------------------------------------------------------------
    pretrained = False
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   一般来讲，从0开始训练效果会很差，因为权值太过随机，特征提取效果不明显。
    #
    #   网络一般不从0开始训练，至少会使用主干部分的权值，有些论文提到可以不用预训练，主要原因是他们 数据集较大 且 调参能力优秀。
    #   如果一定要训练网络的主干部分，可以了解imagenet数据集，首先训练分类模型，分类模型的 主干部分 和该模型通用，基于此进行训练。
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = "logs_mobilenetV3_4/Epoch99-Total_Loss6.7624.pth"
    # model_path = ""
    # -------------------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    # -------------------------------------------------------------------#
    Freeze_Train = False
    # -------------------------------------------------------------------#
    #   用于设置是否使用多线程读取数据，0代表关闭多线程
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    # -------------------------------------------------------------------#
    num_workers = 4

    if backbone == "mobilenetV3":
        cfg = cfg_mnet_4
    elif backbone == "resnet50":
        cfg = cfg_re50
    elif backbone == "resnet152":
        cfg = cfg_re152
    elif backbone == "resnet101":
        cfg = cfg_re101
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))


    class ClassHead(nn.Module):
        def __init__(self, inchannels=512, num_anchors=2):
            super(ClassHead, self).__init__()
            self.num_anchors = num_anchors
            self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

        def forward(self, x):
            out = self.conv1x1(x)  # 调整通道
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


    class FPN(nn.Module):
        def __init__(self, in_channels_list, out_channels):
            super(FPN, self).__init__()
            leaky = 0
            if (out_channels <= 64):
                leaky = 0.1
            self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
            self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
            self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)

            self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
            self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

            self.nlm = NLM(40)
            # self.nlm = NLM(64)

        def forward(self, inputs):
            # -------------------------------------------#
            #   获得三个shape的有效特征层
            #   分别是C3  80, 80, 64
            #         C4  40, 40, 128
            #         C5  20, 20, 256
            # -------------------------------------------#
            # inputs = list(inputs.values())

            # -------------------------------------------#
            #   获得三个shape的有效特征层
            #   分别是output1  80, 80, 64
            #         output2  40, 40, 64
            #         output3  20, 20, 64
            # -------------------------------------------#
            output1 = self.output1(inputs[0])
            output2 = self.output2(inputs[1])
            output3 = self.output2(inputs[2])
            output4 = self.output3(inputs[3])

            # -------------------------------------------#
            #   output3上采样和output2特征融合
            #   output2  40, 40, 64
            # -------------------------------------------#
            up4 = F.interpolate(output4, size=[output3.size(2), output3.size(3)], mode="nearest")
            up4 = self.nlm(up4)
            output3= output3 + up4
            output3 = self.merge2(output3)

            up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
            up3 = self.nlm(up3)
            output2 = output2 + up3
            output2 = self.merge2(output2)

            # -------------------------------------------#
            #   output2上采样和output1特征融合
            #   output1  80, 80, 64
            # -------------------------------------------#
            up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
            up2 = self.nlm(up2)
            output1 = output1 + up2
            output1 = self.merge1(output1)

            out = [output1, output2, output3,output4]
            return out


    class SSH(nn.Module):
        def __init__(self, in_channel, out_channel):
            super(SSH, self).__init__()
            assert out_channel % 4 == 0
            leaky = 0
            if (out_channel <= 64):
                leaky = 0.1

            # 3x3卷积
            self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

            # 利用两个3x3卷积替代5x5卷积
            self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
            self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

            # 利用三个3x3卷积替代7x7卷积
            self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
            self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        def forward(self, inputs):
            conv3X3 = self.conv3X3(inputs)

            conv5X5_1 = self.conv5X5_1(inputs)
            conv5X5 = self.conv5X5_2(conv5X5_1)

            conv7X7_2 = self.conv7X7_2(conv5X5_1)
            conv7X7 = self.conv7x7_3(conv7X7_2)

            # 所有结果堆叠起来
            out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
            out = F.relu(out)
            return out


    class eca_block(nn.Module):
        def __init__(self, channel, b=1, gamma=2):
            super(eca_block, self).__init__()
            kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
            kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
            self.sigmoid = nn.Sigmoid()
            self.Hsigmoid = nn.Hardsigmoid()

        def forward(self, x):
            y = self.avg_pool(x)
            y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y = self.sigmoid(y)
            # y = y*F.relu6(y+3,inplace=True)/6
            return x * y.expand_as(x)

    class RetinaFace(nn.Module):
        def __init__(self, cfg=None, pretrained=False, mode='train'):
            super(RetinaFace, self).__init__()
            backbone = None
            # -------------------------------------------#
            #   选择使用mobilenet0.25、resnet50作为主干
            # -------------------------------------------#
            if cfg['name'] == 'mobilenetV3':
                # backbone = MobileNetV1()
                backbone = MobileNetV3_Large_4()
                # if pretrained:
                #     checkpoint = torch.load("./model_data/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                #     from collections import OrderedDict
                #     new_state_dict = OrderedDict()
                #     for k, v in checkpoint['state_dict'].items():
                #         name = k[7:]
                #         new_state_dict[name] = v
                #     backbone.load_state_dict(new_state_dict)
            # elif cfg['name'] == 'mobilenetV3':
            #     backbone = MobileNetV3_Large_change
            elif cfg['name'] == 'Resnet50':
                backbone = models.resnet50(pretrained=pretrained)
                # backbone  = resnet50
            elif cfg['name'] == 'Resnet152':
                # backbone = resnet152((weights=ResNet152_Weights.DEFAULT)
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
            self.ssh1 = SSH(cfg['out_channel'], cfg['out_channel'])  # 64 64
            self.ssh2 = SSH(cfg['out_channel'], cfg['out_channel'])
            self.ssh3 = SSH(cfg['out_channel'], cfg['out_channel'])  # 256
            self.ssh3 = SSH(cfg['out_channel'], cfg['out_channel'])  # 256

            self.ClassHead = self._make_class_head(fpn_num=4, inchannels=cfg['out_channel'])
            self.BboxHead = self._make_bbox_head(fpn_num=4, inchannels=cfg['out_channel'])
            self.LandmarkHead = self._make_landmark_head(fpn_num=4, inchannels=cfg['out_channel'])
            self.eca_40 = eca_block(40)
            # self.eca_40 = eca_block(64)
            self.eca_80 = eca_block(80)
            # self.eca_80 = eca_block(128)
            self.eca_160 = eca_block(160)
            # self.eca_160 = eca_block(256)
            #
            self.eca_fpn = eca_block(64)
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
            out1 = self.eca_40(inputs[0])
            out2 = self.eca_80(inputs[1])
            out3 = self.eca_80(inputs[2])
            out4 = self.eca_160(inputs[3])
            out = [out1, out2, out3,out4]
            # print(out.size())
            # 出来三个out输出fpn特征金字塔
            # -------------------------------------------#
            #   获得三个shape的有效特征层
            #   分别是output1  80, 80, 64
            #         output2  40, 40, 64
            #         output3  20, 20, 64
            # -------------------------------------------#
            fpn = self.fpn.forward(out)

            feature1 = self.ssh1(self.eca_fpn(fpn[0]))
            feature2 = self.ssh2(self.eca_fpn(fpn[1]))
            feature3 = self.ssh3(self.eca_fpn(fpn[2]))
            feature4 = self.ssh3(self.eca_fpn(fpn[3]))
            features = [feature1, feature2, feature3,feature4]

            # -------------------------------------------#
            #   将所有结果进行堆叠
            # -------------------------------------------#
            bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
            classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
            ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

            if self.mode == 'train':
                output = (bbox_regressions, classifications, ldm_regressions)
            else:
                output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
            return output


    model = RetinaFace(cfg=cfg, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    # -------------------------------#
    #   获得先验框anchors
    # -------------------------------#
    anchors = Anchors(cfg, image_size=(cfg['train_image_size'], cfg['train_image_size'])).get_anchors()
    if Cuda:
        anchors = anchors.cuda()

    criterion = MultiBoxLoss(2, 0.35, 7, cfg['variance'], Cuda)
    loss_history = LossHistory("logs_mobilenetV3_4")


    # ---------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   受到BatchNorm层影响，batch_size最小为2，不能为1。
    # ---------------------------------------------------------#
    # ---------------------------------------------------------#
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ---------------------------------------------------------#
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']


    def fit_one_epoch(model_train, model, loss_history, optimizer, criterion, epoch, epoch_step, gen, Epoch, anchors,
                      cfg,
                      cuda):
        save_period = 1
        total_r_loss = 0
        total_c_loss = 0
        total_landmark_loss = 0

        print('Start Train')
        with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(gen):
                if iteration >= epoch_step:
                    break
                images, targets = batch[0], batch[1]
                if len(images) == 0:
                    continue
                with torch.no_grad():
                    if cuda:
                        images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                        targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                    else:
                        images = torch.from_numpy(images).type(torch.FloatTensor)
                        targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                # ----------------------#
                #   清零梯度
                # ----------------------#
                optimizer.zero_grad()
                # ----------------------#
                #   前向传播
                # ----------------------#
                out = model_train(images)
                # ----------------------#
                #   计算损失
                # ----------------------#
                r_loss, c_loss, landm_loss = criterion(out, anchors, targets)
                loss = cfg['loc_weight'] * r_loss + c_loss + landm_loss

                loss.backward()
                optimizer.step()

                total_c_loss += c_loss.item()
                total_r_loss += cfg['loc_weight'] * r_loss.item()
                total_landmark_loss += landm_loss.item()

                pbar.set_postfix(**{'Conf Loss': total_c_loss / (iteration + 1),
                                    'Regression Loss': total_r_loss / (iteration + 1),
                                    'LandMark Loss': total_landmark_loss / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

        print('Saving state, iter:', str(epoch + 1))
        if (epoch + 1) % save_period == 0:
            torch.save(model.state_dict(), 'logs_mobilenetV3_4/Epoch%d-Total_Loss%.4f.pth' % (
                (epoch + 1), (total_c_loss + total_r_loss + total_landmark_loss) / (epoch_step + 1)),
                       _use_new_zipfile_serialization=False)
        loss_history.append_loss((total_c_loss + total_r_loss + total_landmark_loss) / (epoch_step + 1))


    if True:
        # ----------------------------------------------------#
        #   冻结阶段训练参数
        #   此时模型的主干被冻结了，特征提取网络不发生改变
        #   占用的显存较小，仅对网络进行微调
        # ----------------------------------------------------#
        lr = 1e-3
        Batch_size = 18
        Init_Epoch = 0
        Freeze_Epoch = 55

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = DataGenerator(training_dataset_path, cfg['train_image_size'])
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=detection_collate)

        epoch_step = train_dataset.get_len() // Batch_size

        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.body.parameters():
                param.requires_grad = False

        for epoch in range(Init_Epoch, Freeze_Epoch):
            fit_one_epoch(model_train, model, loss_history, optimizer, criterion, epoch, epoch_step, gen, Freeze_Epoch,
                          anchors, cfg, Cuda)
            lr_scheduler.step()

    if True:
        # ----------------------------------------------------#
        #   解冻阶段训练参数
        #   此时模型的主干不被冻结了，特征提取网络会发生改变
        #   占用的显存较大，网络所有的参数都会发生改变
        # ----------------------------------------------------#
        lr = 1e-4
        Batch_size = 18
        Freeze_Epoch = 55
        Unfreeze_Epoch = 10000

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = DataGenerator(training_dataset_path, cfg['train_image_size'])
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=detection_collate)

        epoch_step = train_dataset.get_len() // Batch_size

        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.body.parameters():
                param.requires_grad = True

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            fit_one_epoch(model_train, model, loss_history, optimizer, criterion, epoch, epoch_step, gen,
                          Unfreeze_Epoch, anchors, cfg, Cuda)
            lr_scheduler.step()
