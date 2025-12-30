import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.retinaface_r import RetinaFace
# from nets.retinaface_NonLocal import RetinaFace
# from nets.retinaface50_self import RetinaFace
# from nets.retinaface_att import RetinaFace
# from nets.retinaface_152 import RetinaFace
# from nets.retinaface_backbone_att import RetinaFace
# from nets.retinaface_backbone_fpn_att import RetinaFace
from nets.retinaface_training import MultiBoxLoss, weights_init
from utils.anchors import Anchors
from utils.callbacks import LossHistory
from utils.config import cfg_mnet, cfg_re50, cfg_re50_self, cfg_re152, cfg_re101
from utils.dataloader import DataGenerator, detection_collate
# from utils.utils_fit_change import fit_one_epoch  # save in logs
import torch
from tqdm import tqdm
from utils.utils import get_lr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils

from torchvision import models

from nets.layers import FPN, SSH
from nets.mobilenet025 import MobileNetV1
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

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
    backbone = "resnet50"
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
    model_path = ""
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

    if backbone == "mobilenet":
        cfg = cfg_mnet
    elif backbone == "resnet50":
        cfg = cfg_re50
    elif backbone == "resnet152":
        cfg = cfg_re152
    elif backbone == "resnet101":
        cfg = cfg_re101
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))



    # ---------------------------------------------------#
    #   种类预测（是否包含人脸）
    # ---------------------------------------------------#
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
                    checkpoint = torch.load("./model_data/mobilenetV1X0.25_pretrain.tar",
                                            map_location=torch.device('cpu'))
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint['state_dict'].items():
                        name = k[7:]
                        new_state_dict[name] = v
                    backbone.load_state_dict(new_state_dict)
            elif cfg['name'] == 'Resnet50':
                backbone = models.resnet50(pretrained=pretrained)
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

            self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
            self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
            self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

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
            # print(out.size())
            # 出来三个out输出fpn特征金字塔
            # -------------------------------------------#
            #   获得三个shape的有效特征层
            #   分别是output1  80, 80, 64
            #         output2  40, 40, 64
            #         output3  20, 20, 64
            # -------------------------------------------#
            fpn = self.fpn.forward(out)

            feature1 = self.ssh1(fpn[0])
            feature2 = self.ssh2(fpn[1])
            feature3 = self.ssh3(fpn[2])
            features = [feature1, feature2, feature3]

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
    loss_history = LossHistory("logs_50_3_r")


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
        save_period = 2
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
            torch.save(model.state_dict(), 'logs_50_3_r/Epoch%d-Total_Loss%.4f.pth' % (
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
        Batch_size = 16
        Init_Epoch = 0
        Freeze_Epoch = 50

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
        Batch_size = 16
        Freeze_Epoch = 50
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
