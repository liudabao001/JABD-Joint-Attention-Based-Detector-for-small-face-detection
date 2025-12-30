#----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import time

import cv2
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"
import torch
from utils.utils_bbox import (decode, decode_landm, non_max_suppression,
                              retinaface_correct_boxes)
from utils.anchors import Anchors
from nets.retinaface_eca_nonlocal import RetinaFace
import torch.nn as nn
from utils.config import cfg_mnet, cfg_re50, cfg_re50_self, cfg_re152_
from utils.utils import letterbox_image, preprocess_input
# from retinaface import Retinaface
# from retinaface_152 import Retinaface
# from retinaface_i import Retinaface
# from retinaface_152 import Retinaface
if __name__ == "__main__":
    class Retinaface(object):
        _defaults = {
            # ---------------------------------------------------------------------#
            #   使用自己训练好的模型进行预测一定要修改model_path
            #   model_path指向logs文件夹下的权值文件
            #   训练好后logs文件夹下存在多个权值文件，选择损失较低的即可。
            # ---------------------------------------------------------------------#
            "model_path": 'logs_mobilenetV1/Epoch103-Total_Loss3.3908.pth',
            # "model_path"        : '/mnt/D/zhangguoliang/retinaface-pytorch-1.0-bilibili/Resnet50_Final.pth',
            # ---------------------------------------------------------------------#
            #   所使用的的主干网络：mobilenet、resnet50
            # ---------------------------------------------------------------------#
            "backbone": 'resnet50',
            # ---------------------------------------------------------------------#
            #   只有得分大于置信度的预测框会被保留下来
            # ---------------------------------------------------------------------#
            "confidence": 0.5,
            # ---------------------------------------------------------------------#
            #   非极大抑制所用到的nms_iou大小
            # ---------------------------------------------------------------------#
            "nms_iou": 0.45,
            # ---------------------------------------------------------------------#
            #   是否需要进行图像大小限制。
            #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
            #   可根据输入图像的大小自行调整input_shape，注意为32的倍数，如[640, 640, 3]
            # ---------------------------------------------------------------------#
            "input_shape": [1280, 1280, 3],
            # ---------------------------------------------------------------------#
            #   是否需要进行图像大小限制。
            # ---------------------------------------------------------------------#
            "letterbox_image": True,
            # --------------------------------#
            #   是否使用Cuda
            #   没有GPU可以设置成False
            # --------------------------------#
            "cuda": True,
        }

        @classmethod
        def get_defaults(cls, n):
            if n in cls._defaults:
                return cls._defaults[n]
            else:
                return "Unrecognized attribute name '" + n + "'"

        # ---------------------------------------------------#
        #   初始化Retinaface
        # ---------------------------------------------------#
        def __init__(self, **kwargs):
            self.__dict__.update(self._defaults)
            for name, value in kwargs.items():
                setattr(self, name, value)

            # ---------------------------------------------------#
            #   不同主干网络的config信息
            # ---------------------------------------------------#
            if self.backbone == "mobilenet":
                self.cfg = cfg_mnet
            elif self.backbone == "resnet50":
                self.cfg = cfg_re50
            elif self.backbone == "resnet152":
                self.cfg = cfg_re152_

            # ---------------------------------------------------#
            #   先验框的生成
            # ---------------------------------------------------#
            if self.letterbox_image:
                self.anchors = Anchors(self.cfg, image_size=[self.input_shape[0], self.input_shape[1]]).get_anchors()
            self.generate()

        # ---------------------------------------------------#
        #   载入模型
        # ---------------------------------------------------#
        def generate(self):
            # -------------------------------#
            #   载入模型与权值
            # -------------------------------#
            self.net = RetinaFace(cfg=self.cfg, mode='eval').eval()

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.net.load_state_dict(torch.load(self.model_path, map_location=device))
            self.net = self.net.eval()
            print('{} model, and classes loaded.'.format(self.model_path))

            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

        # ---------------------------------------------------#
        #   检测图片
        # ---------------------------------------------------#
        def detect_image(self, image):
            # ---------------------------------------------------#
            #   对输入图像进行一个备份，后面用于绘图
            # ---------------------------------------------------#
            old_image = image.copy()
            # ---------------------------------------------------#
            #   把图像转换成numpy的形式
            # ---------------------------------------------------#
            image = np.array(image, np.float32)
            # ---------------------------------------------------#
            #   计算输入图片的高和宽
            # ---------------------------------------------------#
            im_height, im_width, _ = np.shape(image)
            # ---------------------------------------------------#
            #   计算scale，用于将获得的预测框转换成原图的高宽
            # ---------------------------------------------------#
            scale = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
            ]
            scale_for_landmarks = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0]
            ]

            # ---------------------------------------------------------#
            #   letterbox_image可以给图像增加灰条，实现不失真的resize
            # ---------------------------------------------------------#
            if self.letterbox_image:
                image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
            else:
                self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

            with torch.no_grad():
                # -----------------------------------------------------------#
                #   图片预处理，归一化。
                # -----------------------------------------------------------#
                image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(
                    torch.FloatTensor)

                if self.cuda:
                    self.anchors = self.anchors.cuda()
                    image = image.cuda()

                # ---------------------------------------------------------#
                #   传入网络进行预测
                # ---------------------------------------------------------#
                loc, conf, landms = self.net(image)

                # -----------------------------------------------------------#
                #   对预测框进行解码
                # -----------------------------------------------------------#
                boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
                # -----------------------------------------------------------#
                #   获得预测结果的置信度
                # -----------------------------------------------------------#
                conf = conf.data.squeeze(0)[:, 1:2]
                # -----------------------------------------------------------#
                #   对人脸关键点进行解码
                # -----------------------------------------------------------#
                landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])

                # -----------------------------------------------------------#
                #   对人脸识别结果进行堆叠
                # -----------------------------------------------------------#
                boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

                if len(boxes_conf_landms) <= 0:
                    return old_image

                # ---------------------------------------------------------#
                #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
                # ---------------------------------------------------------#
                if self.letterbox_image:
                    boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                                                                 np.array([self.input_shape[0], self.input_shape[1]]),
                                                                 np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

            for b in boxes_conf_landms:
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                # ---------------------------------------------------#
                #   b[0]-b[3]为人脸框的坐标，b[4]为得分
                # ---------------------------------------------------#

                # test = int(b[2]-1.5(b[2]-b[0]))
                # print()
                # n1 = b[3]+1.2(b[1]-b[3])
                # m2 = b[0]+1.2(b[2]-b[0])
                # n2 = int(b[1]-1.2(b[1]-b[3]))
                # cv2.rectangle(old_image, (m1, n1), (m2, n2), (0, 0, 255), 2)
                a = [0, 0, 0, 0]
                w = b[2] - b[0]
                h = b[3] - b[1]
                a[0] = b[0] - int(w * 0.8)
                a[1] = b[1] - int(h * 0.8)
                a[2] = b[2] + int(w * 0.8)
                a[3] = b[3] + int(h * 0.8)
                cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)  # hua juxing
                # cv2.rectangle(old_image, (a[0], a[1]), (a[2], a[3]), (0, 0, 255), 2)  # hua juxing
                # cv2.imshow("zglsb",old_image)
                # cv2.waitKey(0)
                w = b[2] - b[0]
                h = b[3] - b[1]
                # newb0 = b[0] - int(w * 1)
                # newb1 = b[0] + int(h * 1.2)
                # newb2 = b[0] + int(w * 1.2)
                #
                # newb3 = b[0] - int(h * 1.2)

                a = [0, 0, 0, 0]
                a[0] = b[0] - int(w * 1.2)
                a[1] = b[1] - int(h * 1.2)
                a[2] = b[2] + int(w * 1.2)
                a[3] = b[3] + int(h * 1.2)

                cx = b[0]
                cy = b[1] + 12
                cv2.putText(old_image, text, (cx, cy),  # write word
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # print(b[0], b[1], b[2], b[3], b[4])
                # ---------------------------------------------------#
                #   b[5]-b[14]为人脸关键点的坐标
                # ---------------------------------------------------#
                cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)  # 5 facial landmark
            image = image.cpu()
            torch.cuda.empty_cache()
            return old_image

        def get_FPS(self, image, test_interval):
            # ---------------------------------------------------#
            #   把图像转换成numpy的形式
            # ---------------------------------------------------#
            image = np.array(image, np.float32)
            # ---------------------------------------------------#
            #   计算输入图片的高和宽
            # ---------------------------------------------------#
            im_height, im_width, _ = np.shape(image)

            # ---------------------------------------------------------#
            #   letterbox_image可以给图像增加灰条，实现不失真的resize
            # ---------------------------------------------------------#
            if self.letterbox_image:
                image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
            else:
                self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

            with torch.no_grad():
                # -----------------------------------------------------------#
                #   图片预处理，归一化。
                # -----------------------------------------------------------#
                image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(
                    torch.FloatTensor)

                if self.cuda:
                    self.anchors = self.anchors.cuda()
                    image = image.cuda()

                # ---------------------------------------------------------#
                #   传入网络进行预测
                # ---------------------------------------------------------#
                loc, conf, landms = self.net(image)
                # -----------------------------------------------------------#
                #   对预测框进行解码
                # -----------------------------------------------------------#
                boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
                # -----------------------------------------------------------#
                #   获得预测结果的置信度
                # -----------------------------------------------------------#
                conf = conf.data.squeeze(0)[:, 1:2]
                # -----------------------------------------------------------#
                #   对人脸关键点进行解码
                # -----------------------------------------------------------#
                landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])

                # -----------------------------------------------------------#
                #   对人脸识别结果进行堆叠
                # -----------------------------------------------------------#
                boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            t1 = time.time()
            for _ in range(test_interval):
                with torch.no_grad():
                    # ---------------------------------------------------------#
                    #   传入网络进行预测
                    # ---------------------------------------------------------#
                    loc, conf, landms = self.net(image)
                    # -----------------------------------------------------------#
                    #   对预测框进行解码
                    # -----------------------------------------------------------#
                    boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
                    # -----------------------------------------------------------#
                    #   获得预测结果的置信度
                    # -----------------------------------------------------------#
                    conf = conf.data.squeeze(0)[:, 1:2]
                    # -----------------------------------------------------------#
                    #   对人脸关键点进行解码
                    # -----------------------------------------------------------#
                    landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])

                    # -----------------------------------------------------------#
                    #   对人脸识别结果进行堆叠
                    # -----------------------------------------------------------#
                    boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
                    boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            t2 = time.time()
            tact_time = (t2 - t1) / test_interval
            return tact_time

        # ---------------------------------------------------#
        #   检测图片
        # ---------------------------------------------------#
        def get_map_txt(self, image):
            # ---------------------------------------------------#
            #   把图像转换成numpy的形式
            # ---------------------------------------------------#
            image = np.array(image, np.float32)
            # ---------------------------------------------------#
            #   计算输入图片的高和宽
            # ---------------------------------------------------#
            im_height, im_width, _ = np.shape(image)
            # ---------------------------------------------------#
            #   计算scale，用于将获得的预测框转换成原图的高宽
            # ---------------------------------------------------#
            scale = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
            ]
            scale_for_landmarks = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0]
            ]

            # ---------------------------------------------------------#
            #   letterbox_image可以给图像增加灰条，实现不失真的resize
            # ---------------------------------------------------------#
            if self.letterbox_image:
                image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
            else:
                self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

            with torch.no_grad():
                # -----------------------------------------------------------#
                #   图片预处理，归一化。
                # -----------------------------------------------------------#
                image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(
                    torch.FloatTensor)

                if self.cuda:
                    self.anchors = self.anchors.cuda()
                    image = image.cuda()

                # ---------------------------------------------------------#
                #   传入网络进行预测
                # ---------------------------------------------------------#
                loc, conf, landms = self.net(image)
                # -----------------------------------------------------------#
                #   对预测框进行解码
                # -----------------------------------------------------------#
                boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
                # -----------------------------------------------------------#
                #   获得预测结果的置信度
                # -----------------------------------------------------------#
                conf = conf.data.squeeze(0)[:, 1:2]
                # -----------------------------------------------------------#
                #   对人脸关键点进行解码
                # -----------------------------------------------------------#
                landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])

                # -----------------------------------------------------------#
                #   对人脸识别结果进行堆叠
                # -----------------------------------------------------------#
                boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

                if len(boxes_conf_landms) <= 0:
                    return np.array([])

                # ---------------------------------------------------------#
                #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
                # ---------------------------------------------------------#
                if self.letterbox_image:
                    boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                                                                 np.array([self.input_shape[0], self.input_shape[1]]),
                                                                 np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

            return boxes_conf_landms
    retinaface = Retinaface()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #----------------------------------------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = r"/mnt/D/zhangguoliang/retinaface-pytorch-1.0-bilibili/img/【纯净】11个绝平绝杀素材（大概是素材）.mp4"
    video_save_path = "/mnt/D/zhangguoliang/retinaface-pytorch-1.0-bilibili/results/foot.mp4"
    video_fps       = 25.0
    #-------------------------------------------------------------------------#
    #   test_interval用于指定测量fps的时候，图片检测的次数
    #   理论上test_interval越大，fps越准确。
    #-------------------------------------------------------------------------#
    test_interval   = 100
    #-------------------------------------------------------------------------#
    #   dir_origin_path指定了用于检测的图片的文件夹路径
    #   dir_save_path指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "/mnt/D/zhangguoliang/retinaface-pytorch-1.0-bilibili/test_img"
    dir_save_path   = "/mnt/D/zhangguoliang/retinaface-pytorch-1.0-bilibili/test_img_result"

    if mode == "predict":
        '''
        predict.py有几个注意点
        1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用cv2.imread打开图片文件进行预测。
        2、如果想要保存，利用cv2.imwrite("img.jpg", r_image)即可保存。
        3、如果想要获得框的坐标，可以进入detect_image函数，读取(b[0], b[1]), (b[2], b[3])这四个值。
        4、如果想要截取下目标，可以利用获取到的(b[0], b[1]), (b[2], b[3])这四个值在原图上利用矩阵的方式进行截取。
        5、在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
        '''
        while True:
            # img1 = input('Input image filename:')
            # img2 = "img"+"/"+img1
            img2 = r"F:\study\project\毕业\2080ti\data\data\widerface\val\images\56--Voter\56_Voter_peoplevoting_56_122.jpg"
            image = cv2.imread(img2)
            if image is None:
                print('Open Error! Try again!')
                continue

            else:
                image   = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                r_image = retinaface.detect_image(image)
                r_image = cv2.cvtColor(r_image,cv2.COLOR_RGB2BGR)
                cv2.imshow("after",r_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()



    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
            
        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 进行检测
            frame = np.array(retinaface.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                    
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            # frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = cv2.imread('img/street.jpg')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        tact_time = retinaface.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = cv2.imread(image_path)
                image       = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                r_image     = retinaface.detect_image(image)
                r_image     = cv2.cvtColor(r_image,cv2.COLOR_RGB2BGR)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                cv2.imwrite(os.path.join(dir_save_path, img_name), r_image)
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")


