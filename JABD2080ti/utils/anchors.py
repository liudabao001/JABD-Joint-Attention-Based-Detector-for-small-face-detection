from itertools import product as product
from math import ceil

import torch
# from utils.config import cfg_mnet, cfg_re50_self


class Anchors(object):
    def __init__(self, cfg, image_size=None):
        super(Anchors, self).__init__()
        self.min_sizes  = cfg['min_sizes']
        self.steps      = cfg['steps']
        self.clip       = cfg['clip']
        #---------------------------#
        #   图片的尺寸
        #---------------------------#
        self.image_size = image_size
        #---------------------------#
        #   三个有效特征层高和宽
        #---------------------------#
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def get_anchors(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            #-----------------------------------------#
            #   对特征层的高和宽进行循环迭代
            #-----------------------------------------#
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
class Anchors_eval(object):
    def __init__(self, cfg, image_size=None):
        super(Anchors_eval, self).__init__()
        # self.min_sizes  = cfg['min_sizes_eval']
        self.min_sizes  = cfg['min_sizes']
        # self.steps      = cfg['steps_eval']
        self.steps      = cfg['steps']
        self.clip       = cfg['clip']
        #---------------------------#
        #   图片的尺寸
        #---------------------------#
        self.image_size = image_size
        #---------------------------#
        #   三个有效特征层高和宽
        #---------------------------#
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def get_anchors(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            #-----------------------------------------#
            #   对特征层的高和宽进行循环迭代
            #-----------------------------------------#
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


cfg = {
    'name'              : 'Resnet50_self',
    # 'min_sizes'         : [[8,16],[32, 64], [64, 128], [256, 512],[240,480]],
    # 'min_sizes'         : [[8,16],[32, 64], [64, 128], [256, 512]],
    'min_sizes'         : [[8,16],[32, 64], [64, 128], [256, 512]], # 29518
    # 'min_sizes'         : [[4,8],[16, 32], [64, 128], [256, 512]], # 锚框数量29518
    'steps'             : [ 8,16,32,64],
    # 'steps'             : [4 ,8, 16, 32],
    'variance'          : [0.1, 0.2],
    'clip'              : False,
    'loc_weight'        : 2.0,
    'train_image_size'  : 840,
    'return_layers'     : {
                           # 'layer1': 1,  64
                           'layer2': 1,  # 128
                           'layer3': 2,  # 256
                           'layer4': 3,  # 256
                           'layer5': 4,  # 512
                          },
    'in_channel'        : 256,
    'out_channel'       : 256
}
anchors = Anchors(cfg, image_size=(cfg['train_image_size'], cfg['train_image_size'])).get_anchors()
print("生成的锚框数量:", anchors.shape[0])