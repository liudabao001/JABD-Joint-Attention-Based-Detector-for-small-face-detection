cfg_mnet = {
    'name'              : 'mobilenet0.25',
    'min_sizes'         : [[16, 32], [64, 128], [256, 512]],
    'steps'             : [8, 16, 32],
    'variance'          : [0.1, 0.2],
    'clip'              : False,
    'loc_weight'        : 2.0,
    #------------------------------------------------------------------#
    #   视频上看到的训练图片大小为640，为了提高大图状态下的困难样本
    #   的识别能力，我将训练图片进行调大
    #------------------------------------------------------------------#
    'train_image_size'  : 840,
    # 'return_layers'     : {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'return_layers'     : {'layer1': 1, 'layer2': 2, 'layer3': 3},
    'in_channel'        : 20,
    # 'in_channel'        : 32,
    # 'out_channel'       : 64
    'out_channel'       : 40
}
cfg_mnet_4 =  {
    'name'              : 'mobilenetV3',
    # 'min_sizes'         : [[16, 32], [64, 128], [256, 512]],
    # 'min_sizes'         : [[8,16],[32, 64], [64, 128], [256, 512]],
    # 'min_sizes'         : [[4,8],[16, 32], [64, 128], [256, 512]],
    'min_sizes'         : [[4,12],[16, 32], [64, 128], [256, 512]],
    'steps'             : [8, 16, 16,32],
    'variance'          : [0.1, 0.2],
    'clip'              : False,
    'loc_weight'        : 2.0,
    #------------------------------------------------------------------#
    #   视频上看到的训练图片大小为640，为了提高大图状态下的困难样本
    #   的识别能力，我将训练图片进行调大
    #------------------------------------------------------------------#
    'train_image_size'  : 840,
    # 'return_layers'     : {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'return_layers'     : {'layer1': 1, 'layer2': 2, 'layer3': 3,'layer4':4},
    'in_channel'        : 20,
    # 'in_channel'        : 32,
    # 'out_channel'       : 64
    'out_channel'       : 40
}

cfg_re50 = {
    'name'              : 'Resnet50',
    'min_sizes'         : [[16, 32], [64, 128], [256, 512]],
    'steps'             : [8, 16, 32],
    'variance'          : [0.1, 0.2],
    'clip'              : False,
    'loc_weight'        : 2.0,
    'train_image_size'  : 840,
    'return_layers'     : {'layer2': 1,
                           'layer3': 2,
                           'layer4': 3},
    'in_channel'        : 256,
    'out_channel'       : 256
}
cfg_re50_self = {
    'name'              : 'Resnet50_self',
    # 'min_sizes'         : [[8,16],[32, 64], [64, 128], [256, 512],[240,480]],

    'min_sizes'         : [[8,16],[32, 64], [64, 128], [256, 512]],#可以训练 但是测试精度为0
    # 'min_sizes'         : [[4,8],[16, 32], [64, 128], [256, 512]],
    # 1,2  2 4  2 4 4 8
    # 'steps'             : [4,8, 16, 32],

    'steps'             : [ 8,16,32,64],
    # 'steps'             : [4 ,8, 16, 32,60],
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
cfg_re152_ = {
    'name'              : 'Resnet152',
    'min_sizes'         : [[16, 32], [64, 128], [256, 512]],
    'steps'             : [8, 16, 32],
    'variance'          : [0.1, 0.2],
    'clip'              : False,
    'loc_weight'        : 2.0,
    'train_image_size'  : 840,
    'return_layers'     : {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel'        : 256,
    'out_channel'       : 256
}

cfg_re152 = {
    'name'              : 'Resnet152',
    'min_sizes'         : [[8,16],[32, 64], [64, 128], [256, 512]],
    'steps'             : [4 ,8, 16, 32],
    # 'steps'             : [8, 16, 32],
    'variance'          : [0.1, 0.2],
    'clip'              : False,
    'loc_weight'        : 2.0,
    'train_image_size'  : 840,
    'return_layers'     : {
                           'layer1': 1,
                           'layer2': 2,
                           'layer3': 3,
                           'layer4': 4,
                          },
    'in_channel'        : 256,
    'out_channel'       : 256
}
cfg_re101 = {
    'name'              : 'Resnet101',
    'min_sizes'         : [[32, 64], [64, 128], [256, 512],[240,480]],
    'steps'             : [8, 16, 32,60],
    # 'steps'             : [8, 16, 32],
    'variance'          : [0.1, 0.2],
    'clip'              : False,
    'loc_weight'        : 2.0,
    'train_image_size'  : 840,
    'return_layers'     : {
                           # 'layer1': 1,
                           'layer2': 2,
                           'layer3': 3,
                           'layer4': 4,
                           'layer5': 5,
                          },
    'in_channel'        : 256,
    'out_channel'       : 256
}
cfg_re152_new = {
    'name'              : 'Resnet152',
    # 'min_sizes'         : [[8,16],[32, 64], [64, 128], [256, 512],[240,480]],
    'min_sizes'         : [[8,16],[32, 64], [64, 128], [256, 512]],
    'steps'             :  [4 ,8, 16, 32],
    # 'steps'             : [4 ,8, 16, 32,60],
    # 'steps'             : [8, 16, 32],
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
