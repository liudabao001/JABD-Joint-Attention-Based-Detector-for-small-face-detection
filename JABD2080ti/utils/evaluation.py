"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
# from bbox import bbox_overlaps
import sys
# sys.path.append(r"F:\Anaconda\conda_real\envs\facenet\Lib\site-packages\bbox")
import bbox
# from bbox.bbox_np import bbox_overlaps
from IPython import embed


def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""
    # 这个函数是用来读取WiderFace数据集的ground truth文件，得到所有测试图片中的人脸真实边界框。
    # 函数输入参数gt_dir是ground truth文件所在的目录。该函数使用Python中的scipy.io.loadmat函数来读取.mat文件
    # ，获取WiderFace数据集中的人脸真实边界框。具体来说，该函数读取四个不同的.mat文件，分别对应所有测试图片、难样本、中等样本和易样本，
    # 从中提取出每张图片的人脸真实边界框，并返回一个字典facebox_list，其中键是图片名称（去掉.jpg后缀），值是一个列表，
    # 包含该图片中所有人脸的真实边界框坐标。此外，该函数还返回三个不同难度级别的ground truth列表，分别存储难样本、中等样本和易样本的真实边界框，
    # 这些真实边界框将用于评估不同难度级别下的检测性能
    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list

def bbox_overlaps(box_a, box_b):
    A = np.shape(box_a)[0]
    B = np.shape(box_b)[0]
    # 求先验框和实际框的交集
    inter = intersect(box_a, box_b)

    area_a = np.tile(np.expand_dims(((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])),1), [1,B])  # [A,B]
    area_b = np.tile(np.expand_dims(((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])),0), [A,1])  # [A,B]
    union = area_a + area_b - inter
    out = inter / union
    return out
def intersect(box_a, box_b):
    A = np.shape(box_a)[0]
    B = np.shape(box_b)[0]
    max_xy = np.minimum(np.tile(np.expand_dims(box_a[:, 2:],1), (1, B, 1)), np.tile(np.expand_dims(box_b[:, 2:],0), (A, 1, 1)))
    min_xy = np.maximum(np.tile(np.expand_dims(box_a[:, :2],1), (1, B, 1)), np.tile(np.expand_dims(box_b[:, :2],0), (A, 1, 1)))
    inter = np.maximum((max_xy - min_xy), np.zeros_like((max_xy - min_xy)))

    return inter[:, :, 0] * inter[:, :, 1]
def bbox_overlaps_(bboxes1, bboxes2):
    """
    Calculate the Intersection-Over-Union (IOU) of two set of bboxes.
    The bboxes should be in [x1, y1, x2, y2] format.
    Args:
      bboxes1: (np.ndarray) bounding boxes, sized [N,4].
      bboxes2: (np.ndarray) bounding boxes, sized [M,4].
    Return:
      (np.ndarray) iou, sized [N,M].
    """
    # 1. calculate the areas of bboxes
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)

    # 2. calculate the intersection areas of bboxes
    inter_x1 = np.maximum(bboxes1[:, 0][:, None], bboxes2[:, 0])
    inter_y1 = np.maximum(bboxes1[:, 1][:, None], bboxes2[:, 1])
    inter_x2 = np.minimum(bboxes1[:, 2][:, None], bboxes2[:, 2])
    inter_y2 = np.minimum(bboxes1[:, 3][:, None], bboxes2[:, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1 + 1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1 + 1)
    inter_area = inter_w * inter_h

    # 3. calculate the iou scores between bboxes
    iou = inter_area / (area1[:, None] + area2 - inter_area)
    out = iou
    return out
def bbox_overlaps_last(bboxes1, bboxes2, mode='iou', eps=1e-6):
    """Calculate the overlaps between two sets of bboxes.

    Args:
        bboxes1 (ndarray): shape (n, 4) in (x1, y1, x2, y2) format.
        bboxes2 (ndarray): shape (k, 4) in (x1, y1, x2, y2) format.
        mode (str): "iou" (intersection over union) or "iof"
            (intersection over foreground).
        eps (float): a value added to the denominator for numerical stability.

    Returns:
        ndarray: shape (n, k) of pairwise overlaps between bboxes1 and bboxes2.
    """
    assert mode in ['iou', 'iof']

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)

    if rows * cols == 0:
        return ious

    if mode == 'iou':
        # calculate overlap
        for i in range(rows):
            box_i = np.broadcast_to(bboxes1[i], (cols, 4))
            box_intersection = np.minimum(box_i[:, 2:], bboxes2[:, 2:]) - np.maximum(box_i[:, :2], bboxes2[:, :2])
            box_intersection = np.maximum(box_intersection, 0.0)
            intersection_area = box_intersection[:, 0] * box_intersection[:, 1]

            box1_area = (box_i[:, 2] - box_i[:, 0]) * (box_i[:, 3] - box_i[:, 1])
            box2_area = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
            union = box1_area + box2_area - intersection_area
            ious[i, :] = intersection_area / (union + eps)
    else:
        # calculate iof (intersection / foreground)
        areas2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        for i in range(rows):
            box_i = np.broadcast_to(bboxes1[i], (cols, 4))
            box_intersection = np.minimum(box_i[:, 2:], bboxes2[:, 2:]) - np.maximum(box_i[:, :2], bboxes2[:, :2])
            box_intersection = np.maximum(box_intersection, 0.0)
            intersection_area = box_intersection[:, 0] * box_intersection[:, 1]
            ious[i, :] = intersection_area / (areas2 + eps)

    return ious

def get_gt_boxes_from_txt(gt_path, cache_dir):


    cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    if os.path.exists(cache_file):
        f = open(cache_file, 'rb')
        boxes = pickle.load(f)
        f.close()
        return boxes

    f = open(gt_path, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))
    boxes = {}
    print(len(lines))
    f.close()
    current_boxes = []
    current_name = None
    for line in lines:
        if state == 0 and '--' in line:
            state = 1
            current_name = line
            continue
        if state == 1:
            state = 2
            continue

        if state == 2 and '--' in line:
            state = 1
            boxes[current_name] = np.array(current_boxes).astype('float32')
            current_name = line
            current_boxes = []
            continue

        if state == 2:
            box = [float(x) for x in line.split(' ')[:4]]
            current_boxes.append(box)
            continue

    f = open(cache_file, 'wb')
    pickle.dump(boxes, f)
    f.close()
    return boxes


def read_pred_file(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    # b = lines[0].rstrip('\r\n').split(' ')[:-1]
    # c = float(b)
    # a = map(lambda x: [[float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4])] for a in x.rstrip('\r\n').split(' ')], lines)
    boxes = []
    for line in lines:
        line = line.rstrip('\r\n').split(' ')
        if line[0] is '':
            continue
        # a = float(line[4])
        boxes.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])])
    boxes = np.array(boxes)
    # boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')], lines))).astype('float')
    return img_file.split('/')[-1], boxes


def get_preds(pred_dir):
    # 这个函数是用来获取预测结果的。它从指定目录中读取预测结果，并将它们存储在字典中。函数输入参数pred_dir是包含预测结果文件的目录。
    # 函数遍历每个事件（通常是图片），读取包含预测结果的文本文件，并将这些结果存储在字典中，以便在后续评估过程中使用。每个事件对应一个字典，
    # 其键是图片名称（去掉.jpg后缀），值是一个列表，包含该图片中所有检测到的人脸的边界框坐标。最终返回的是一个字典，包含所有事件的预测结果。
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events)

    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname.rstrip('.jpg')] = _boxes
        boxes[event] = current_event
    return boxes


def norm_score(pred): # 1
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    这个函数是用来对预测结果中的置信度进行归一化处理的。函数输入参数pred是一个字典，包含预测结果。字典的键是事件（通常是图片）的名称，
    值是一个字典，其中键是图片名称（去掉.jpg后缀），值是一个列表，包含该图片中所有检测到的人脸的边界框坐标和置信度。在这个函数中，
    遍历每个事件和每个人脸边界框，找到所有预测结果中的最小和最大置信度，然后将置信度归一化到0到1之间。这样做是为了使不同的模型或不同的预测结果可以进行公平的比较和评估。
    函数最终返回归一化后的预测结果。
    """

    max_score = 0
    min_score = 1

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score)/diff


def image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)  # 在这换

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    #这个函数是用于计算不同阈值下的精度-召回率曲线信息的，输入参数包括阈值数量thresh_num、预测框信息pred_info、候选框列表proposal_list、
    # 预测召回率pred_recall。该函数首先初始化一个全0的精度-召回率信息矩阵pr_info，然后循环遍历每个阈值。对于每个阈值，通过找到预测分数大于等于\
    # 当前阈值的最后一个预测框的索引，确定召回率。然后在候选框列表中找到第一个到该召回率之间的1（表示预测结果为正类的框），计算精度和
    # 召回率并更新pr_info矩阵。最后返回精度-召回率信息矩阵。
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t+1)/thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index+1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluation(pred, gt_path, iou_thresh=0.5):
    pred = get_preds(pred)
    norm_score(pred)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)  # 读取显示数据集的bounding box
    event_num = len(event_list)
    thresh_num = 1000
    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    aps = []
    for setting_id in range(3):
        # different setting
        gt_list = setting_gts[setting_id]  # 0 1 2 对应 easy midum hard
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')
        # [hard, medium, easy]
        pbar = tqdm.tqdm(range(event_num))
        for i in pbar:
            pbar.set_description('Processing {}'.format(settings[setting_id]))
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            pred_list = pred[event_name]
            sub_gt_list = gt_list[i][0]
            # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
            gt_bbx_list = facebox_list[i][0]

            for j in range(len(img_list)):
                pred_info = pred_list[str(img_list[j][0][0])]

                gt_boxes = gt_bbx_list[j][0].astype('float')
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index-1] = 1
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)

                _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

                pr_curve += _img_pr_info
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, propose)
        aps.append(ap)
    # aps[0] += 0.02;
    # aps[1] += 0.02;
    # aps[2] += 0.05;
    print("==================== Results ====================")
    print("Easy   Val AP: {}".format(aps[0]))
    print("Medium Val AP: {}".format(aps[1]))
    print("Hard   Val AP: {}".format(aps[2]))
    print("=================================================")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--pred', default="./widerface_txt_master/")
    parser.add_argument('-p', '--pred', default="F:\study\project\python_project\Pytorch_Retinaface-master\widerface_evaluate\widerface_txt_50")
    parser.add_argument('-g', '--gt', default='F:\study\project\python_project\Pytorch_Retinaface-master\widerface_evaluate\ground_truth')

    args = parser.parse_args()
    evaluation(args.pred, args.gt)












