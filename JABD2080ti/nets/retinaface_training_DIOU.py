import torch
import torch.nn as nn
import torch.nn.functional as F

import math
# ------------------------------#
#   获得框的左上角和右下角
# ------------------------------#
def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)


# ------------------------------#
#   获得框的中心和宽高
# ------------------------------#
def center_size(boxes):
    return torch.cat((boxes[:, 2:] + boxes[:, :2]) / 2,
                     boxes[:, 2:] - boxes[:, :2], 1)


# ----------------------------------#
#   计算所有真实框和先验框的交面积
# ----------------------------------#
def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    # ------------------------------#
    #   获得交矩形的左上角
    # ------------------------------#
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    # ------------------------------#
    #   获得交矩形的右下角
    # ------------------------------#
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    # -------------------------------------#
    #   计算先验框和所有真实框的重合面积
    # -------------------------------------#
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    # -------------------------------------#
    #   返回的inter的shape为[A,B]
    #   代表每一个真实框和先验框的交矩形
    # -------------------------------------#
    inter = intersect(box_a, box_b)
    # -------------------------------------#
    #   计算先验框和真实框各自的面积
    # -------------------------------------#
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

    union = area_a + area_b - inter
    # -------------------------------------#
    #   每一个真实框和先验框的交并比[A,B]
    # -------------------------------------#
    return inter / union  # [A,B]


def encode(matched, priors, variances):
    # 进行编码的操作
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # 中心编码
    g_cxcy /= (variances[0] * priors[:, 2:])

    # 宽高编码
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def encode_landm(matched, priors, variances):
    matched = torch.reshape(matched, (matched.size(0), 5, 2))
    priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)

    # 减去中心后除上宽高
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    g_cxcy /= (variances[0] * priors[:, :, 2:])
    g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
    return g_cxcy


def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


def DIOU(box1, box2):
    """
    diou loss
    :param box1: tensor [batch, w, h, num_anchor, 4], xywh 预测值
    :param box2: tensor [batch, w, h, num_anchor, 4], xywh 真实值
    :return: tensor [batch, w, h, num_anchor, 1]
    """
    b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
    b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
    b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
    b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

    box1_xy, box1_wh = box1[..., :2], box1[..., 2:4]
    box1_wh_half = box1_wh / 2.
    box1_mines = box1_xy - box1_wh_half
    box1_maxes = box1_xy + box1_wh_half

    box2_xy, box2_wh = box2[..., :2], box2[..., 2:4]
    box2_wh_half = box2_wh / 2.
    box2_mines = box2_xy - box2_wh_half
    box2_maxes = box2_xy + box2_wh_half

    # 求真实值和预测值所有的iou
    intersect_mines = torch.max(box1_mines, box2_mines)
    intersect_maxes = torch.min(box1_maxes, box2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mines, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box1_area = box1_wh[..., 0] * box1_wh[..., 1]
    box2_area = box2_wh[..., 0] * box2_wh[..., 1]
    union_area = box1_area + box2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area, min=1e-6)

    # 计算最小包围框的宽和高
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width

    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + 1e-16

    # 两个框中心点距离的平方
    rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
    diou = iou - rho2 / c2
    loss = 1 - diou
    return loss.sum()

    return


def DIOU_loss(pred, target):
    # 计算 DIOU 损失
    b1_x1, b1_y1, b1_x2, b1_y2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = b1_area + b2_area - inter_area

    iou = inter_area / (union_area + 1e-16)

    # 计算最小包围框的宽和高
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + 1e-16

    # 两个框中心点距离的平方
    rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4

    diou = iou - rho2 / c2
    loss = 1 - diou
    return loss.sum()

def match_iou(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    # 这个函数是用于在目标检测任务中进行匹配操作的关键步骤，通常用于计算网络预测框（先验框）和真实框之间的匹配关系。
    # 函数的输入包括一个阈值（threshold）用于判断重叠度是否足够大以被认定为正样本，真实框的坐标（truths）、
    # 先验框的坐标（priors）、方差（variances）、真实框的类别标签（labels）、真实框的关键点坐标（landms）、
    # 以及用于保存匹配结果的张量（loc_t、conf_t、landm_t）

    # ----------------------------------------------#
    #   计算所有的先验框和真实框的重合程度
    # ----------------------------------------------#
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # ----------------------------------------------#
    #   所有真实框和先验框的最好重合程度
    #   best_prior_overlap [truth_box,1]
    #   best_prior_idx [truth_box,1]
    # ----------------------------------------------#
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    # ----------------------------------------------#
    #   所有先验框和真实框的最好重合程度
    #   best_truth_overlap [1,prior]
    #   best_truth_idx [1,prior]
    # ----------------------------------------------#
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    # ----------------------------------------------#
    #   用于保证每个真实框都至少有对应的一个先验框
    # ----------------------------------------------#
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    # 对best_truth_idx内容进行设置
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    # ----------------------------------------------#
    #   获取每一个先验框对应的真实框[num_priors,4]
    # ----------------------------------------------#
    matches = truths[best_truth_idx]
    # Shape: [num_priors] 此处为每一个anchor对应的label取出来
    conf = labels[best_truth_idx]
    matches_landm = landms[best_truth_idx]

    # ----------------------------------------------#
    #   如果重合程度小于threhold则认为是背景
    # ----------------------------------------------#
    conf[best_truth_overlap < threshold] = 0
    # ----------------------------------------------#
    #   利用真实框和先验框进行编码
    #   编码后的结果就是网络应该有的预测结果
    # ----------------------------------------------#
    # loc = encode(matches, priors, variances)
    loc = matches
    landm = encode_landm(matches_landm, priors, variances)

    # ----------------------------------------------#
    #   [num_priors, 4]
    # ----------------------------------------------#
    loc_t[idx] = loc
    # ----------------------------------------------#
    #   [num_priors]
    # ----------------------------------------------#
    conf_t[idx] = conf
    # ----------------------------------------------#
    #   [num_priors, 10]
    # ----------------------------------------------#
    landm_t[idx] = landm

def match(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    # 这个函数是用于在目标检测任务中进行匹配操作的关键步骤，通常用于计算网络预测框（先验框）和真实框之间的匹配关系。
    # 函数的输入包括一个阈值（threshold）用于判断重叠度是否足够大以被认定为正样本，真实框的坐标（truths）、
    # 先验框的坐标（priors）、方差（variances）、真实框的类别标签（labels）、真实框的关键点坐标（landms）、
    # 以及用于保存匹配结果的张量（loc_t、conf_t、landm_t）

    # ----------------------------------------------#
    #   计算所有的先验框和真实框的重合程度
    # ----------------------------------------------#
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # ----------------------------------------------#
    #   所有真实框和先验框的最好重合程度
    #   best_prior_overlap [truth_box,1]
    #   best_prior_idx [truth_box,1]
    # ----------------------------------------------#
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    # ----------------------------------------------#
    #   所有先验框和真实框的最好重合程度
    #   best_truth_overlap [1,prior]
    #   best_truth_idx [1,prior]
    # ----------------------------------------------#
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    # ----------------------------------------------#
    #   用于保证每个真实框都至少有对应的一个先验框
    # ----------------------------------------------#
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    # 对best_truth_idx内容进行设置
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    # ----------------------------------------------#
    #   获取每一个先验框对应的真实框[num_priors,4]
    # ----------------------------------------------#
    matches = truths[best_truth_idx]
    # Shape: [num_priors] 此处为每一个anchor对应的label取出来
    conf = labels[best_truth_idx]
    matches_landm = landms[best_truth_idx]

    # ----------------------------------------------#
    #   如果重合程度小于threhold则认为是背景
    # ----------------------------------------------#
    conf[best_truth_overlap < threshold] = 0
    # ----------------------------------------------#
    #   利用真实框和先验框进行编码
    #   编码后的结果就是网络应该有的预测结果
    # ----------------------------------------------#
    loc = encode(matches, priors, variances)
    landm = encode_landm(matches_landm, priors, variances)

    # ----------------------------------------------#
    #   [num_priors, 4]
    # ----------------------------------------------#
    loc_t[idx] = loc
    # ----------------------------------------------#
    #   [num_priors]
    # ----------------------------------------------#
    conf_t[idx] = conf
    # ----------------------------------------------#
    #   [num_priors, 10]
    # ----------------------------------------------#
    landm_t[idx] = landm

def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    #print(boxes)
    return boxes
def bbox_overlaps_iou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows))
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
        bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
        bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    union = area1+area2-inter_area
    ious = inter_area / union
    ious = torch.clamp(ious,min=0,max = 1.0)
    if exchange:
        ious = ious.T
    return ious
def bbox_overlaps_giou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows))
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
        bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
        bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])

    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])

    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])

    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_area = outer[:, 0] * outer[:, 1]
    union = area1+area2-inter_area
    closure = outer_area

    ious = inter_area / union - (closure - union) / closure
    ious = torch.clamp(ious,min=-1.0,max = 1.0)
    if exchange:
        ious = ious.T
    return ious
def bbox_overlaps_diou(bboxes1, bboxes2):

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious,min=-1.0,max = 1.0)
    if exchange:
        dious = dious.T
    return dious

def bbox_overlaps_ciou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
    with torch.no_grad():
        S = 1 - iou
        alpha = v / (S + v)
    cious = iou - (u + alpha * v)
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    if exchange:
        cious = cious.T
    return cious
class IouLoss(nn.Module):

    def __init__(self, pred_mode='Center', size_sum=True, variances=None, losstype='Diou'):
        super(IouLoss, self).__init__()
        self.size_sum = size_sum
        self.pred_mode = pred_mode
        self.variances = variances
        self.loss = losstype

    def forward(self, loc_p, loc_t, prior_data):
        num = loc_p.shape[0]

        if self.pred_mode == 'Center':
            decoded_boxes = decode(loc_p, prior_data, self.variances)
        else:
            decoded_boxes = loc_p
        if self.loss == 'Iou':
            loss = torch.sum(1.0 - bbox_overlaps_iou(decoded_boxes, loc_t))
        else:
            if self.loss == 'Giou':
                loss = torch.sum(1.0 - bbox_overlaps_giou(decoded_boxes, loc_t))
            else:
                if self.loss == 'Diou':
                    loss = torch.sum(1.0 - bbox_overlaps_diou(decoded_boxes, loc_t))
                else:
                    loss = torch.sum(1.0 - bbox_overlaps_ciou(decoded_boxes, loc_t))

        if self.size_sum:
            loss = loss
        else:
            loss = loss / num
        return loss

class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, neg_pos, variance, cuda=True):
        super(MultiBoxLoss, self).__init__()
        # ----------------------------------------------#
        #   对于retinaface而言num_classes等于2
        # ----------------------------------------------#
        self.num_classes = num_classes
        # ----------------------------------------------#
        #   重合程度在多少以上认为该先验框可以用来预测
        # ----------------------------------------------#
        self.threshold = overlap_thresh
        # ----------------------------------------------#
        #   正负样本的比率
        # ----------------------------------------------#
        self.negpos_ratio = neg_pos
        self.variance = variance
        self.cuda = cuda
        self.gious = IouLoss(pred_mode = 'Center',size_sum=True,variances=self.variance, losstype="Diou")

    def forward(self, predictions, priors, targets):
        # --------------------------------------------------------------------#
        #   取出预测结果的三个值：框的回归信息，置信度，人脸关键点的回归信息
        # --------------------------------------------------------------------#
        loc_data, conf_data, landm_data = predictions
        # --------------------------------------------------#
        #   计算出batch_size和先验框的数量
        # --------------------------------------------------#
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # --------------------------------------------------#
        #   创建一个tensor进行处理
        # --------------------------------------------------#
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)

        for idx in range(num):
            # 获得真实框与标签
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            landms = targets[idx][:, 4:14].data

            # 获得先验框
            defaults = priors.data
            # --------------------------------------------------#
            #   利用真实框和先验框进行匹配。
            #   如果真实框和先验框的重合度较高，则认为匹配上了。
            #   该先验框用于负责检测出该真实框。
            # --------------------------------------------------#
            match_iou(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)

        # --------------------------------------------------#
        #   转化成Variable
        #   loc_t   (num, num_priors, 4)
        #   conf_t  (num, num_priors)
        #   landm_t (num, num_priors, 10)
        # --------------------------------------------------#
        zeros = torch.tensor(0)
        if self.cuda:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()
            zeros = zeros.cuda()

        # ------------------------------------------------------------------------#
        #   有人脸关键点的人脸真实框的标签为1，没有人脸关键点的人脸真实框标签为-1
        #   所以计算人脸关键点loss的时候pos1 = conf_t > zeros
        #   计算人脸框的loss的时候pos = conf_t != zeros
        # ------------------------------------------------------------------------#

        # 这是人脸关键点损失
        pos1 = conf_t > zeros  # 如果一个先验框与任何真实框的重叠度大于0，那么它就被认定为正样本，并且会被标记为对应的目标类别
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)  # 预测数据
        landm_t = landm_t[pos_idx1].view(-1, 10)  # 真实数据 在match中被重新赋值
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')  # L1损失

        # 这是人脸框损失
        pos = conf_t != zeros  # 来确定哪些先验框被认定为正样本
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)  # 将 pos 扩展为与 loc_data 具有相同形状的张量，以便
        # 与预测的框的回归信息 loc_data 进行匹配
        loc_p = loc_data[pos_idx].view(-1, 4)  # 预测数据
        loc_t = loc_t[pos_idx].view(-1, 4)  # 真实数据

        # loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')  # L1损失
        # loss_l = DIOU(loc_p, loc_t)
        giou_priors = priors.data.unsqueeze(0).expand_as(loc_data)
        loss_l = self.gious(loc_p, loc_t, giou_priors[pos_idx].view(-1, 4))

        # --------------------------------------------------#
        #   batch_conf  (num * num_priors, 2)
        #   loss_c      (num, num_priors)
        # --------------------------------------------------#
        conf_t[pos] = 1  # 将所有被认定为正样本的先验框的标签设置为1
        # 将预测的分类置信度张量 conf_data 变换为形状 (batch_size * num_priors, num_classes) 的二维张量，
        # 其中每行表示一个先验框的分类得分。
        batch_conf = conf_data.view(-1, self.num_classes)
        # 这个地方是在寻找难分类的先验框
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # 难分类的先验框不把正样本考虑进去，只考虑难分类的负样本
        loss_c[pos.view(-1, 1)] = 0
        loss_c = loss_c.view(num, -1)
        # --------------------------------------------------#
        #   loss_idx    (num, num_priors)
        #   idx_rank    (num, num_priors)
        # --------------------------------------------------#
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)  # 表示每个先验框在降序排列后的排名。
        # --------------------------------------------------#
        #   求和得到每一个图片内部有多少正样本
        #   num_pos     (num, )
        #   neg         (num, num_priors)
        # --------------------------------------------------#
        num_pos = pos.long().sum(1, keepdim=True)  # 计算每张图片内部有多少正样本
        # 限制负样本数量 self.negpos_ratio 是固定的负正样本比例
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)  # 计算负样本的数量
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # --------------------------------------------------#
        #   求和得到每一个图片内部有多少正样本
        #   pos_idx   (num, num_priors, num_classes)
        #   neg_idx   (num, num_priors, num_classes)
        # --------------------------------------------------#
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        # 选取出用于训练的正样本与负样本，计算loss
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)  # 获取用于训练的正样本和负样本的索引
        targets_weighted = conf_t[(pos + neg).gt(0)]  # 真实数据
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')  # Cross损失

        N = max(num_pos.data.sum().float(), 1)  # 计算了所有图片内部的正样本数量的总和，并将其转换为浮点型。
        # 如果正样本数量总和小于1，则将其设置为1，以避免除零错误
        loss_l /= N  # 损失归一化
        loss_c /= N

        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)  # 关键点正样本数量综合
        loss_landm /= N1
        return loss_l, loss_c, loss_landm


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)
