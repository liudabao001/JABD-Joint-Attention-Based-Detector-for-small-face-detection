import numpy as np
import torch
from torchvision.ops import nms
from torch import Tensor

#-----------------------------------------------------------------#
#   将输出调整为相对于原图的大小
#-----------------------------------------------------------------#
def retinaface_correct_boxes(result, input_shape, image_shape):
    new_shape   = image_shape*np.min(input_shape/image_shape)

    offset      = (input_shape - new_shape) / 2. / input_shape
    scale       = input_shape / new_shape
    
    scale_for_boxs      = [scale[1], scale[0], scale[1], scale[0]]
    scale_for_landmarks = [scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0]]

    offset_for_boxs         = [offset[1], offset[0], offset[1],offset[0]]
    offset_for_landmarks    = [offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0]]

    result[:, :4] = (result[:, :4] - np.array(offset_for_boxs)) * np.array(scale_for_boxs)
    result[:, 5:] = (result[:, 5:] - np.array(offset_for_landmarks)) * np.array(scale_for_landmarks)

    return result

#-----------------------------#
#   中心解码，宽高解码
#-----------------------------#
def decode(loc, priors, variances):
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                    priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

#-----------------------------#
#   关键点解码
#-----------------------------#
def decode_landm(pre, priors, variances):
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms

def iou(b1,b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)
    
    area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1)
    area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1)
    
    iou = inter_area/np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
    return iou
def softer_nms(dets, confidence=None, thresh=0.01, sigma=0.5, ax=None):

    N = len(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ious = np.zeros((N, N))
    kls = np.zeros((N, N))
    for i in range(N):
        xx1 = np.maximum(x1[i], x1).cpu()
        yy1 = np.maximum(y1[i], y1)
        xx2 = np.minimum(x2[i], x2)
        yy2 = np.minimum(y2[i], y2)

        w = np.maximum(0.0, xx2 - xx1 + 1.)
        h = np.maximum(0.0, yy2 - yy1 + 1.)
        inter = w * h
        ovr = inter / (areas[i] + areas - inter)
        ious[i, :] = ovr.copy()

    i = 0
    while i < N:
        maxpos = dets[i:N, 4].argmax()
        maxpos += i
        dets[[maxpos, i]] = dets[[i, maxpos]]
        confidence[[maxpos, i]] = confidence[[i, maxpos]]
        ious[[maxpos, i]] = ious[[i, maxpos]]
        ious[:, [maxpos, i]] = ious[:, [i, maxpos]]

        ovr_bbox = np.where((ious[i, i:N] > thresh))[0] + i

        pos = i + 1
        while pos < N:
            if ious[i, pos] > 0:
                ovr = ious[i, pos]
                dets[pos, 4] *= np.exp(-(ovr * ovr) / sigma)
                if dets[pos, 4] < 0.001:
                    dets[[pos, N - 1]] = dets[[N - 1, pos]]
                    confidence[[pos, N - 1]] = confidence[[N - 1, pos]]
                    ious[[pos, N - 1]] = ious[[N - 1, pos]]
                    ious[:, [pos, N - 1]] = ious[:, [N - 1, pos]]
                    N -= 1
                    pos -= 1
            pos += 1
        i += 1
    keep = [i for i in range(N)]
    return dets[keep], keep

def nms_r(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

def diounms(boxes, scores, overlap=0.5, top_k=200, beta1=1.0):
    """Apply DIoU-NMS at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
        beta1: (float) DIoU=IoU-R_DIoU^{beta1}.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        inx1 = torch.clamp(xx1, min=x1[i])
        iny1 = torch.clamp(yy1, min=y1[i])
        inx2 = torch.clamp(xx2, max=x2[i])
        iny2 = torch.clamp(yy2, max=y2[i])
        center_x1 = (x1[i] + x2[i]) / 2
        center_y1 = (y1[i] + y2[i]) / 2
        center_x2 = (xx1 + xx2) / 2
        center_y2 = (yy1 + yy2) / 2
        d = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2
        cx1 = torch.clamp(xx1, max=x1[i])
        cy1 = torch.clamp(yy1, max=y1[i])
        cx2 = torch.clamp(xx2, min=x2[i])
        cy2 = torch.clamp(yy2, min=y2[i])
        c = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
        u= d / c
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = inx2 - inx1
        h = iny2 - iny1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union - u ** beta1 # store result in diou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

def non_max_suppression(detection, conf_thres=0.5, nms_thres=0.3):
    #------------------------------------------#
    #   找出该图片中得分大于门限函数的框。
    #   在进行重合框筛选前就
    #   进行得分的筛选可以大幅度减少框的数量。
    #------------------------------------------#
    mask        = detection[:, 4] >= conf_thres
    detection   = detection[mask]

    if len(detection) <= 0:
        return []
    
    #------------------------------------------#
    #   使用官方自带的非极大抑制会速度更快一些！
    #------------------------------------------#
    keep = nms(
        detection[:, :4],
        detection[:, 4],
        nms_thres
    )
    # dets_keep,keep = softer_nms(detection)
    best_box = detection[keep]
    
    # best_box = []
    # scores = detection[:, 4]
    # # 2、根据得分对框进行从大到小排序。
    # arg_sort = np.argsort(scores)[::-1]
    # detection = detection[arg_sort]
    #
    # while np.shape(detection)[0]>0:
    #     # 3、每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
    #     best_box.append(detection[0])
    #     if len(detection) == 1:
    #         break
    #     ious = iou(best_box[-1], detection[1:])
    #     detection = detection[1:][ious<nms_thres]
    return best_box.cpu().numpy()



def forward_traditional_nms(self, loc_data, conf_data, prior_data):
    """
    Args:
        loc_data: (tensor) Loc preds from loc layers
            Shape: [batch,num_priors*4]
        conf_data: (tensor) Shape: Conf preds from conf layers
            Shape: [batch*num_priors,num_classes]
        prior_data: (tensor) Prior boxes and variances from priorbox layers
            Shape: [1,num_priors,4]
        nms_kind: greedynms or diounms
    """

    # This funtion is no longer supported. Due to extremely time-consuming.

    num = loc_data.size(0)
    num_priors = prior_data.size(0)
    output = torch.zeros(num, self.num_classes, self.top_k, 5)
    conf_preds = conf_data.view(num, num_priors,
                                self.num_classes).transpose(2, 1)

    # Decode predictions into bboxes.
    for i in range(num):
        decoded_boxes = decode(loc_data[i], prior_data, self.variance)
        # For each class, perform nms
        conf_scores = conf_preds[i].clone()

        for cl in range(1, self.num_classes):
            c_mask = conf_scores[cl].gt(self.conf_thresh)
            scores = conf_scores[cl][c_mask]
            if scores.size(0) == 0:
                continue
            l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
            boxes = decoded_boxes[l_mask].view(-1, 4)
            # idx of highest scoring and non-overlapping boxes per class
            if self.nms_kind == "greedynms":
                ids, count = diounms(boxes, scores, self.nms_thresh, self.top_k)
            else:
                if self.nms_kind == "diounms":
                    ids, count = diounms(boxes, scores, self.nms_thresh, self.top_k, self.beta1)
                else:
                    print("use default greedy-NMS")
                    ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
            output[i, cl, :count] = \
                torch.cat((scores[ids[:count]].unsqueeze(1),
                           boxes[ids[:count]]), 1)
    flt = output.contiguous().view(num, -1, 5)
    _, idx = flt[:, :, 0].sort(1, descending=True)
    _, rank = idx.sort(1)
    flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
    return output


