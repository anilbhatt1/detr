# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area

"""
Let us say we are passing out_bbox of [200, 4] and tgt_bbox of [14, 4] to box_cxcywh_to_xyxy.
x.unbind(-1) will return tuples of len 200 and 14 respectively.
Each tuple will have (x_c, y_c, w, h) ie unbind of out_bbox will give a tuple of 200 bbox coordinates whereas tgt_bbox of 14 bbox coordinates.
b will be a list of length 4 each having 200 elements as shown for out_bbox-> [[lt_x1, lt_x2, ...., lt_x200],
                                                                               [lt_y1, lt_y2, ...., lt_y200],
                                                                               [rb_x1, rb_x2, ...., rb_x200],
                                                                               [rb_y1, rb_y2, ...., rb_y200] ]  lt -> left top, rb -> right bottom
Similarly b will be a list of length 4 each having 14 elements as shown for tgt_bbox->  [[lt_x1, lt_x2, ...., lt_x14],
                                                                                         [lt_y1, lt_y2, ...., lt_y14],
                                                                                         [rb_x1, rb_x2, ...., rb_x14],
                                                                                         [rb_y1, rb_y2, ...., rb_y14]] 
Then b will be stacked torch.stack(b, dim=-1), for out_bbox it will be [200, 4] like [lt_x1, lt_y1, rb_x1, rb_y1]
                                                                                      .. 
                                                                                     [lt_x200, lt_y200, rb_x200, rb_y200] 
                                               for tgt_bbox it will be [14, 4] like [lt_x1, lt_y1, rb_x1, rb_y1]
                                                                                      .. 
                                                                                     [lt_x14, lt_y14, rb_x14, rb_y14]                                                                                      
"""
def box_cxcywh_to_xyxy(x):        
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
"""
torchvision.ops.boxes.box_area -> Computes the area of a set of bounding boxes, which are specified by their (x1, y1, x2, y2) coordinates.
boxes1 -> Comes from out_bbox of [200, 4] so area1 -> [200]
boxes2 -> Comes from tgt_bbox of [14, 4] so area2 -> [14]
Calculation of lt (left-top coordinate) and rb (right-bottom coordinate) are as follows:

boxes1[:, None, :2] -> ([200, 1, 2])
boxes1: tensor([[-2.3103e-03,  2.7565e-01,  7.7548e-02,  9.8273e-01],     boxes1[:, None, :2] : tensor([[[-2.3103e-03,  2.7565e-01]],
                [ 2.3402e-01,  1.3517e-01,  7.7009e-01,  3.8345e-01]   ->                               [[ 2.3402e-01,  1.3517e-01]],
                ..                                                                                      ..
                [ 3.4362e-02,  6.8599e-02,  7.7951e-01,  3.3515e-01]])                                  [[ 3.4362e-02,  6.8599e-02]]])
                
boxes2[:, :2] -> ([14, 2])
boxes2: tensor([[0.1910, 0.2559, 0.7207, 0.6136]        boxes2[:, :2] : tensor([[0.1910, 0.2559]
                [0.3477, 0.3764, 1.0000, 0.9509],                               [0.3477, 0.3764],
                ..                                   ->                          ..
                [0.0366, 0.0697, 0.4580, 0.7051]])                              [0.0366, 0.0697]])
                
lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) -> ([200, 14, 2])
Each row of boxes1[:, None, :2] is compared against 14 rows of boxes2[:, :2] and max is fetched. Thus each row of boxes1[:, None, :2] will produce 14 rows
getting a shape of ([200, 14, 2]).
lt -> tensor([[[0.1910, 0.2757],  -> First row [-2.3103e-03,  2.7565e-01] compared against tensor([[0.1910, 0.2559],
               [0.3477, 0.3764],                (-0.0023103)  (0.27565)                            [0.3477, 0.3764],
               ..                                                                                  ..                 
               [0.0366, 0.2757]]                                                                   [0.0366, 0.0697]])
               .
               .               
              [[0.1910, 0.2559],  -> 200th row [ 3.4362e-02,  6.8599e-02] compared against  tensor([[0.1910, 0.2559]
               [0.3477, 0.3764],                 (0.034362)   (0.068599)                            [0.3477, 0.3764],
               ..                                                                                   ..
               [0.0366, 0.0697]]])                                                                  [0.0366, 0.0697]])
Similarly rb is also calculated but using torch.min of coordinates.

Next step is to calculate wh (width & height) from lt & rb.
wh = (rb - lt).clamp(min=0) -> [200, 14, 2] where difference in x-coordinates give w & difference in y-coordinates give h.
wh[:, :, 0] -> width, wh[:, :, 1] -> height 
So wh[:, :, 0] * wh[:, :, 1] -> intersection area of [200, 14]

Next step is to calculate union
union = area1[:, None] + area2 - inter
area1[:, None] -> [200, 1] + area2 -> [14] will give tensor of [200, 14] which is subtracted - from inter [200, 14] to give union [200, 14]

Next iou = inter -> [200, 14] / union -> [200, 14] is calculated
iou -> torch.Size([200, 14]
"""
def box_iou(boxes1, boxes2, print_flag):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2, print_flag):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    '''
    For GIOU, iou & union that we got from box_iou is used whereas area used in equation is calculated differently from box_iou.
    Difference is that lt is calculated with torch.min and rb is calculated with torch.max.
    Rest of area calculation is same as that we seen in box_iou.
    GIOU will be of shape [200, 14] for the example batch we are considering.
    '''
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2, print_flag)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]
    
    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
