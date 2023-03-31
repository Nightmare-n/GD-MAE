import torch

from ...ops.iou3d_nms import iou3d_nms_utils


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
        )
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]


def multi_class_agnostic_nms(box_scores, box_ious, box_labels, box_preds, nms_config):
    iou_rectifier = box_scores.new_tensor(nms_config.IOU_RECTIFIER)
    iou_rectifier = iou_rectifier[box_labels]
    rect_scores = torch.pow(box_scores, 1 - iou_rectifier) * torch.pow(box_ious, iou_rectifier)
    selected = []
    for cls in range(len(nms_config.NMS_THRESH)):
        class_mask = box_labels == cls
        if class_mask.sum() > 0:
            src_idx = class_mask.nonzero(as_tuple=True)[0]
            box_scores_nms, indices = torch.topk(rect_scores[class_mask], k=min(nms_config.NMS_PRE_MAXSIZE[cls], rect_scores[class_mask].shape[0]))
            boxes_for_nms = box_preds[class_mask][indices]
            keep_idx, _ = iou3d_nms_utils.nms_gpu(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH[cls]
            )
            per_selected = src_idx[indices[keep_idx[:nms_config.NMS_POST_MAXSIZE[cls]]]]
            selected.append(per_selected)
    if len(selected) > 0:
        selected = torch.cat(selected, dim=0)
    return selected, rect_scores[selected]


def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]
            cur_box_preds = box_preds

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
            )
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes
