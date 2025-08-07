# mmrotate/core/evaluation/far.py
import numpy as np


def calculate_far(det_results, annotations, iou_thr=0.5):
    """计算虚警率(FAR)"""
    # 延迟导入以避免循环依赖
    from mmrotate.core.bbox import rbbox_overlaps

    fp_total = 0
    tp_total = 0
    img_count = len(annotations)

    for dets, ann in zip(det_results, annotations):
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        gt_ignore = ann.get('bboxes_ignore', None)

        # 过滤低置信度检测
        valid_dets = [d for d in dets if d[5] > 0.3]  # 置信度阈值0.3
        if not valid_dets:
            continue

        det_bboxes = np.array([d[:5] for d in valid_dets])
        det_scores = np.array([d[5] for d in valid_dets])

        # 计算IoU
        if gt_bboxes.size > 0:
            ious = rbbox_overlaps(
                det_bboxes.astype(np.float32),
                gt_bboxes.astype(np.float32),
                mode='iou')
        else:
            ious = np.zeros((len(det_bboxes), 0))

        # 匹配检测和真实框
        matched = set()
        for det_idx in range(len(det_bboxes)):
            max_iou = 0
            max_gt_idx = -1

            for gt_idx in range(len(gt_bboxes)):
                if ious[det_idx, gt_idx] > max_iou:
                    max_iou = ious[det_idx, gt_idx]
                    max_gt_idx = gt_idx

            if max_iou >= iou_thr:
                matched.add(max_gt_idx)
                tp_total += 1
            else:
                fp_total += 1

    # 计算虚警率
    if tp_total + fp_total == 0:
        far = 0.0
    else:
        far = fp_total / (tp_total + fp_total) * 100  # 百分比

    # 每张图像的虚警数
    fppi = fp_total / img_count if img_count > 0 else 0

    return far, fppi, fp_total
