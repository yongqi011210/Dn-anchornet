import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from ..builder import ROTATED_DETECTORS, ROTATED_HEADS
from .two_stage import RotatedTwoStageDetector
from torchvision import models
from mmcv.utils import build_from_cfg  # 动态构建模块


def ssim_loss(pred, target, window_size=11, data_range=1.0):

    mu_x = F.avg_pool2d(pred, window_size, 1, padding=window_size // 2, count_include_pad=False)
    mu_y = F.avg_pool2d(target, window_size, 1, padding=window_size // 2, count_include_pad=False)


    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    sigma_x_sq = F.avg_pool2d(pred * pred, window_size, 1, padding=window_size // 2, count_include_pad=False) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(target * target, window_size, 1, padding=window_size // 2,
                              count_include_pad=False) - mu_y_sq
    sigma_xy = F.avg_pool2d(pred * target, window_size, 1, padding=window_size // 2,
                            count_include_pad=False) - mu_x * mu_y


    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
                (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))

    return 1 - ssim_map.mean()

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.ssim_loss = ssim_loss

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        ssim_val = self.ssim_loss(pred, target, data_range=2.0)
        return self.alpha * l1 + (1 - self.alpha) * ssim_val



@ROTATED_DETECTORS.register_module()
class Dn_anchornet(RotatedTwoStageDetector):


    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 denoising_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(Dn_anchornet, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        # 确保 denoising_head 是一个实例化的模块
        if isinstance(denoising_head, dict):
            self.denoising_head = build_from_cfg(denoising_head, ROTATED_HEADS)
        else:
            self.denoising_head = denoising_head
        self.iter = 0
        self.denoise_loss = HybridLoss(alpha=0.7)

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, proposals=None,
                      **kwargs):

        denoised = self.denoising_head(img)

        x = self.extract_feat(denoised)

        losses = dict()

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        # 计算去噪损失
        losses['denoise_loss'] = self.denoise_loss(denoised, img) * 0.2
        self.iter += 1

        return losses

