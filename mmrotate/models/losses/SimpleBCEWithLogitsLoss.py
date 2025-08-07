import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ROTATED_LOSSES
from mmdet.models import weight_reduce_loss, weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def adaptive_smooth_l1_loss(pred, target, beta=1.0, pos_weight=1.0, neg_weight=0.0, eps=1e-6):
    """含调试打印的完整损失函数"""
    if target.numel() == 0:
        # print("[DEBUG] Empty targets, return zero loss")
        return pred.sum() * 0

    # 打印输入维度信息
    # print(f"[DEBUG] Input shapes - pred: {pred.shape}, target: {target.shape}")
    assert pred.shape == target.shape, f"预测维度{pred.shape}与目标{target.shape}不匹配"

    # 处理beta参数
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, device=pred.device, dtype=pred.dtype)
    # print(f"[DEBUG] Initial beta shape: {beta.shape}")

    # 自动扩展beta维度
    param_dim = pred.size(-1)
    if beta.dim() == 0:  # 标量扩展
        beta = beta.repeat(param_dim)
    elif beta.size(-1) == 1:  # 单值扩展
        beta = beta.repeat(param_dim)
    # print(f"[DEBUG] After expand beta shape: {beta.shape}")

    # 维度对齐 (B, N, C) -> (1,1,C)
    beta = beta.view(*([1] * (pred.dim() - 1)), param_dim)
    # print(f"[DEBUG] Final beta shape: {beta.shape}")

    # 计算差异
    diff = torch.abs(pred - target)
    # print(f"[DEBUG] Diff shape: {diff.shape}")

    # 核心计算
    smooth_l1_loss = torch.where(
        diff < beta,
        0.5 * diff.pow(2) / beta,
        diff - 0.5 * beta
    )

    # 生成正负样本掩码
    pos_mask = (torch.abs(target).sum(dim=-1, keepdim=True) > 0).float()
    neg_mask = 1.0 - pos_mask
    # print(f"[DEBUG] pos_mask shape: {pos_mask.shape}")

    # 应用权重
    weighted_loss = smooth_l1_loss * (pos_weight * pos_mask + neg_weight * neg_mask)
    # print(f"[DEBUG] Final loss shape: {weighted_loss.shape}")

    return weighted_loss


@ROTATED_LOSSES.register_module()
class AdaptiveSmoothL1Loss(nn.Module):
    """含动态beta调整的损失类"""

    def __init__(self,
                 beta=1.0,
                 pos_weight=1.0,
                 neg_weight=0.0,
                 reduction='mean',
                 loss_weight=1.0,
                 dynamic_beta=False):
        super().__init__()

        # print(f"[INIT] Initializing with beta: {beta}")

        # 处理beta参数类型
        if isinstance(beta, (list, tuple)):
            beta = torch.tensor(beta, dtype=torch.float32)
        elif isinstance(beta, float):
            beta = torch.tensor([beta])

        # 注册为buffer确保设备同步
        self.register_buffer('base_beta', beta)
        # print(f"[INIT] Registered base_beta shape: {self.base_beta.shape}")

        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.dynamic_beta = dynamic_beta

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ** kwargs):
        # 输入维度检查
        # print(f"[FORWARD] Input shapes - pred: {pred.shape}, target: {target.shape}")
        assert pred.shape == target.shape, "预测与目标形状不一致"

        # 动态beta调整
        current_beta = self.base_beta.clone().to(pred.device)
        if self.dynamic_beta and self.training:
            with torch.no_grad():
                # 按参数维度计算均值
                diff_mean = torch.abs(pred.detach() - target).mean(dim=0)  # 关键修改点
                current_beta = 0.9 * current_beta + 0.1 * diff_mean
                # print(f"[DYNAMIC] Updated beta: {current_beta}")

        # 维度扩展检查
        # print(f"[FORWARD] Current beta shape before expand: {current_beta.shape}")
        param_dim = pred.size(-1)
        if current_beta.size(-1) != param_dim:
            raise RuntimeError(
                f"Beta维度{current_beta.shape}不匹配参数维度{param_dim}，"
                f"请检查beta初始化值是否为{param_dim}个参数"
            )

        # 扩展维度匹配输入
        current_beta = current_beta.view(*([1] * (pred.dim() - 1)), param_dim)
        # print(f"[FORWARD] Beta after expansion: {current_beta.shape}")

        # 计算损失
        loss_bbox = self.loss_weight * adaptive_smooth_l1_loss(
            pred,
            target,
            weight=weight,
            beta=current_beta,
            pos_weight=self.pos_weight,
            neg_weight=self.neg_weight,
            reduction=reduction_override if reduction_override else self.reduction,
            avg_factor=avg_factor,
            ** kwargs)

        return loss_bbox