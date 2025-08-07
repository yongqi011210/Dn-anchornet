import torch
import matplotlib.pyplot as plt
from mmcv.utils import to_2tuple
from mmdet.core.anchor import AnchorGenerator
from .builder import ROTATED_ANCHOR_GENERATORS


@ROTATED_ANCHOR_GENERATORS.register_module()
class AdaptiveAnchorGenerator(AnchorGenerator):
    """带调试功能的自适应锚框生成器

    Args:
        strides (list[int|tuple]): 每个特征层的步长（像素单位）
        ratios (list[float]): 锚框宽高比列表，例如[0.5, 1, 2]
        scales (list[float], optional): 手动指定的缩放比例，默认为None时自动生成
        base_sizes (list[float], optional): 每个层级的基础尺寸
        num_scales (int): 自动生成的缩放级别数量
        adaptive (bool): 是否启用自适应模式
        debug (bool): 是否启用调试模式
    """

    def __init__(self,
                 strides=[8, 16, 32, 64, 128],
                 ratios=[0.2, 0.5, 1, 2, 5],
                 scales=None,
                 base_sizes=None,
                 num_scales=5,
                 adaptive=True,
                 debug=False):
        super().__init__(strides, scales=scales, ratios=ratios, base_sizes=base_sizes)

        # 参数验证
        assert len(strides) > 0, "至少需要1个特征层"
        assert len(ratios) >= 1, "至少需要1个宽高比"

        self.strides = [to_2tuple(s) for s in strides]
        self.num_scales = num_scales
        self.adaptive = adaptive
        self.debug = debug
        self.anchor_cache = {}  # 存储调试数据

        # if debug:
        #     print(f"舰船检测锚框配置：\n"
        #           f"特征层数：{len(strides)}\n"
        #           f"基础步长：{strides}\n"
        #           f"宽高比：{ratios}\n"
        #           f"缩放级数：{num_scales}")

    def _generate_base_anchors(self, stride, dtype=torch.float32, device='cuda'):
        """生成基础锚框（核心算法）"""
        stride_x, stride_y = stride

        # 自动生成等比缩放系数
        if self.scales is None:
            scales = torch.linspace(0.8, 1.2, self.num_scales, dtype=dtype, device=device)
        else:
            scales = torch.tensor(self.scales, dtype=dtype, device=device)

        ratios = torch.tensor(self.ratios, dtype=dtype, device=device)

        # 生成组合矩阵
        scale_ratio = torch.meshgrid(scales, ratios, indexing='ij')
        combined = torch.stack(scale_ratio, dim=-1).reshape(-1, 2)

        # 计算锚框尺寸（考虑雷达目标特性）
        base_area = (stride_x * 8) * (stride_y * 8)  # 经验公式
        ws = torch.sqrt(base_area * combined[:, 1]) * combined[:, 0]
        hs = ws / combined[:, 1]

        # 生成坐标形式 [x1,y1,x2,y2]
        return torch.stack([-ws / 2, -hs / 2, ws / 2, hs / 2], dim=1)

    def single_level_grid_anchors(self, featmap_size, level_idx, dtype=torch.float32, device='cuda'):
        """生成单层锚框"""
        base_anchors = self._generate_base_anchors(self.strides[level_idx], dtype, device)

        # 生成网格偏移量
        feat_h, feat_w = featmap_size
        stride_x, stride_y = self.strides[level_idx]

        shift_x = torch.arange(0, feat_w, device=device, dtype=dtype) * stride_x + stride_x // 2
        shift_y = torch.arange(0, feat_h, device=device, dtype=dtype) * stride_y + stride_y // 2

        # 生成坐标网格
        shift_xx, shift_yy = torch.meshgrid(shift_x, shift_y, indexing='ij')
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)

        # 广播相加生成最终锚框
        all_anchors = base_anchors.view(1, -1, 4) + shifts.view(-1, 1, 4)
        all_anchors = all_anchors.view(-1, 4)

        if self.debug:
            self._cache_anchor_info(level_idx, base_anchors, all_anchors, featmap_size)

        return all_anchors

    def _cache_anchor_info(self, level, base, all_anchors, featmap_size):
        """存储调试信息"""
        self.anchor_cache[level] = {
            'base_anchors': base.cpu(),
            'all_anchors': all_anchors.cpu(),
            'coverage_ratio': self._calc_coverage(all_anchors, featmap_size),
            'size_distribution': self._analyze_sizes(all_anchors)
        }

    def _calc_coverage(self, anchors, featmap_size):
        """计算特征图覆盖质量"""
        img_w = 512  # 根据输入尺寸设定
        img_h = 512
        coverage_x = (anchors[:, 2].max() - anchors[:, 0].min()) / img_w
        coverage_y = (anchors[:, 3].max() - anchors[:, 1].min()) / img_h
        return (coverage_x.item(), coverage_y.item())

    def _analyze_sizes(self, anchors):
        """分析锚框尺寸分布"""
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        return {
            'widths': torch.std_mean(widths),
            'heights': torch.std_mean(heights),
            'aspect_ratios': (widths / heights).unique()
        }

    def visualize_anchor_distribution(self, level=0, figsize=(12, 6)):
        """可视化锚框分布"""
        if level not in self.anchor_cache:
            raise ValueError(f"层级 {level} 无调试数据")

        info = self.anchor_cache[level]
        plt.figure(figsize=figsize)

        # 尺寸分布直方图
        plt.subplot(1, 2, 1)
        widths = info['all_anchors'][:, 2] - info['all_anchors'][:, 0]
        heights = info['all_anchors'][:, 3] - info['all_anchors'][:, 1]
        plt.hist(widths.numpy(), bins=20, alpha=0.5, label='宽度')
        plt.hist(heights.numpy(), bins=20, alpha=0.5, label='高度')
        plt.title(f'尺寸分布 (层级 {level})')
        plt.xlabel('像素值')
        plt.ylabel('频次')
        plt.legend()

        # 宽高比分布
        plt.subplot(1, 2, 2)
        ratios = widths / heights
        plt.scatter(widths.numpy(), heights.numpy(), s=2, alpha=0.5)
        plt.title(f'宽高比分布（均值:{ratios.mean():.2f}）')
        plt.xlabel('宽度')
        plt.ylabel('高度')
        plt.grid(True)

        plt.tight_layout()
        plt.show()