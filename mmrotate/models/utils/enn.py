# mmrotate/models/necks/utils.py

import e2cnn.nn as enn
from e2cnn import gspaces

# 定义全局的 gspace
N = 8
gspace = gspaces.Rot2dOnR2(N=N)
# print(f"gspace.regular_repr.size: {gspace.regular_repr.size}")  # 应输出 8


def build_enn_feature(planes):
    """
    构建 Enn 特征映射，确保 planes 和 gspace.regular_repr.size 匹配。

    Args:
        planes (int): 通道数，应为 gspace.regular_repr.size 的倍数。

    Returns:
        enn.FieldType: 构建的 FieldType 对象。
    """
    # print(f"Building enn feature with {planes} channels.")
    assert planes % gspace.regular_repr.size == 0, (
        f"planes ({planes}) must be divisible by gspace.regular_repr.size ({gspace.regular_repr.size})"
    )
    num_reprs = planes // gspace.regular_repr.size
    field_type = enn.FieldType(gspace, [gspace.regular_repr] * num_reprs)
    # print(f"FieldType.size: {field_type.size}")
    return field_type


def build_enn_divide_feature(planes):
    """
    构建 Enn 特征映射，planes 由 N 分割。

    Args:
        planes (int): 通道数，应为 N 的倍数。

    Returns:
        enn.FieldType: 构建的 FieldType 对象。
    """
    assert gspace.fibergroup.order() > 0, "gspace.fibergroup.order() must be greater than 0."
    N_order = gspace.fibergroup.order()
    planes_divided = planes // N_order
    # print(f"Building enn divide feature with {planes_divided} representations (planes: {planes}).")
    return enn.FieldType(gspace, [gspace.regular_repr] * planes_divided)


def build_enn_trivial_feature(planes):
    """
    构建 Enn trivial 特征映射。

    Args:
        planes (int): 通道数。

    Returns:
        enn.FieldType: 构建的 trivial FieldType 对象。
    """
    # print(f"Building enn trivial feature with {planes} channels.")
    return enn.FieldType(gspace, planes * [gspace.trivial_repr])


def build_enn_norm_layer(num_features, postfix='', norm_cfg=None):
    """
    构建 Enn 归一化层。

    Args:
        num_features (int): 归一化层的通道数。
        postfix (str): 后缀，用于命名层。

    Returns:
        tuple: (层名, 归一化层对象)
    """
    in_type = build_enn_divide_feature(num_features)  # 使用 build_enn_divide_feature
    return 'bn' + str(postfix), enn.InnerBatchNorm(in_type)


def ennConv(inplanes, outplanes, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1):
    """
    构建 Enn 卷积层。

    Args:
        inplanes (int): 输入通道数，应为 gspace.regular_repr.size 的倍数。
        outplanes (int): 输出通道数，应为 gspace.regular_repr.size 的倍数。
        kernel_size (int): 卷积核大小。
        stride (int): 步幅。
        padding (int): 填充。
        groups (int): 分组数。
        bias (bool): 是否包含偏置。
        dilation (int): 膨胀。

    Returns:
        enn.R2Conv: 构建的 R2Conv 层。
    """
    in_type = build_enn_divide_feature(inplanes)
    out_type = build_enn_divide_feature(outplanes)
    return enn.R2Conv(
        in_type,
        out_type,
        kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
        sigma=None,
        frequencies_cutoff=lambda r: 3 * r,
    )


def ennTrivialConv(inplanes, outplanes, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1):
    """
    构建 Enn trivial 卷积层。

    Args:
        inplanes (int): 输入通道数。
        outplanes (int): 输出通道数，应为 gspace.regular_repr.size 的倍数。
        kernel_size (int): 卷积核大小。
        stride (int): 步幅。
        padding (int): 填充。
        groups (int): 分组数。
        bias (bool): 是否包含偏置。
        dilation (int): 膨胀。

    Returns:
        enn.R2Conv: 构建的 R2Conv 层。
    """
    in_type = build_enn_trivial_feature(inplanes)
    out_type = build_enn_divide_feature(outplanes)
    return enn.R2Conv(
        in_type,
        out_type,
        kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
        sigma=None,
        frequencies_cutoff=lambda r: 3 * r,
    )


def ennReLU(inplanes):
    """
    构建 Enn ReLU 激活层。

    Args:
        inplanes (int): 输入通道数，应为 gspace.regular_repr.size 的倍数。

    Returns:
        enn.ReLU: 构建的 ReLU 层。
    """
    in_type = build_enn_divide_feature(inplanes)
    return enn.ReLU(in_type, inplace=False)


def ennAvgPool(inplanes, kernel_size=1, stride=None, padding=0, ceil_mode=False):
    """
    构建 Enn 平均池化层。

    Args:
        inplanes (int): 输入通道数，应为 gspace.regular_repr.size 的倍数。
        kernel_size (int): 池化核大小。
        stride (int): 步幅。
        padding (int): 填充。
        ceil_mode (bool): 是否使用 ceil 模式。

    Returns:
        enn.PointwiseAvgPool: 构建的平均池化层。
    """
    in_type = build_enn_divide_feature(inplanes)
    return enn.PointwiseAvgPool(
        in_type,
        kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode)


def ennMaxPool(inplanes, kernel_size, stride=1, padding=0):
    """
    构建 Enn 最大池化层。

    Args:
        inplanes (int): 输入通道数，应为 gspace.regular_repr.size 的倍数。
        kernel_size (int): 池化核大小。
        stride (int): 步幅。
        padding (int): 填充。

    Returns:
        enn.PointwiseMaxPool: 构建的最大池化层。
    """
    in_type = build_enn_divide_feature(inplanes)
    return enn.PointwiseMaxPool(
        in_type, kernel_size=kernel_size, stride=stride, padding=padding)


def ennInterpolate(inplanes, scale_factor, mode='nearest', align_corners=False):
    """
    构建 Enn 上采样层。

    Args:
        inplanes (int): 输入通道数，应为 gspace.regular_repr.size 的倍数。
        scale_factor (float): 上采样倍数。
        mode (str): 上采样模式。
        align_corners (bool): 是否对齐角落。

    Returns:
        enn.R2Upsampling: 构建的上采样层。
    """
    in_type = build_enn_divide_feature(inplanes)
    return enn.R2Upsampling(
        in_type, scale_factor, mode=mode, align_corners=align_corners)
