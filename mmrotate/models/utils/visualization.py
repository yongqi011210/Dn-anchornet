# mmrotate/utils/visualization.py

import matplotlib.pyplot as plt
import torch
import os


def visualize_feature_maps(feature, title, num_channels=8):
    """
    可视化特征图。

    Args:
        feature (Tensor): 特征图，形状为 (batch_size, channels, height, width)。
        title (str): 标题。
        num_channels (int): 要可视化的通道数（默认前8个）。
    """
    if not isinstance(feature, torch.Tensor):
        raise ValueError("Feature must be a torch.Tensor")

    batch_size, channels, height, width = feature.shape
    num_plots = min(channels, num_channels)

    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 2, 2))
    fig.suptitle(title)

    for i in range(num_plots):
        ax = axes[i] if num_plots > 1 else axes
        fmap = feature[0, i].detach().cpu().numpy()
        ax.imshow(fmap, cmap='viridis')
        ax.axis('off')

    plt.show()


def save_feature_maps(feature, title, save_dir, num_channels=8):
    """
    保存特征图为图像文件。

    Args:
        feature (Tensor): 特征图，形状为 (batch_size, channels, height, width)。
        title (str): 标题，用于命名文件。
        save_dir (str): 保存目录。
        num_channels (int): 要保存的通道数（默认前8个）。
    """
    if not isinstance(feature, torch.Tensor):
        raise ValueError("Feature must be a torch.Tensor")

    batch_size, channels, height, width = feature.shape
    num_plots = min(channels, num_channels)

    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_plots):
        fmap = feature[0, i].detach().cpu().numpy()
        plt.imshow(fmap, cmap='viridis')
        plt.axis('off')
        plt.title(f"{title} Channel {i}")
        plt.savefig(os.path.join(save_dir, f"{title}_channel_{i}.png"))
        plt.close()
