# Dn-anchornet
This repository contains the official implementation of the DN-AnchorNet framework proposed in our paper, which addresses the challenges of small target detection in SAR imagery by jointly modeling denoising and anchor adaptation.

📌 项目简介 | Project Overview
本项目基于开源目标检测框架 mmrotate，针对合成孔径雷达（SAR）图像中舰船目标检测中存在的 小目标难检测、图像噪声干扰、尺度与形态变化复杂 等挑战，提出了一个集成式改进方法：

✅ 结构感知图像去噪模块（DenoisingHead）：在保留目标边缘结构的同时抑制 SAR 雪花噪声；

✅ 尺度自适应锚框机制（AdaptiveAnchorGenerator）：提升小目标的检测覆盖率；

✅ 自适应加权回归损失函数（AdaptiveSmoothL1Loss）：在边界框回归中提升对难样本的鲁棒性。

This project is based on mmrotate and proposes a set of integrated improvements for ship detection in SAR (Synthetic Aperture Radar) images, tackling challenges such as small object detection, noise interference, and diverse object scales/orientations:

✅ Structure-aware Denoising Module (DenoisingHead): Suppresses SAR-specific noise while preserving target contours;

✅ Scale-Adaptive Anchor Generator (AdaptiveAnchorGenerator): Improves coverage for small and irregular targets;

✅ Adaptive Weighted Regression Loss (AdaptiveSmoothL1Loss): Enhances robustness for difficult regression samples.
