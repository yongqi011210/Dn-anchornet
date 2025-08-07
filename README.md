# Dn-anchornet
This repository contains the official implementation of the DN-AnchorNet framework proposed in our paper, which addresses the challenges of small target detection in SAR imagery by jointly modeling denoising and anchor adaptation.

📌 Project Overview


This project is based on mmrotate and proposes a set of integrated improvements for ship detection in SAR (Synthetic Aperture Radar) images, tackling challenges such as small object detection, noise interference, and diverse object scales/orientations:

✅ Structure-aware Denoising Module (DenoisingHead): Suppresses SAR-specific noise while preserving target contours;

✅ Scale-Adaptive Anchor Generator (AdaptiveAnchorGenerator): Improves coverage for small and irregular targets;

✅ Adaptive Weighted Regression Loss (AdaptiveSmoothL1Loss): Enhances robustness for difficult regression samples.

🚀 Quick Start
pip install -U openmim
mim install mmcv-full==1.7.1
mim install mmdet==2.28.2
pip install mmrotate

python tools/train.py configs/your_config.py

📜  Citation & Acknowledgement
This work is built on top of mmrotate. Thanks to the OpenMMLab team for their great open-source contributions.

