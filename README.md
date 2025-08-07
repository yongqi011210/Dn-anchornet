# Dn-anchornet
This repository contains the official implementation of the DN-AnchorNet framework proposed in our paper, which addresses the challenges of small target detection in SAR imagery by jointly modeling denoising and anchor adaptation.

## 📌 Project Overview

This project is based on [mmrotate](https://github.com/open-mmlab/mmrotate) and proposes a set of integrated improvements for ship detection in SAR (Synthetic Aperture Radar) images, tackling challenges such as small object detection, noise interference, and diverse object scales/orientations:

✅ Structure-aware Denoising Module (DenoisingHead): Suppresses SAR-specific noise while preserving target contours;

✅ Scale-Adaptive Anchor Generator (AdaptiveAnchorGenerator): Improves coverage for small and irregular targets;

✅ Adaptive Weighted Regression Loss (AdaptiveSmoothL1Loss): Enhances robustness for difficult regression samples.

## 🔧 Installation Steps
We recommend setting up the environment based on the official [mmrotate installation guide](https://mmrotate.readthedocs.io/en/latest/install.html). The simplified steps for this project are:
```bash
Python == 3.8

pip install -U openmim

mim install mmcv-full==1.7.1
mim install mmdet==2.28.2

pip install mmrotate==0.34
pip install tensorboard

python tools/train.py ${CONFIG_FILE} --work-dir ${WORK_DIR}
```
## 📁 Dataset Access
📦 [RSDD-SAR Dataset](https://radars.ac.cn/web/data/getData?newsColumnId=b75bff0c-d3d6-4fb1-863d-53154109dd10)
📦 [SSDD+ Dataset](https://gitcode.com/open-source-toolkit/760a8)
## 📜 Citation & Acknowledgement
This work is built on top of mmrotate. Thanks to the OpenMMLab team for their great open-source contributions.

