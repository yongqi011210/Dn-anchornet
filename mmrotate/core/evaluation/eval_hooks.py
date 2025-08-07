import os
import time
import numpy as np
import mmcv
import logging
import sys
from mmcv.runner import HOOKS, Hook
from mmrotate.core.bbox import rbbox_overlaps
from datetime import datetime


@HOOKS.register_module()
class FAREvaluationHook(Hook):
    """虚警率评估钩子 - 针对DOTA数据集优化"""

    def __init__(self,
                 iou_thr=0.5,
                 score_thr=0.3,
                 interval=1,
                 output_dir=None,
                 log_file=None):
        self.iou_thr = iou_thr
        self.score_thr = score_thr
        self.interval = interval
        self.output_dir = output_dir
        self.log_file = log_file

        # 结果存储
        self.detections = []
        self.annotations = []
        self.images = []
        self.logger = None
        self.file_handler = None
        self.start_time = None

    def before_run(self, runner):
        """运行前初始化"""
        # 创建输出目录
        if self.output_dir:
            abs_output_dir = os.path.abspath(self.output_dir)
            mmcv.mkdir_or_exist(abs_output_dir)

        # 初始化日志记录器
        self.logger = runner.logger

        # 创建单独的日志文件
        if self.log_file:
            log_path = os.path.abspath(self.log_file)
            log_dir = os.path.dirname(log_path)
            os.makedirs(log_dir, exist_ok=True)

            # 创建文件处理器 - 使用标准logging模块，指定UTF-8编码
            self.file_handler = logging.FileHandler(log_path, encoding='utf-8')

            # 获取当前日志格式
            if hasattr(self.logger, 'handlers') and self.logger.handlers:
                formatter = self.logger.handlers[0].formatter
                self.file_handler.setFormatter(formatter)
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                self.file_handler.setFormatter(formatter)

            # 添加到日志记录器
            self.logger.addHandler(self.file_handler)

            self.logger.info('FAR log will be saved to: %s', log_path)

        # 记录初始化信息
        self.logger.info('================================================')
        self.logger.info('          FAR Evaluation Hook Initialized       ')
        self.logger.info('------------------------------------------------')
        self.logger.info('IoU threshold:    %s', self.iou_thr)
        self.logger.info('Score threshold:  %s', self.score_thr)
        self.logger.info('Output directory: %s', str(self.output_dir))
        self.logger.info('Log file:         %s', str(self.log_file))
        self.logger.info('================================================')

    def before_val_epoch(self, runner):
        """验证周期前重置收集器"""
        self.logger.info('Resetting FAR evaluation collector for epoch %s', runner.epoch + 1)
        self.detections = []
        self.annotations = []
        self.images = []

        # 记录验证开始时间
        self.start_time = time.time()
        self.logger.info('Starting validation epoch %s...', runner.epoch + 1)

    def after_val_iter(self, runner):
        """每次验证迭代后收集数据"""
        if not self.every_n_inner_iters(runner, self.interval):
            return

        # 添加调试信息
        self.logger.debug('Collecting validation data for iter %s', runner.iter)

        # 获取当前批次数据
        outputs = runner.outputs
        data_batch = runner.data_batch

        # 处理每张图像的输出
        for i in range(len(outputs)):
            # 获取检测结果
            try:
                if isinstance(outputs[i], dict):
                    dets = outputs[i]['det_bboxes'].cpu().numpy()
                else:
                    dets = outputs[i].cpu().numpy()
            except Exception as e:
                self.logger.error('Error getting detections: %s', str(e))
                dets = np.empty((0, 6))

            # 获取标注信息
            try:
                gt_bboxes = data_batch['gt_bboxes'][i]
                if hasattr(gt_bboxes, 'cpu'):
                    gt_bboxes = gt_bboxes.cpu().numpy()
                else:
                    gt_bboxes = np.array(gt_bboxes, dtype=np.float32)

                if 'gt_bboxes_ignore' in data_batch:
                    gt_bboxes_ignore = data_batch['gt_bboxes_ignore'][i]
                    if hasattr(gt_bboxes_ignore, 'cpu'):
                        gt_bboxes_ignore = gt_bboxes_ignore.cpu().numpy()
                    else:
                        gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
                else:
                    gt_bboxes_ignore = np.zeros((0, 5), dtype=np.float32)
            except Exception as e:
                self.logger.error('Error getting annotations: %s', str(e))
                gt_bboxes = np.empty((0, 5))
                gt_bboxes_ignore = np.empty((0, 5))

            ann = {
                'bboxes': gt_bboxes,
                'bboxes_ignore': gt_bboxes_ignore
            }

            # 存储结果
            self.detections.append(dets)
            self.annotations.append(ann)

            # 如果需要可视化，存储图像路径
            if self.output_dir:
                try:
                    img_meta = data_batch['img_metas'][i].data[0]
                    img = img_meta['filename']
                    self.images.append(img)
                except Exception as e:
                    self.logger.error('Error getting image path: %s', str(e))
                    self.images.append('unknown')
        # 记录收集状态
        self.logger.debug('Collected %d detections for %d images',
                          len(self.detections), len(self.annotations))

    def after_val_epoch(self, runner):
        """验证周期后计算并记录虚警率"""
        if not self.start_time:
            self.logger.warning('Validation start time not set, skipping FAR calculation')
            return

        # 计算验证耗时
        elapsed = time.time() - self.start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        # 记录收集状态
        self.logger.info('Finished validation for epoch %s', runner.epoch + 1)
        self.logger.info('Collected detections for %d images', len(self.annotations))

        if not self.detections:
            self.logger.warning('No detections collected for FAR evaluation')
            return

        # 计算虚警率
        try:
            far, fppi, fp_total, tp_total = self.calculate_far()
        except Exception as e:
            self.logger.error('Error calculating FAR: %s', str(e))
            return

        # 记录到日志
        runner.log_buffer.output['far'] = far
        runner.log_buffer.output['fppi'] = fppi
        runner.log_buffer.output['fp_total'] = fp_total
        runner.log_buffer.output['tp_total'] = tp_total

        # 格式化日志消息
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = (
            f"\n{'=' * 80}\n"
            f"FAR Evaluation Results - Epoch {runner.epoch + 1}\n"
            f"Validation Time: {elapsed_str}\n"
            f"Timestamp: {timestamp}\n"
            f"{'-' * 80}\n"
            f"FAR:     {far:.2f}%    (False Alarm Rate)\n"
            f"FPPI:    {fppi:.2f}    (False Positives Per Image)\n"
            f"FP:      {fp_total}     (Total False Positives)\n"
            f"TP:      {tp_total}     (Total True Positives)\n"
            f"Images:  {len(self.annotations)}  (Processed)\n"
            f"{'=' * 80}"
        )

        # 打印到控制台和日志
        self.logger.info(log_msg)

        # 可视化假阳性
        if self.output_dir:
            self.visualize_false_positives(runner, num_samples=10)

    def calculate_far(self):
        """计算虚警率核心逻辑 - 增强调试信息"""
        self.logger.info('Starting FAR calculation for %d images', len(self.annotations))

        fp_total = 0
        tp_total = 0
        img_count = len(self.annotations)
        valid_images = 0

        for idx, (dets, ann) in enumerate(zip(self.detections, self.annotations)):
            # 记录处理进度
            if (idx + 1) % 10 == 0:
                self.logger.debug('Processing image %d/%d', idx + 1, img_count)

            gt_bboxes = ann['bboxes']
            gt_ignore = ann.get('bboxes_ignore', np.zeros((0, 5)))

            # 过滤低置信度检测
            if len(dets) == 0:
                continue

            try:
                # 应用置信度阈值
                if dets.ndim == 2 and dets.shape[1] >= 6:
                    valid_mask = dets[:, 5] > self.score_thr
                    det_bboxes = dets[valid_mask, :5]
                else:
                    self.logger.warning('Invalid detections shape at index %d: %s',
                                        idx, str(dets.shape))
                    continue

                if len(det_bboxes) == 0:
                    continue

                # 计算IoU
                if len(gt_bboxes) > 0:
                    ious = rbbox_overlaps(
                        det_bboxes.astype(np.float32),
                        gt_bboxes.astype(np.float32))
                else:
                    ious = np.zeros((len(det_bboxes), 0))

                # 匹配检测和真实框
                for det_idx in range(len(det_bboxes)):
                    if ious.shape[1] == 0:
                        fp_total += 1
                    else:
                        max_iou = ious[det_idx].max()
                        if max_iou >= self.iou_thr:
                            tp_total += 1
                        else:
                            fp_total += 1

                valid_images += 1

            except Exception as e:
                self.logger.error('Error processing image %d: %s', idx, str(e))

        self.logger.info('Processed %d valid images out of %d', valid_images, img_count)

        # 计算虚警率
        total_detections = tp_total + fp_total
        if total_detections == 0:
            far = 0.0
            self.logger.warning('No detections found for FAR calculation')
        else:
            far = fp_total / total_detections * 100

        # 每张图像的虚警数
        fppi = fp_total / img_count if img_count > 0 else 0

        return far, fppi, fp_total, tp_total
    def visualize_false_positives(self, runner, num_samples=10):
        """可视化假阳性检测 - DOTA数据集优化版"""
        import matplotlib.pyplot as plt
        from mmcv.image import imread

        # 随机选择样本
        indices = np.random.choice(len(self.detections),
                                   min(num_samples, len(self.detections)),
                                   replace=False)

        for idx in indices:
            img_path = self.images[idx]
            dets = self.detections[idx]
            ann = self.annotations[idx]

            # 读取图像
            img = imread(img_path, channel_order='rgb')

            # 过滤低置信度检测
            if dets.ndim == 2 and dets.shape[1] >= 6:
                valid_mask = dets[:, 5] > self.score_thr
                dets = dets[valid_mask]
            else:
                continue

            # 创建可视化
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(img)

            # 绘制真实目标（绿色）
            for gt in ann['bboxes']:
                rect = plt.Rectangle(
                    (gt[0], gt[1]),
                    gt[2], gt[3],
                    angle=gt[4],
                    fill=False,
                    edgecolor='green',
                    linewidth=1.5)
                ax.add_patch(rect)

            # 如果没有检测结果，跳过
            if len(dets) == 0:
                continue

            # 计算IoU
            if len(ann['bboxes']) > 0:
                ious = rbbox_overlaps(
                    dets[:, :5].astype(np.float32),
                    ann['bboxes'].astype(np.float32))
            else:
                ious = np.zeros((len(dets), 0))

            # 绘制检测结果并标记假阳性
            for i, det in enumerate(dets):
                if len(det) < 5:  # 确保有5个参数 (x, y, w, h, angle)
                    continue

                max_iou = ious[i].max() if ious.size > 0 else 0

                # 真阳性：蓝色，假阳性：红色
                color = 'blue' if max_iou >= self.iou_thr else 'red'

                rect = plt.Rectangle(
                    (det[0], det[1]),
                    det[2], det[3],
                    angle=det[4],
                    fill=False,
                    edgecolor=color,
                    linewidth=1.5)
                ax.add_patch(rect)

                # 添加置信度分数
                if color == 'red' and len(det) >= 6:  # 只标记假阳性
                    ax.text(det[0], det[1], f'{det[5]:.2f}',
                            color='red', fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.7))

            # 保存图像
            img_name = os.path.basename(img_path)
            save_path = os.path.join(self.output_dir, f'far_{img_name}')
            plt.title(f'False Positives (Red boxes) - {img_name}')
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()

            # 记录保存路径（使用纯ASCII字符）
            self.logger.info('Saved false positive visualization: %s', save_path)

    def after_run(self, runner):
        """运行结束后关闭日志处理器"""
        if self.file_handler:
            # 移除处理器
            self.logger.removeHandler(self.file_handler)
            self.file_handler.close()
            self.logger.info('Closed FAR log file: %s', self.log_file)

        # 添加结束标记
        self.logger.info('=' * 80)
        self.logger.info('FAR Evaluation Hook completed successfully')
        self.logger.info('=' * 80)
