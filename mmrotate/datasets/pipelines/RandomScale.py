import random
import numpy as np
from mmcv.runner import auto_fp16
from ..builder import PIPELINES

@PIPELINES.register_module()
class RandomScale:
    def __init__(self, scale_factor_range=(0.8, 1.2)):
        self.scale_factor_range = scale_factor_range

    @auto_fp16()
    def __call__(self, results):
        scale_factor = random.uniform(*self.scale_factor_range)
        results['img'] = self.scale_image(results['img'], scale_factor)
        if 'gt_bboxes' in results:
            results['gt_bboxes'] = self.scale_bboxes(results['gt_bboxes'], scale_factor)
        if 'gt_masks' in results:
            results['gt_masks'] = self.scale_masks(results['gt_masks'], scale_factor)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = self.scale_image(results['gt_semantic_seg'], scale_factor)
        if 'gt_boundary' in results:
            results['gt_boundary'] = self.scale_image(results['gt_boundary'], scale_factor)
        return results

    def scale_image(self, img, scale_factor):
        new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
        scaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        return scaled_img

    def scale_bboxes(self, bboxes, scale_factor):
        scaled_bboxes = bboxes * scale_factor
        return scaled_bboxes

    def scale_masks(self, masks, scale_factor):
        scaled_masks = []
        for mask in masks:
            new_size = (int(mask.shape[1] * scale_factor), int(mask.shape[0] * scale_factor))
            scaled_mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
            scaled_masks.append(scaled_mask)
        return np.array(scaled_masks)

    def __repr__(self):
        return f'{self.__class__.__name__}(scale_factor_range={self.scale_factor_range})'
