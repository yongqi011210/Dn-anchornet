import numpy as np
from mmcv.utils import Registry
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class RandomRotate90:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() < self.prob:
            angle = np.random.choice([90, 180, 270])
            image = results['img']
            results['img'] = np.rot90(image, k=angle // 90)
            for key in results.get('bbox_fields', []):
                bboxes = results[key]
                # Perform bbox rotation here if needed
                results[key] = bboxes
            for key in results.get('mask_fields', []):
                masks = results[key]
                results[key] = [np.rot90(mask, k=angle // 90) for mask in masks]
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(prob={self.prob})'


