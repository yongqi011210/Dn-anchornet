import torch
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from torch import nn
from ..builder import ROTATED_HEADS, build_loss

@ROTATED_HEADS.register_module()
class EdgeHead(BaseDenseHead):
    def __init__(self,
                 in_channels,
                 conv_out_channels=256,
                 num_convs=4,
                 loss_edge=dict(type='CrossEntropyLoss', loss_weight=1.0)):
        super(EdgeHead, self).__init__()
        self.loss_edge = build_loss(loss_edge)
        self.conv_out_channels = conv_out_channels

        convs = []
        for _ in range(num_convs):
            convs.append(nn.Conv2d(in_channels, conv_out_channels, kernel_size=3, padding=1))
            convs.append(nn.ReLU(inplace=True))
            in_channels = conv_out_channels
        self.convs = nn.Sequential(*convs)
        self.conv_logits = nn.Conv2d(conv_out_channels, 1, kernel_size=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.convs(x)
        edge_pred = self.conv_logits(x)
        return edge_pred

    def loss(self, edge_pred, gt_edges):
        loss_edge = self.loss_edge(edge_pred, gt_edges)
        return dict(loss_edge=loss_edge)

    def forward_train(self, x, img_metas, proposals, gt_edges, gt_bboxes_ignore, **kwargs):
        edge_pred = self.forward(x)
        loss_edge = self.loss(edge_pred, gt_edges)
        return loss_edge

    def simple_test(self, x, proposals, img_metas, rescale=False):
        return self.forward(x)

    def aug_test(self, x, proposals, img_metas, rescale=False):
        return self.simple_test(x, proposals, img_metas, rescale=rescale)
