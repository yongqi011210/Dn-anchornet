import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import ROTATED_HEADS


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze())
        max_out = self.fc(self.max_pool(x).squeeze())
        scale = (avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * att


class ColorAwareConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1):
        super().__init__()
        self.main_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel,
                      padding=(kernel // 2) * dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.color_adjust = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.main_conv(x)
        color_weights = self.color_adjust(feat)
        c1, c2 = color_weights.chunk(2, dim=1)
        return feat * (c1 + c2) * 0.8 + feat * 0.2


class EnhancedAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.color_guard = nn.Sequential(
            nn.Conv2d(3, channels // 4, 3, padding=1),
            nn.InstanceNorm2d(channels // 4),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1)
        )

    def forward(self, x, orig_img):
        channel_att = self.channel_att(x)
        spatial_att = self.spatial_att(x)
        color_info = self.color_guard(orig_img)
        color_mask = torch.sigmoid(color_info)
        return x * (channel_att * 0.7 + spatial_att * 0.3) * color_mask


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x + self.gamma * self.conv(x)


class LiteNonLocalBlock(nn.Module):
    def __init__(self, channels, stride=4):
        super().__init__()
        self.stride = stride
        self.theta = nn.Conv2d(channels, channels // 16, 1)
        self.phi = nn.Conv2d(channels, channels // 16, 1)
        self.g = nn.Conv2d(channels, channels // 16, 1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels // 16, channels, 1),
            nn.GroupNorm(8, channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_down = F.avg_pool2d(x, self.stride)
        theta = self.theta(x_down).view(B, -1, (H // self.stride) * (W // self.stride))
        phi = self.phi(x_down).view(B, -1, (H // self.stride) * (W // self.stride))
        att = torch.bmm(theta.transpose(1, 2), phi)
        att = F.softmax(att, dim=-1)
        g = self.g(x_down).view(B, -1, (H // self.stride) * (W // self.stride))
        out = torch.bmm(g, att.transpose(1, 2))
        out = out.view(B, -1, H // self.stride, W // self.stride)
        return x + F.interpolate(self.out_conv(out), (H, W), mode='bilinear') * 0.5


class EnhancedFusion(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            ResidualBlock(out_ch),
            ChannelAttention(out_ch),
            SpatialAttention(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch // 8)
        )

    def forward(self, x):
        out = self.fusion(x)
        return out


class DetailEnhancer(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        self.detail_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, padding=2),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, 3, padding=1)
        )

    def forward(self, x):
        identity = x
        detail = self.detail_extractor(x)
        return identity + detail * 0.5


class ContrastAwareFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.contrast_net = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, denoised, orig):
        contrast_map = self.contrast_net(orig) * 1.8
        enhanced = denoised * contrast_map + orig * (1 - contrast_map)
        return torch.clamp(enhanced, -1, 1)


class EnhancedDenoiseDecoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.noise_prior = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.InstanceNorm2d(32)
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ColorAwareConv(in_ch, 64)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ColorAwareConv(64, 32),
            nn.Conv2d(32, 32, 3, padding=1)
        )

        self.noise_gate = nn.Sequential(
            nn.Conv2d(32 + 32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
            nn.Sigmoid()
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            ResidualBlock(32),
            DetailEnhancer(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        self.attention = EnhancedAttention(32)
        self.contrast_fusion = ContrastAwareFusion()

    def forward(self, x, orig_img):


        x = self.up1(x)


        x = self.up2(x)


        noise_feat = F.interpolate(self.noise_prior(orig_img), x.shape[2:])


        noise_mask = self.noise_gate(torch.cat([x, noise_feat], dim=1))


        x = self.attention(x * (1 - noise_mask), orig_img)


        denoised = self.final_conv(x)


        enhanced = self.contrast_fusion(
            denoised,
            F.interpolate(orig_img, denoised.shape[2:], mode='bilinear')
        )

        return enhanced


@ROTATED_HEADS.register_module()
class DenoisingHead(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()

        # 高频保护分支
        self.hf_protect = nn.Sequential(
            nn.Conv2d(3, base_ch // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch // 2, 3, 3, padding=1)
        )

        self.hf_extractor = nn.Sequential(
            nn.Conv2d(3, base_ch, 5, stride=2, padding=2),
            ResidualBlock(base_ch),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            LiteNonLocalBlock(base_ch * 2, stride=4),
            nn.InstanceNorm2d(base_ch * 2)
        )

        self.encoder = nn.ModuleList([
            nn.Sequential(
                ColorAwareConv(3, base_ch, dilation=2),
                nn.MaxPool2d(2)
            ),
            nn.Sequential(
                ColorAwareConv(base_ch, base_ch * 2, dilation=1),
                nn.MaxPool2d(2)
            )
        ])

        self.align_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(base_ch * 2)
            ),
            nn.Identity()
        ])

        self.fusion = EnhancedFusion(base_ch * 6, base_ch * 2)
        self.decoder = EnhancedDenoiseDecoder(base_ch * 2)

        self.res_scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_ch * 2, 32, 1),
            nn.LayerNorm([32, 1, 1]),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):


        orig = x.clone()

        # 高频保护分支
        hf_protected = self.hf_protect(x) * 0.3 + x * 0.7


        hf_feat = self.hf_extractor(hf_protected)


        # 编码器处理
        enc_feats = []
        x = orig  # 重置输入
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            enc_feats.append(x)


        # 特征对齐
        aligned_feats = []
        for i, feat in enumerate(enc_feats):
            aligned = self.align_layers[i](feat)
            aligned_feats.append(aligned)


        enc_concat = torch.cat([aligned_feats[0], aligned_feats[1], hf_feat], dim=1)


        # 特征融合
        fused = self.fusion(enc_concat)


        # 解码器处理
        denoised = self.decoder(fused, orig)


        # 残差缩放
        scale_map = F.interpolate(self.res_scale(fused), denoised.shape[2:])


        res = F.interpolate(orig, denoised.shape[2:], mode='bilinear')
        output = torch.sqrt(scale_map) * denoised + (1 - torch.sqrt(scale_map)) * res


        return torch.clamp(output, -1, 1)
