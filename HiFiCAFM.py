# HiFiCAFM.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 通道混洗
def channel_shuffle(x, groups):
    B, C, H, W = x.size()
    x = x.view(B, groups, C // groups, H, W)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x.view(B, C, H, W)

# CAFM 模块
class CAFM(nn.Module):
    def __init__(self, dim, num_heads=4, groups=4):
        super(CAFM, self).__init__()
        self.groups = groups
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=groups, bias=False),
            nn.LeakyReLU(inplace=True)
        )

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.attn_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape

        # 局部分支
        local_feat = self.local_conv(x)
        local_feat = channel_shuffle(local_feat, self.groups)

        # 全局注意力分支
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # 保证通道数能被 num_heads 整除
        assert C % self.num_heads == 0, f"Channel {C} not divisible by heads {self.num_heads}"

        q = rearrange(q, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        k = rearrange(k, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        v = rearrange(v, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.temperature, dim=-1)
        out = attn @ v
        out = rearrange(out, 'b h c (h1 w1) -> b (h c) h1 w1', h1=H, w1=W)

        global_feat = self.attn_proj(out)
        return local_feat + global_feat

# 编码器
class Encoder(nn.Module):
    def __init__(self, msg_dim=1):
        super().__init__()
        self.pre_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.cafm = CAFM(dim=64, num_heads=4, groups=4)

        self.encoder = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.embedder = nn.Sequential(
            nn.Conv2d(128 + msg_dim, 128, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 3, 1)
        )
        self.residual = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 3, 1)
        )

    def forward(self, image, msg):
        B, _, H, W = image.shape
        x = self.pre_conv(image)
        x = self.cafm(x)
        x = self.encoder(x)
        msg = msg.view(B, 1, 8, 8).repeat(1, 1, H // 8, W // 8)
        x = torch.cat([x, msg], dim=1)
        embed = self.embedder(x)
        out = image + self.residual(torch.cat([image, embed], dim=1))
        return out

# 解码器
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.cafm = CAFM(dim=64, num_heads=4, groups=4)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.PixelUnshuffle(8)
        )

    def forward(self, image):
        x = self.pre_conv(image)
        x = self.cafm(x)
        x = self.decoder(x)
        return x.mean(dim=[2, 3])  # 输出 [B, 64]

# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.out = nn.Linear(32, 1)

    def forward(self, image):
        x = self.net(image).view(image.size(0), -1)
        return self.out(x)

# 消息损失
class MessageLoss(nn.Module):
    def __init__(self):
        super(MessageLoss, self).__init__()

    def forward(self, extract_msg, origin_msg):
        return nn.functional.binary_cross_entropy_with_logits(extract_msg, origin_msg)
