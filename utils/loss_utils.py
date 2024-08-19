#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()  # 每个像素每个通道的颜色差的绝对值的平均值 (3, height, width)
    # \Sigma_{i=0}^{i=c*h*w} (render_c,h,w - gt_c,h,w)/CHW

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

# 创建高斯窗口，这里的高斯窗口用于平滑图像
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)    # 创建一个一维的高斯分布，并将其转换为二维高斯分布
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)    # 将两个一维高斯分布矩阵相乘，得到一个二维高斯分布矩阵；增加两个维度，使其适应卷积操作
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous()) # 扩展窗口，使其适应图像的通道数。
    return window

def ssim(img1, img2, window_size=11, size_average=True):    # 结构相似性指数（SSIM），一种衡量两张图像之间相似性的指标
    # https://en.wikipedia.org/wiki/Structural_similarity_index_measure
    # 考虑了亮度 (luminance)、对比度 (contrast) 和结构 (structure)指标，更接近人类感觉
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2) # 局部均值
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq  #  局部方差
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2 # 局部协方差

    C1 = 0.01 ** 2  # 为了避免分母为零而引入的常数
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()  # 所有像素平均值
    else:
        return ssim_map.mean(1).mean(1).mean(1)

