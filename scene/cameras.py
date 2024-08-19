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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid  # cam_info的id，以images名字的顺序排列1111123334231；遍历cam_infos时enumerate生成的index, 从0开始
        self.colmap_id = colmap_id  # 对应的cam_intrinsics的相机的id
        self.R = R  # COLMAP生成的是W2C，之前取了转置，C2W   RC2W, tW2C
        self.T = T  # tcw
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)    # 这里的gt_image是resize以后的
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        
        # 距离相机平面znear和zfar之间且在视锥内的物体才会被渲染
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans  # 相机中心的平移
        self.scale = scale  # 相机中心坐标的缩放

        # 世界到相机坐标系的变换矩阵的转置,4×4, W^T
        # getWorld2View2(RC2W, tW2C, trans, scale): W2C
        # getWorld2View2(R, T, trans, scale)).transpose(0, 1): W2C的转置W^T，注意不是C2W
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()

        # 投影矩阵的转置 J^T
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()

        # 从世界坐标系到图像的变换矩阵  W^T J^T
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        # 相机在世界坐标系下的坐标，twc
        # world_view_transform：W2C的转置 Tcw^T
        # world_view_transform.inverse()：Tcw^T^{-1} = Tcw^{-1}^T = Twc^T = [Rwc  0]
        #                                                                   [twc  1]    
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

