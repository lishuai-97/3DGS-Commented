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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]: # 原始尺寸/ (缩放倍数)，缩放倍数 = 缩放因子（默认1.0） * 传入缩放参数（默认-1）
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:   # 默认执行这里
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600 # 这里的1600指的是宽度，1920会缩放至1600，1920/1600=1.2
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)    # 1.2*1.0=1.2
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)  # 将原始图像resize，然后添加颜色通道，成为CHW的图像

    gt_image = resized_image_rgb[:3, ...]   # 提取张量的前3个通道，表示RGB图像，这里的GTimage已经是根据分辨率resize以后的
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4: # 如果 C 通道上的维度为4（有alpha） # Q：这里为什么是第二个？HWC到底是怎样排列的？
        loaded_mask = resized_image_rgb[3:4, ...]   # 提取C中的第4个通道（Alpha通道）作为 loaded_mask，加mask掩膜在这加

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))  #  按不同分辨率缩放倍数存储的Camera的list

    return camera_list

def camera_to_JSON(id, camera : Camera): # Rwc和tcw RC2W, tW2C  
    Rt = np.zeros((4, 4))   # Tcw
    Rt[:3, :3] = camera.R.transpose()   # Rcw
    Rt[:3, 3] = camera.T    # tcw
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt) # 这里代码错了，应该是C2W，如果想要W2C，不应该求逆; 这个函数生成的json文件没有被后续使用
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]   # 旋转转化为序列化的列表
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
