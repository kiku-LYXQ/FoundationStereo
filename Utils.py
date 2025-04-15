# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os, sys, time,torch,torchvision,pickle,trimesh,itertools,datetime,imageio,logging,joblib,importlib,argparse
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
import pandas as pd
import open3d as o3d
import cv2
import numpy as np
from transformations import *
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)



def set_logging_format(level=logging.INFO):
  importlib.reload(logging)
  FORMAT = '%(message)s'
  logging.basicConfig(level=level, format=FORMAT, datefmt='%m-%d|%H:%M:%S')

set_logging_format()



def set_seed(random_seed):
  import torch,random
  np.random.seed(random_seed)
  random.seed(random_seed)
  torch.manual_seed(random_seed)
  torch.cuda.manual_seed_all(random_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def toOpen3dCloud(points,colors=None,normals=None):
  cloud = o3d.geometry.PointCloud()
  cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
  if colors is not None:
    if colors.max()>1:
      colors = colors/255.0
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
  if normals is not None:
    cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
  return cloud


def depth2xyzmap(depth: np.ndarray, K, uvs: np.ndarray = None, zmin=0.1):
  """
  将深度图转换为三维点云坐标图（每个像素对应三维坐标）

  参数说明：
  depth : np.ndarray - 深度图矩阵，形状(H,W)，单位米
  K     : np.ndarray - 相机内参矩阵，形状(3,3)，格式：
                      [[fx, 0,  cx],
                       [0,  fy, cy],
                       [0,  0,  1 ]]
  uvs   : np.ndarray - 可选，指定要计算的像素坐标集合，形状(N,2)
  zmin  : float      - 有效深度最小值，小于该值的深度视为无效

  返回：
  xyz_map : np.ndarray - 三维坐标图，形状(H,W,3)，每个像素存储[X,Y,Z]坐标
  """

  # 生成无效点掩码（深度小于zmin的区域）
  invalid_mask = (depth < zmin)

  # 获取深度图尺寸
  H, W = depth.shape[:2]

  # 生成像素坐标网格 --------------------------------------------------------
  if uvs is None:
    # 当未指定uvs时，计算全图像素坐标
    # vs: 垂直坐标矩阵（行索引），形状(H,W)
    # us: 水平坐标矩阵（列索引），形状(H,W)
    vs, us = np.meshgrid(
      np.arange(0, H),  # 0到H-1的垂直坐标
      np.arange(0, W),  # 0到W-1的水平坐标
      sparse=False,
      indexing='ij'  # 矩阵索引模式（i=行，j=列）
    )
    # 展平为1D数组（用于后续向量化计算）
    vs = vs.reshape(-1)  # 形状(H*W,)
    us = us.reshape(-1)  # 形状(H*W,)
  else:
    # 当指定uvs时，只计算特定像素坐标
    uvs = uvs.round().astype(int)  # 坐标取整
    us = uvs[:, 0]  # 提取水平坐标列
    vs = uvs[:, 1]  # 提取垂直坐标列

  # 根据相机模型计算三维坐标 ------------------------------------------------
  # 公式原理（透视投影逆变换）：
  # X = (u - cx) * Z / fx
  # Y = (v - cy) * Z / fy
  # Z = depth[v,u]

  # 获取深度值（注意坐标顺序是vs,us）
  zs = depth[vs, us]  # 形状(N,)

  # 计算X坐标（水平方向）
  xs = (us - K[0, 2]) * zs / K[0, 0]  # (u - cx)*Z/fx
  # 计算Y坐标（垂直方向）
  ys = (vs - K[1, 2]) * zs / K[1, 1]  # (v - cy)*Z/fy

  # 组合三维坐标
  pts = np.stack((xs.reshape(-1),  # X坐标
                  ys.reshape(-1),  # Y坐标
                  zs.reshape(-1)),  # Z坐标
                 1)  # 形状(N,3)

  # 构建三维坐标图 --------------------------------------------------------
  xyz_map = np.zeros((H, W, 3), dtype=np.float32)  # 初始化全零矩阵
  xyz_map[vs, us] = pts  # 将计算的三维坐标填入对应像素位置

  # 处理无效点（将深度小于zmin的位置坐标置零）
  if invalid_mask.any():
    xyz_map[invalid_mask] = 0  # 无效区域坐标设为(0,0,0)

  return xyz_map



def freeze_model(model):
  model = model.eval()
  for p in model.parameters():
    p.requires_grad = False
  for p in model.buffers():
    p.requires_grad = False
  return model



def get_resize_keep_aspect_ratio(H, W, divider=16, max_H=1232, max_W=1232):
  assert max_H%divider==0
  assert max_W%divider==0

  def round_by_divider(x):
    return int(np.ceil(x/divider)*divider)

  H_resize = round_by_divider(H)   #!NOTE KITTI width=1242
  W_resize = round_by_divider(W)
  if H_resize>max_H or W_resize>max_W:
    if H_resize>W_resize:
      W_resize = round_by_divider(W_resize*max_H/H_resize)
      H_resize = max_H
    else:
      H_resize = round_by_divider(H_resize*max_W/W_resize)
      W_resize = max_W
  return int(H_resize), int(W_resize)


def vis_disparity(disp, min_val=None, max_val=None, invalid_thres=np.inf, color_map=cv2.COLORMAP_TURBO, cmap=None, other_output={}):
  """
  @disp: np array (H,W)
  @invalid_thres: > thres is invalid
  """
  disp = disp.copy()
  H,W = disp.shape[:2]
  invalid_mask = disp>=invalid_thres
  if (invalid_mask==0).sum()==0:
    other_output['min_val'] = None
    other_output['max_val'] = None
    return np.zeros((H,W,3))
  if min_val is None:
    min_val = disp[invalid_mask==0].min()
  if max_val is None:
    max_val = disp[invalid_mask==0].max()
  other_output['min_val'] = min_val
  other_output['max_val'] = max_val
  vis = ((disp-min_val)/(max_val-min_val)).clip(0,1) * 255
  if cmap is None:
    vis = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), color_map)[...,::-1]
  else:
    vis = cmap(vis.astype(np.uint8))[...,:3]*255
  if invalid_mask.any():
    vis[invalid_mask] = 0
  return vis.astype(np.uint8)



def depth_uint8_decoding(depth_uint8, scale=1000):
  depth_uint8 = depth_uint8.astype(float)
  out = depth_uint8[...,0]*255*255 + depth_uint8[...,1]*255 + depth_uint8[...,2]
  return out/float(scale)

