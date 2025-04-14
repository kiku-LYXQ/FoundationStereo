# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch,pdb,os,sys
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from Utils import *

class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, dx=None):
        """
            多尺度几何编码与相关性融合模块（金字塔结构）

            参数:
                init_fmap1 (Tensor): 初始特征图1，形状 (B, C, H, W)
                init_fmap2 (Tensor): 初始特征图2，形状 (B, C, H, W)
                geo_volume (Tensor): 3D几何体积数据，形状 (B, C, D, H, W)
                num_levels (int): 金字塔层级数（默认2层）
                dx (Tensor): 初始位移参数（用于坐标对齐）
        """
        self.num_levels = num_levels
        self.geo_volume_pyramid = [] # 存储多尺度几何体积
        self.init_corr_pyramid = []  # 存储多尺度初始相关性
        self.dx = dx

        # ------------------- 阶段1：计算初始相关性图 -------------------
        # 计算左右视图特征图之间的归一化相关性
        # 输入形状: (B,C,H,W) x2 → 输出形状: (B,H,W,1,W2)
        # 其中W2为右视图特征图的宽度（相关窗口）
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        # ------------------- 阶段2：数据形状预处理 -------------------
        # 解析张量形状
        b, h, w, _, w2 = init_corr.shape # init_corr形状: (B,H,W,1,W2)
        b, c, d, h, w = geo_volume.shape  # geo_volume形状:(B, C, D, H, W)
        # 几何体积形状重塑：
        # (B,C,D,H,W) → [维度置换] → (B,H,W,C,D) → [合并空间维度] → (B*H*W,C,1,D)
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d).contiguous()

        init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        for i in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)

        for i in range(self.num_levels-1):
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)


    def __call__(self, disp, coords, low_memory=False):
        b, _, h, w = disp.shape
        self.dx = self.dx.to(disp.device)
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            x0 = self.dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            disp_lvl = torch.cat([x0,y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl, low_memory=low_memory)
            geo_volume = geo_volume.reshape(b, h, w, -1)

            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + self.dx   # X on right image
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl, low_memory=low_memory)
            init_corr = init_corr.reshape(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)
        out_pyramid = torch.cat(out_pyramid, dim=-1)
        return out_pyramid.permute(0, 3, 1, 2).contiguous()   #(B,C,H,W)


    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.reshape(B, D, H, W1)
        fmap2 = fmap2.reshape(B, D, H, W2)
        with torch.cuda.amp.autocast(enabled=False):
          corr = torch.einsum('aijk,aijh->ajkh', F.normalize(fmap1.float(), dim=1), F.normalize(fmap2.float(), dim=1))
        corr = corr.reshape(B, H, W1, 1, W2)
        return corr