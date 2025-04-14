# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch,pdb,os,sys
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from core.submodule import *
from core.extractor import *

class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(DispHead, self).__init__()
        self.conv = nn.Sequential(
          nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
          nn.ReLU(),
          EdgeNextConvEncoder(input_dim, expan_ratio=4, kernel_size=7, norm=None),
          EdgeNextConvEncoder(input_dim, expan_ratio=4, kernel_size=7, norm=None),
          nn.Conv2d(input_dim, output_dim, 3, padding=1),
        )

    def forward(self, x):
        return self.conv(x)

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)
        h = (1-z) * h + z * q
        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, args, ngroup=8):
        super(BasicMotionEncoder, self).__init__()
        self.args = args
        cor_planes = args.corr_levels * (2*args.corr_radius + 1) * (ngroup+1)
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 256, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+256, 128-1, 3, padding=1)

    def forward(self, disp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


class RaftConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, kernel_size=3):
        super().__init__()
        # 公式(1)的卷积操作: 更新门计算
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim,
                               kernel_size, padding=kernel_size // 2)
        # 公式(2)的卷积操作: 重置门计算
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim,
                               kernel_size, padding=kernel_size // 2)
        # 公式(3)的卷积操作: 候选状态计算
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim,
                               kernel_size, padding=kernel_size // 2)

    def forward(self, h, x, hx):
        # 公式(1): 更新门计算
        z = torch.sigmoid(self.convz(hx))  # hx = Concat(h, x)

        # 公式(2): 重置门计算
        r = torch.sigmoid(self.convr(hx))

        # 公式(3): 候选状态计算
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))  # r*h 对应公式中的 r_t ⊙ h_{t-1}

        # 公式(4): 状态更新
        h = (1 - z) * h + z * q
        return h


class SelectiveConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256,
                 small_kernel_size=1, large_kernel_size=3,
                 patch_size=None):
        """
        选择性卷积GRU模块：结合不同感受野的GRU单元，通过注意力机制动态融合结果 (改进的GRU实现方法)

        参数说明:
            hidden_dim:  隐藏状态的通道数 (默认128)
            input_dim:   输入特征的通道数 (默认256)
            small_kernel_size: 小卷积核尺寸 (默认1x1)
            large_kernel_size: 大卷积核尺寸 (默认3x3)
            patch_size:  未使用（保留参数）
        """
        super(SelectiveConvGRU, self).__init__()

        # -------------------- 特征预处理层 --------------------
        # conv0: 输入特征初步编码（保持维度）
        # 输入: (B, input_dim, H, W) → 输出: (B, input_dim, H, W)
        self.conv0 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(),  # 保持空间维度不变
        )

        # -------------------- 特征融合层 ----------------------
        # conv1: 混合输入特征和隐藏状态（通道数翻倍）
        # 输入: (B, input_dim + hidden_dim, H, W) → 输出: (B, input_dim + hidden_dim, H, W)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim,
                      input_dim + hidden_dim,
                      kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # -------------------- 双路GRU单元 ---------------------
        # 小核GRU：捕获局部细节（如边缘对齐）
        self.small_gru = RaftConvGRU(hidden_dim, input_dim, small_kernel_size)

        # 大核GRU：捕获上下文关系（如物体形状）
        self.large_gru = RaftConvGRU(hidden_dim, input_dim, large_kernel_size)

    def forward(self, att, h, *x):
        """
        前向传播流程（维度说明以默认参数为例）

        参数:
            att: 注意力图 (B,1,H,W) → 控制局部/全局信息的融合权重
            h:   当前隐藏状态 (B,hidden_dim=128,H,W)
            x:   输入特征列表，每个元素维度为 (B,C,H,W)

        流程:
            (1) 拼接输入特征 → (B,256,H,W)
            (2) 特征预处理 → (B,256,H,W)
            (3) 拼接隐藏状态 → (B,384,H,W)
            (4) 特征融合 → (B,384,H,W)              改进点1： 对输入的特征图做了几次卷积操作融合特征
            (5) 双路GRU计算 → (B,128,H,W) * 2
            (6) 注意力加权融合 → (B,128,H,W)         改进点2： 对大小核的GRU模块进行加权融合，以融合后做为最终输出
        """
        # 3×3卷积 + ReLU 的组合是CNN的经典设计（如VGG、ResNet），能有效提取局部特征并增强非线性表达能力。
        # 步骤1: 拼接输入特征（假设x包含多个特征图）
        # x = [feat1, feat2] → (B,256,H,W)（当input_dim=256时）   这里对应为池化后的隐藏初始net和inp做拼接
        x = torch.cat(x, dim=1)  # 沿通道维度拼接

        # 步骤2: 输入特征初步编码（3x3卷积 + ReLU）
        # (B,256,H,W) → (B,256,H,W)
        x = self.conv0(x)

        # 步骤3: 拼接编码后的特征与隐藏状态
        # h.shape = (B,128,H,W) → 拼接后 (B,256+128=384,H,W)
        hx = torch.cat([x, h], dim=1)

        # 步骤4: 混合特征融合（3x3卷积 + ReLU）
        # (B,384,H,W) → (B,384,H,W)
        hx = self.conv1(hx)

        # 步骤5: 并行计算双路GRU（数学公式见下方）
        # 小核GRU: 处理局部细节 → (B,128,H,W)
        # 大核GRU: 处理全局关系 → (B,128,H,W)
        small_h = self.small_gru(h, x, hx)  # 使用h, x, hx作为输入
        large_h = self.large_gru(h, x, hx)

        # 步骤6: 注意力加权融合（核心创新点）
        # 公式: h_new = small_h * att + large_h * (1 - att)
        # att.shape = (B,1,H,W)，通过广播机制进行通道对齐
        h = small_h * att + large_h * (1 - att)

        return h  # 输出更新后的隐藏状态 (B,128,H,W)


class BasicSelectiveMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, volume_dim=8):
        super().__init__()
        self.args = args

        # 运动特征编码器：将视差(disp)和代价体积(corr)编码为运动特征
        # 输入: disp(B,1,H,W) + corr(B,D,H,W) → 输出: (B, volume_dim, H, W)
        self.encoder = BasicMotionEncoder(args, volume_dim)

        # 分层GRU定义
        # ------------------------- 第3层GRU (1/16分辨率) -------------------------
        if args.n_gru_layers == 3:
            # gru16输入维度: hidden_dim*2=256（来自pool后的下层特征）
            # 输出维度: hidden_dim=128（保持与隐藏状态一致）
            self.gru16 = SelectiveConvGRU(
                hidden_dim,  # 输入通道数 256
                hidden_dim * 2  # 隐藏状态维度 128
            )

        # ------------------------- 第2层GRU (1/8分辨率) --------------------------
        if args.n_gru_layers >= 2:
            # gru08输入维度动态变化:
            # - 当存在第3层时: hidden_dim + hidden_dim*2=384 (下层池化特征+上层插值特征)
            # - 当仅有2层时: hidden_dim*2=256
            input_dim = hidden_dim * (args.n_gru_layers == 3) + hidden_dim * 2
            self.gru08 = SelectiveConvGRU(hidden_dim, input_dim)

        # ------------------------- 第1层GRU (1/4分辨率) ---------------------------
        # gru04输入维度:
        # - 当存在上层时: hidden_dim + hidden_dim*2=384 (下层池化特征+上层插值特征)
        # - 当无上层时: hidden_dim*2=256
        self.gru04 = SelectiveConvGRU(
            hidden_dim,
            hidden_dim * (args.n_gru_layers > 1) + hidden_dim * 2
        )

        # 视差预测头: 将GRU隐藏状态映射为视差增量
        # 输入: (B,128,H/4,W/4) → 输出: (B,1,H/4,W/4)
        self.disp_head = DispHead(
            hidden_dim,  # 128
            256  # 中间特征维度
        )

        # 上采样掩膜生成网络
        # 输入: (B,128,H/4,W/4) → 输出: (B,32,H/4,W/4)
        self.mask = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # (B,64,H/4,W/4)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),  # (B,32,H/4,W/4)
            nn.ReLU(inplace=True),
        )

    def forward(self, net, inp, corr, disp, att):
        """前向传播过程
        Args:
            net (list): 隐藏状态列表，对应GRU各层h的初始值，各元素维度为:
                net[0] (B,128,H/4,W/4)  第1层（高分辨率）
                net[1] (B,128,H/8,W/8)  第2层（中分辨率）
                net[2] (B,128,H/16,W/16) 第3层（低分辨率）
            inp (list): 上下文特征列表，各元素维度:
                inp[0] (B,128,H/4,W/4)
                inp[1] (B,128,H/8,W/8)
                inp[2] (B,128,H/16,W/16)
            corr (Tensor): 代价体积特征 (B,D,H,W)
            disp (Tensor): 当前视差估计 (B,1,H,W)
            att (list): 空间注意力图，各元素维度与inp对应层一致
        """
        # ------------------------- 第3层GRU处理 (1/16分辨率) ------------------------
        if self.args.n_gru_layers == 3:
            # 输入特征组成:
            # - att[2] (B,1,H/16,W/16)      → 空间注意力
            # - net[2] (B,128,H/16,W/16)    → 当前隐藏状态
            # - inp[2] (B,128,H/16,W/16)    → 上下文特征
            # - pool2x(net[1]) (B,128,H/16,W/16) → 第2层特征池化到1/16分辨率
            net[2] = self.gru16(
                att[2], net[2], inp[2],
                pool2x(net[1])  # (B,128,H/8,W/8) → (B,128,H/16,W/16)
            )

        # ------------------------- 第2层GRU处理 (1/8分辨率) --------------------------
        if self.args.n_gru_layers >= 2:
            if self.args.n_gru_layers > 2:
                # 存在第3层时，输入特征包括:
                # - att[1] (B,1,H/8,W/8)
                # - net[1] (B,128,H/8,W/8)
                # - inp[1] (B,128,H/8,W/8)
                # - pool2x(net[0]) (B,128,H/8,W/8) → 第1层池化
                # - interp(net[2], net[1]) (B,128,H/8,W/8) → 第3层插值到1/8
                net[1] = self.gru08(
                    att[1], net[1], inp[1],
                    pool2x(net[0]),
                    interp(net[2], net[1])
                )
            else:
                # 仅2层时，输入特征减少:
                net[1] = self.gru08(
                    att[1], net[1], inp[1],
                    pool2x(net[0])  # (B,128,H/4,W/4) → (B,128,H/8,W/8)
                )

        # ------------------------- 运动特征编码 --------------------------------
        # 输入: disp(B,1,H,W) + corr(B,D,H,W) → 输出: (B,8,H,W)
        motion_features = self.encoder(disp, corr)

        # 与第1层上下文特征拼接 → (B,128+8=136,H/4,W/4)
        motion_features = torch.cat([inp[0], motion_features], dim=1)

        # ------------------------- 第1层GRU处理 (1/4分辨率) ---------------------
        if self.args.n_gru_layers > 1:
            # 输入特征包括:
            # - att[0] (B,1,H/4,W/4)
            # - net[0] (B,128,H/4,W/4)
            # - motion_features (B,136,H/4,W/4)
            # - interp(net[1], net[0]) (B,128,H/4,W/4) → 第2层插值到1/4
            net[0] = self.gru04(
                att[0], net[0], motion_features,
                interp(net[1], net[0])
            )

        # ------------------------- 视差增量预测 -----------------------------
        # 输入: net[0] (B,128,H/4,W/4) → 输出: delta_disp (B,1,H/4,W/4)
        delta_disp = self.disp_head(net[0])

        # ------------------------- 上采样掩膜生成 ---------------------------
        # 输入: net[0] (B,128,H/4,W/4) → 输出: mask (B,32,H/4,W/4)
        mask = 0.25 * self.mask(net[0])  # 缩放梯度平衡

        return net, mask, delta_disp
