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
import numpy as np
from einops import rearrange
from torch import einsum
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from Utils import *
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


def _is_contiguous(tensor: torch.Tensor) -> bool:
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


class LayerNorm2d(nn.LayerNorm):
    r""" https://huggingface.co/spaces/Roll20/pet_score/blob/b258ef28152ab0d5b377d9142a23346f863c1526/lib/timm/models/convnext.py#L85
    LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        """
        @x: (B,C,H,W)
        """
        if _is_contiguous(x):
            return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()
        else:
            s, u = torch.var_mean(x, dim=1, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x



class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, norm='batch', **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        self.bn = nn.Identity()
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
              if norm=='batch':
                self.bn = nn.BatchNorm3d(out_channels)
              elif norm=='instance':
                self.bn = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
              if norm=='batch':
                self.bn = nn.BatchNorm2d(out_channels)
              elif norm=='instance':
                self.bn = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x


class Conv3dNormActReduced(nn.Module):
    def __init__(self, C_in, C_out, hidden=None, kernel_size=3, kernel_disp=None, stride=1, norm=nn.BatchNorm3d):
        super().__init__()
        if kernel_disp is None:
          kernel_disp = kernel_size
        if hidden is None:
            hidden = C_out
        self.conv1 = nn.Sequential(
            nn.Conv3d(C_in, hidden, kernel_size=(1,kernel_size,kernel_size), padding=(0, kernel_size//2, kernel_size//2), stride=(1, stride, stride)),
            norm(hidden),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(hidden, C_out, kernel_size=(kernel_disp, 1, 1), padding=(kernel_disp//2, 0, 0), stride=(stride, 1, 1)),
            norm(C_out),
            nn.ReLU(),
        )


    def forward(self, x):
        """
        @x: (B,C,D,H,W)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x




class ResnetBasicBlock(nn.Module):
  def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm2d, bias=False):
    super().__init__()
    self.norm_layer = norm_layer
    if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride


  def forward(self, x):
    identity = x

    out = self.conv1(x)
    if self.norm_layer is not None:
      out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    if self.norm_layer is not None:
      out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)

    return out


class ResnetBasicBlock3D(nn.Module):
  def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm3d, bias=False):
    super().__init__()
    self.norm_layer = norm_layer
    if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv3d(planes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride


  def forward(self, x):
    identity = x

    out = self.conv1(x)
    if self.norm_layer is not None:
      out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    if self.norm_layer is not None:
      out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)

    return out


class FlashMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, window_size=(-1,-1)):
        """
        @query: (B,L,C)
        """
        B,L,C = query.shape
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.head_dim)
        K = K.view(K.size(0), K.size(1), self.num_heads, self.head_dim)
        V = V.view(V.size(0), V.size(1), self.num_heads, self.head_dim)

        # todo: onnx库中自定义算子实现没有的解决方法
        attn_output = flash_attn_func(Q, K, V, window_size=window_size)  # Replace with actual FlashAttention function   pytorch用这个
        # attn_output = F.scaled_dot_product_attention(Q, K, V) # onnx用这个

        attn_output = attn_output.reshape(B,L,-1)
        output = self.out_proj(attn_output)

        return output



class FlashAttentionTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1, act=nn.GELU, norm=nn.LayerNorm):
        super().__init__()
        self.self_attn = FlashMultiheadAttention(embed_dim, num_heads)
        self.act = act()

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, window_size=(-1, -1)):
        src2 = self.self_attn(src, src, src, src_mask, window_size=window_size)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src



class UpsampleConv(nn.Module):
    def __init__(self, C_in, C_out, is_3d=False, kernel_size=3, bias=True, stride=1, padding=1):
        super().__init__()
        self.is_3d = is_3d
        if is_3d:
          self.conv = nn.Conv3d(C_in, C_out, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=bias)
        else:
          self.conv = nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=bias)

    def forward(self, x):
        if self.is_3d:
          mode = 'trilinear'
        else:
          mode = 'bilinear'
        x = F.interpolate(x, size=None, scale_factor=2, align_corners=False, mode=mode)
        x = self.conv(x)
        return x



class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=bn, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=bn, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode='bilinear')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class BasicConv_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x


class Conv2x_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = ResnetBasicBlock(out_channels*2, out_channels*mul, kernel_size=3, stride=1, padding=1, norm_layer=nn.InstanceNorm2d)
        else:
            self.conv2 = BasicConv_IN(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode='bilinear')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


def groupwise_correlation(fea1, fea2, num_groups):
    # 计算时W宽度通常为1，循环迭代这个函数计算相关性
    # 此函数对应为论文中的求点积过程
    # 每个输出位置的值表示：在特定组内，左右特征图的对应位置特征相似度
    # 输入特征形状: fea1 和 fea2 均为 [B, C, H, W]
    B, C, H, W = fea1.shape

    # 检查通道数是否可被分组数整除
    assert C % num_groups == 0, f"C:{C}, num_groups:{num_groups}"

    # 计算每个组的通道数
    channels_per_group = C // num_groups

    # 将特征按通道维度分组
    # [B, C, H, W] -> [B, num_groups, C/num_groups, H, W]
    fea1 = fea1.reshape(B, num_groups, channels_per_group, H, W)
    fea2 = fea2.reshape(B, num_groups, channels_per_group, H, W)

    # 使用混合精度上下文 (禁用自动类型转换)
    with torch.cuda.amp.autocast(enabled=False):
      # 第一部分
      # 特征归一化: 沿通道维度(channels_per_group)进行L2归一化
      # 输入形状: [B, num_groups, C/num_groups, H, W]
      # 输出形状: 保持原形状，但每个通道向量被归一化为单位向量
      # 第二部分
      # 计算分组相关性: 逐元素相乘后沿通道维度求和
      # 数学等价于两个单位向量的点积 -> 余弦相似度
      cost = (F.normalize(fea1.float(), dim=2) * F.normalize(fea2.float(), dim=2)).sum(dim=2)  #!NOTE Divide first for numerical stability
    assert cost.shape == (B, num_groups, H, W) # 输出形状
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups, stride=1):
    """
    构建分组相关代价体积(Group-wise Correlation Cost Volume)

    max_disp 是立体匹配任务中定义的 ​最大视差搜索范围，表示在左图和右图之间寻找对应点时，允许的最大水平位移（单位为像素）。
    例如：若 max_disp=192，表示算法会在左图的每个像素点，向右侧搜索最多 192 个像素的距离，寻找右图中的匹配点。

    数学原理：
    给定左特征F^L ∈ ℝ^(B×C×H×W) 和右特征F^R ∈ ℝ^(B×C×H×W)
    代价体积V ∈ ℝ^(B×G×D×H×W)的计算过程：

    对于每个视差d ∈ [0, maxdisp):
        1. 右特征水平平移d个像素得到F^R_shifted
        2. 将F^L和F^R_shifted的通道分为G组：
           每组特征维度 ℝ^(B×(C/G)×H×W)
        3. 计算逐像素分组点积：
           V[b,g,d,h,w] = ∑_{c∈第g组} F^L[b,c,h,w] * F^R_shifted[b,c,h,w]

    参数说明：
    @refimg_fea:    左图特征张量，形状 (Batch, Channels, Height, Width)
    @targetimg_fea: 右图特征张量，形状需与左图特征相同
    @maxdisp:       最大视差搜索范围（通常基于下采样后的特征图分辨率设定）
    @num_groups:    分组数量（G），将通道划分为G个独立计算组（GPU显存有限时，增大 G 可降低单组计算量；污染仅影响少数组，其余组仍能提供可靠匹配信息；每组生成独立的匹配置信度，最终融合时能覆盖更全面的匹配线索）本质上是 特征解耦 和 计算并行化 的权衡
    @stride:        滑动步长（当前代码未使用，保留参数供扩展）

    返回：
    5D代价体积张量，形状 (Batch, Groups, DisparityLevels, Height, Width)

    运行示例：
    假设输入特征尺寸：(B=2, C=32, H=128, W=128), maxdisp=64, num_groups=8
    则：
    - 每组通道数 = 32/8 = 4
    - 当d=16时：(检测到视差有16个像素点)
      左图有效区域：[:, :, :, 16:] (128-16=112列)
      右图有效区域：[:, :, :, :-16] (128-16=112列)
      计算结果填充到 volume[:, :, 16, :, 16:] (仅有效区域)
    - 最终体积形状：(2,8,64,128,128)
    """

    # 获取输入特征图的基本形状参数
    B, C, H, W = refimg_fea.shape  # B:批大小 | C:通道数 | H:特征图高度 | W:特征图宽度

    # 初始化全零代价体积（显存预分配）
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])  # 形状(B,G,D,H,W)

    # 遍历所有可能的视差值（构建视差维度）
    for i in range(maxdisp):
        """
        视差计算数学表达：
        当视差为i时，等效于将右图向左平移i个像素
        匹配关系满足：x_R = x_L - i (需保证x_R ≥ 0)
        因此有效区域为：左图x ∈ [i, W) 对应右图x ∈ [0, W-i)
        """
        if i > 0:  # 处理非零视差情况
            # 切片操作实现平移对齐（避免无效区域计算）
            # 当视差为 i 时，右图需要向左平移 i 个像素才能与左图对应区域对齐。此时：
            # 左图的有效区域：x_L ∈[i,W)（左图从第 i 列开始到末尾）。
            # 右图的有效区域：x_R ∈[0,W−i)（右图从第 0 列开始到倒数第 i 列）。
            # 计算后的相关性结果填充到右图有效区域对应的位置（第i列之后）
            volume[:, :, i, :, i:] = groupwise_correlation(
                refimg_fea[:, :, :, i:],  # 左图有效区域：i到W列
                targetimg_fea[:, :, :, :-i],  # 右图有效区域：0到W-i列
                num_groups
            )
        else:  # 处理零视差情况（完全对齐）
            # 全图直接计算相关性
            volume[:, :, i, :, :] = groupwise_correlation(
                refimg_fea,
                targetimg_fea,
                num_groups
            )

    # 确保内存连续布局（优化后续卷积操作效率）
    volume = volume.contiguous()

    return volume


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    """
    构建通道拼接的代价体积（用于后续3D卷积处理）

    参数说明：
    @refimg_fea:    左图特征张量，形状 (Batch, Channels, Height, Width)
    @targetimg_fea: 右图特征张量，形状需与左图相同
    @maxdisp:       最大视差搜索范围

    返回：
    5D代价体积张量，形状 (Batch, 2*Channels, MaxDisp, Height, Width)
    """

    # 获取左图特征的基本形状参数
    B, C, H, W = refimg_fea.shape  # B:批大小 | C:通道数 | H:高度 | W:宽度

    # 初始化全零代价体积（显存预分配）
    # 体积维度：通道维度是左右特征拼接后的2*C
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])

    # 遍历所有可能的视差值（构建视差维度）
    for i in range(maxdisp):
        """ 当视差为i时的处理逻辑：
        1. 左特征保持完整
        2. 右特征需左移i个像素（即切片[:, :, :, :-i]）
        3. 将左右特征沿通道维度拼接，填充到对应视差层
        """
        if i > 0:  # 非零视差情况（右特征需要水平平移）
            # 左特征填充到前C个通道（所有位置）
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]

            # 右特征填充到后C个通道（仅有效区域）
            # 右特征切片：去除右侧i列（相当于左移i像素）
            # 填充位置：从第i列开始（与左图第i列对齐）
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:  # 零视差情况（完全对齐）
            # 左右特征直接拼接
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea

    # 确保内存连续布局（优化后续卷积效率）
    volume = volume.contiguous()

    return volume



def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.reshape(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


class FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        """
        特征注意力模块：通过特征图生成注意力权重，调整Cost Volume的通道特征

        参数:
            cv_chan (int): Cost Volume的通道数（最终输出的通道数）
            feat_chan (int): 输入特征图(feat)的通道数
        """
        super(FeatureAtt, self).__init__()

        # 特征注意力权重生成器 输出的是特征
        self.feat_att = nn.Sequential(
            # 降维卷积：将特征图通道数减半
            # 输入形状：(B, feat_chan, H, W)
            # 输出形状：(B, feat_chan//2, H, W)
            BasicConv(feat_chan, feat_chan//2, kernel_size=1, stride=1, padding=0),
            # 1x1卷积：将通道数对齐到Cost Volume的通道数(cv_chan)
            # 输入形状：(B, feat_chan//2, H, W)
            # 输出形状：(B, cv_chan, H, W)
            nn.Conv2d(feat_chan//2, cv_chan, 1)
            )

    def forward(self, cv, feat):
        '''
        @cv: cost volume (B,C,D,H,W)
        @feat: (B,C,H,W)
        前向传播流程

        参数:
            cv: Cost Volume张量，形状 (B, C, D, H, W)
                其中:
                - B: 批次大小
                - C: 通道数（cv_chan）
                - D: 深度/视差维度
                - H/W: 空间高度/宽度
            feat: 输入特征图，形状 (B, C_feat, H, W)
                其中 C_feat = feat_chan（初始化参数）

        返回:
            调整后的Cost Volume，形状 (B, C, D, H, W)
        '''
        # 步骤1: 生成注意力权重（2D → 3D广播）
        # todo:why we need unsqueeze(2)?
        # 保证和cv维度一致方便相乘， 在相乘时，pytorch会自动扩展1到D，也就是复制D次，此时feat_att变为(B, C, D, H, W)
        # 每个深度位置 D 共享相同的空间注意力权重
        feat_att = self.feat_att(feat).unsqueeze(2)   # 输出形状: (B, cv_chan, H, W) -> 插入深度维度 → (B, cv_chan, 1, H, W)

        # 步骤2: 应用注意力权重调整Cost Volume
        cv = torch.sigmoid(feat_att)*cv # 权重归一化到[0, 1]
        return cv    # 广播乘法 → (B, C, D, H, W)

def context_upsample(disp_low, up_weights):
    """
    @disp_low: (b,1,h,w)  1/4 resolution
    @up_weights: (b,9,4*h,4*w)  Image resolution
    """
    b, c, h, w = disp_low.shape

    disp_unfold = F.unfold(disp_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    disp_unfold = F.interpolate(disp_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)

    disp = (disp_unfold*up_weights).sum(1)

    return disp



class PositionalEmbedding(nn.Module):
  def __init__(self, d_model, max_len=512):
    super().__init__()

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model).float()
    pe.require_grad = False

    position = torch.arange(0, max_len).float().unsqueeze(1)  #(N,1)
    div_term = (torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).exp()[None]

    pe[:, 0::2] = torch.sin(position * div_term)  #(N, d_model/2)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    self.pe = pe
    # self.register_buffer('pe', pe)  #(1, max_len, D)


  def forward(self, x, resize_embed=False):
    '''
    @x: (B,N,D)
    '''
    self.pe = self.pe.to(x.device).to(x.dtype)
    pe = self.pe
    if pe.shape[1]<x.shape[1]:
      if resize_embed:
        pe = F.interpolate(pe.permute(0,2,1), size=x.shape[1], mode='linear', align_corners=False).permute(0,2,1)
      else:
        raise RuntimeError(f'x:{x.shape}, pe:{pe.shape}')
    return x + pe[:, :x.size(1)]



class CostVolumeDisparityAttention(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, act=nn.GELU, norm_first=False, num_transformer=6, max_len=512, resize_embed=False):
    super().__init__()
    self.resize_embed = resize_embed
    self.sa = nn.ModuleList([])
    for _ in range(num_transformer):
      self.sa.append(FlashAttentionTransformerEncoderLayer(embed_dim=d_model, num_heads=nhead, dim_feedforward=dim_feedforward, act=act, dropout=dropout))
    self.pos_embed0 = PositionalEmbedding(d_model, max_len=max_len)


  def forward(self, cv, window_size=(-1,-1)):
    """
    @cv: (B,C,D,H,W) where D is max disparity
    """
    x = cv
    B,C,D,H,W = x.shape
    x = x.permute(0,3,4,2,1).reshape(B*H*W, D, C)
    x = self.pos_embed0(x, resize_embed=self.resize_embed)  #!NOTE No resize since disparity is pre-determined
    for i in range(len(self.sa)):
        x = self.sa[i](x, window_size=window_size)
    x = x.reshape(B,H,W,D,C).permute(0,4,3,1,2)

    return x



class ChannelAttentionEnhancement(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionEnhancement, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentionExtractor(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionExtractor, self).__init__()

        self.samconv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.samconv(x)
        return self.sigmoid(x)



class EdgeNextConvEncoder(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6, expan_ratio=4, kernel_size=7, norm='layer'):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        if norm=='layer':
          self.norm = LayerNorm2d(dim, eps=1e-6)
        else:
          self.norm = nn.Identity()
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x