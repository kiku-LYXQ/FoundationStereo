# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch,logging,os,sys,urllib,warnings
import torch.nn as nn
import torch.nn.functional as F
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from core.submodule import *
from Utils import *
import timm


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn=='layer':
            self.norm1 = LayerNorm2d(planes)
            self.norm2 = LayerNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = LayerNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)



class MultiBasicEncoder(nn.Module):
    def __init__(self, output_dim=[128], norm_fn='batch', dropout=0.0, downsample=3):
        super(MultiBasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn=='layer':
            self.norm1 = LayerNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        output_list = []

        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs04 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dual_inp=False, num_layers=3):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if dual_inp:
            v = x
            x = x[:(x.shape[0]//2)]

        outputs04 = [f(x) for f in self.outputs04]
        if num_layers == 1:
            return (outputs04, v) if dual_inp else (outputs04,)

        y = self.layer4(x)
        outputs08 = [f(y) for f in self.outputs08]

        if num_layers == 2:
            return (outputs04, outputs08, v) if dual_inp else (outputs04, outputs08)

        z = self.layer5(y)
        outputs16 = [f(z) for f in self.outputs16]

        return (outputs04, outputs08, outputs16, v) if dual_inp else (outputs04, outputs08, outputs16)



class ContextNetDino(MultiBasicEncoder):
    def __init__(self, args, output_dim=[128], norm_fn='batch', downsample=3):
        nn.Module.__init__(self)

        # ------------------- ViT相关参数 -------------------
        self.args = args
        self.patch_size = 14    # ViT的patch划分尺寸
        self.image_size = 518   # ViT的输入图像尺寸
        self.vit_feat_dim = 384 # ViT输出特征维度

        code_dir = os.path.dirname(os.path.realpath(__file__))

        # ------------------- 主干网络结构 -------------------
        self.out_dims = output_dim # 各层级输出通道配置（e.g. downsample = 3 [[128, 128, 128]]）

        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn=='layer':
            self.norm1 = LayerNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)
        self.down = nn.Sequential(
          nn.Conv2d(128, 128, kernel_size=4, stride=4, padding=0),
          nn.BatchNorm2d(128),
        )
        vit_dim = DepthAnythingFeature.model_configs[self.args.vit_size]['features']//2
        self.conv2 = BasicConv(128+vit_dim, 128, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(256)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs04 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

    def forward(self, x_in, vit_feat, dual_inp=False, num_layers=3):
        """
        前向传播过程（假设输入图像尺寸为HxW）
        Args:
            x_in (Tensor): 输入图像，形状为 (B,3,H,W)
            vit_feat (Tensor): ViT提取的特征，形状需与CNN特征对齐
        Returns:
            tuple: 多尺度输出 (outputs04, outputs08, outputs16)
        """
        # ------------------- 初始卷积处理 -------------------
        B,C,H,W = x_in.shape
        x = self.conv1(x_in)    # 输入：B,3,H,W → 输出：B,64,H',W'（若stride=2则H'=H//2）
        x = self.norm1(x)       # 输入：B,64,H',W' → 输出：B,64,H',W'
        x = self.relu1(x)       # 输入：B,64,H',W' → 输出：B,64,H',W'
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)      # 输出：B,126,H/4,W/4

        divider = np.lcm(self.patch_size, 16)
        H_resize, W_resize = get_resize_keep_aspect_ratio(H,W, divider=divider, max_H=1344, max_W=1344)
        x = torch.cat([x, vit_feat], dim=1)
        x = self.conv2(x)
        outputs04 = [f(x) for f in self.outputs04]

        y = self.layer4(x)
        outputs08 = [f(y) for f in self.outputs08]

        z = self.layer5(y)
        outputs16 = [f(z) for f in self.outputs16]

        return (outputs04, outputs08, outputs16)


class DepthAnythingFeature(nn.Module):
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }

    def __init__(self, encoder='vits'):
        super().__init__()
        from depth_anything.dpt import DepthAnything
        self.encoder = encoder
        depth_anything = DepthAnything(self.model_configs[encoder])
        self.depth_anything = depth_anything

        self.intermediate_layer_idx = {   #!NOTE For V2
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }


    def forward(self, x):
        """
        @x: (B,C,H,W) (2, 3, 560, 1008)

        return:
        'out':      (B, D, H, W)  # 最终特征图（D由encoder类型决定，如vitl为256）  (2, 128, 560, 1008)
        'path_1':   (B, D1, H//2, W//2)       # 1/2分辨率中间特征（用于可视化）    (2, 128, 320, 576)
        'path_2':   (B, D2, H_path_1//2, W_path_1//2)       # 1/4分辨率中间特征（与CNN特征融合的关键层） (2, 128, 160, 288)
        'path_3':   (B, D3, H//8, W//8)       # 1/8分辨率中间特征  (2, 128, 160, 288)
        'path_4':   (B, D4, H//16, W//16)     # 1/16分辨率中间特征 (2, 128, 80, 144)
        'features': [features_1, ..., features_4]  # ViT各中间层原始输出（未重塑）
        'disp':     (B, 1, H, W)              # 单目深度估计结果（未被STA模块使用）
        """
        h, w = x.shape[-2:]
        features = self.depth_anything.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)


        patch_size = self.depth_anything.pretrained.patch_size   # 实际值为14
        patch_h, patch_w = h // patch_size, w // patch_size
        out, path_1, path_2, path_3, path_4, disp = self.depth_anything.depth_head.forward(features, patch_h, patch_w, return_intermediate=True)

        return {'out':out, 'path_1':path_1, 'path_2':path_2, 'path_3':path_3, 'path_4':path_4, 'features':features, 'disp':disp}  # path_1 is 1/2; path_2 is 1/4


class Feature(nn.Module):
    def __init__(self, args):
        """
        STA（Stereo-Temporal Adaptation）模块的核心特征提取网络
        融合CNN的局部特征与ViT的全局语义特征，实现跨模态特征对齐
        """
        super(Feature, self).__init__()
        self.args = args
        # ------------------- 主干网络初始化 -------------------
        # 使用预训练的EdgeNeXt-small作为基础CNN特征提取器
        # 论文中强调使用轻量级CNN保证计算效率，同时结合ViT的语义先验

        model = timm.create_model('edgenext_small', pretrained=True, features_only=False)
        self.stem = model.stem # 初始下采样层（4x）

        # 四阶段下采样:
        # Stage0: (B,48,H//4,W//4) → (B,96,H//8,W//8)
        # Stage1: → (B,160,H//16,W//16)
        # Stage2: → (B,304,H//32,W//32)
        self.stages = model.stages # 多阶段特征提取器（4个阶段）
        chans = [48, 96, 160, 304]
        self.chans = chans
        self.dino = DepthAnythingFeature(encoder=self.args.vit_size)
        self.dino = freeze_model(self.dino)
        vit_feat_dim = DepthAnythingFeature.model_configs[self.args.vit_size]['features']//2

        # ------------------- 特征解码器（Deconvolution） -------------------
        # 构建由粗到精的特征解码路径，逐步恢复空间分辨率
        # 每个解码阶段包含特征融合与上采样（对应论文中的多层级特征融合）
        # 特征解码器（对应论文图示左侧的Deconv路径）
        self.deconv32_16 = Conv2x_IN(chans[3], chans[2], deconv=True, concat=True)  # 输入: 160+160=320 → 输出160 	(B, 160, H//16, W//16)
        self.deconv16_8 = Conv2x_IN(chans[2]*2, chans[1], deconv=True, concat=True)  # 输入: 320+96=416 → 输出96      (B, 96, H//8, W//8)
        self.deconv8_4 = Conv2x_IN(chans[1]*2, chans[0], deconv=True, concat=True)   # 输入: 96+48=144 → 输出48       (B, 48, H//4, W//4)
        # ViT特征融合（对应论文图示右侧的STA效果区）
        self.conv4 = nn.Sequential(

          # 通道数：CNN特征(chans[0]*2) + ViT特征(128) = chans[0]*2+128
          BasicConv(chans[0]*2+vit_feat_dim, chans[0]*2+vit_feat_dim, kernel_size=3, stride=1, padding=1, norm='instance'),
          ResidualBlock(chans[0]*2+vit_feat_dim, chans[0]*2+vit_feat_dim, norm_fn='instance'),  # 残差连接保持梯度流
          ResidualBlock(chans[0]*2+vit_feat_dim, chans[0]*2+vit_feat_dim, norm_fn='instance'),  # 论文强调重复残差块提升特征鲁棒性
        )

        # ------------------- 单目先验特征提取（DepthAnything ViT-L） -------------------
        # 冻结预训练的深度估计模型，提取全局语义特征（对应论文中的STA核心设计）
        # ------------------- 特征增强模块 -------------------
        # 在1/4分辨率层级融合ViT特征并进行特征精炼
        self.dino = DepthAnythingFeature(encoder='vitl')  # 使用DepthAnything V2的ViT-Large版本
        self.dino = freeze_model(self.dino)  # 冻结参数，保持单目先验的稳定性
        # todo: patch_size的功能
        self.patch_size = 14  # ViT的patch划分尺寸，影响特征图分辨率
        # 各阶段输出通道数（用于后续立体匹配网络）
        self.d_out = [chans[0]*2+vit_feat_dim, chans[1]*2, chans[2]*2, chans[3]]


    def forward(self, x):
        """
        前向传播流程（对应论文图3右侧的零样本推理流程）
        输入形状: (B, 3, H, W)
        输出:
          features: 多尺度特征列表 [
            (B, 272, H//4, W//4),   # x4 (1/4尺度，含ViT融合)
            (B, 192, H//8, W//8),   # x8
            (B, 320, H//16, W//16), # x16
            (B, 304, H//32, W//32)  # x32
          ]
          vit_feat: (B, 128, H//4, W//4)  # 对齐后的ViT特征
        """
        B,C,H,W = x.shape

        # ------------------- ViT特征提取分支 -------------------
        # 动态调整输入尺寸以满足ViT的patch整除要求（论文中的跨分辨率对齐策略） 调整输入尺寸为14的倍数（ViT的patch尺寸）
        # todo: why?
        divider = np.lcm(self.patch_size, 16)  # 计算最小公倍数（保证同时适配CNN和ViT）
        H_resize, W_resize = get_resize_keep_aspect_ratio(H,W, divider=divider, max_H=1344, max_W=1344)
        # 双三次插值调整尺寸（保持几何形变最小化）
        x_in_ = F.interpolate(x, size=(H_resize, W_resize), mode='bicubic', align_corners=False)   # 调整输入尺寸为 14 的倍数

        # 冻结模式下的ViT特征提取（对应论文中的单目先验提取）
        self.dino = self.dino.eval()

        # 冻结ViT提取单目先验特征（输出分辨率：H_resize//14, W_resize//14）
        with torch.no_grad():  # 阻止梯度回传，保持预训练权重稳定
          output = self.dino(x_in_) # 形状：(B, D, H_resize//14, W_resize//14) D由encoder决定
        vit_feat = output['out']  # 提取ViT的最后一层特征

        # 将ViT特征对齐到CNN的1/4分辨率（H//4, W//4）
        # 论文强调多尺度特征融合，此处选择与CNN的1/4特征（x4）融合
        # 最终输出形状：(B, D, H//4, W//4)
        vit_feat = F.interpolate(vit_feat, size=(H//4,W//4), mode='bilinear', align_corners=True)  # 将 ViT 特征分辨率对齐到 CNN 特征的 1/4 尺度（与 x4 特征融合）

        # ================= CNN主干处理（对应论文图3左半流程） =================
        x = self.stem(x)           # 4x下采样 → (B, 48, H//4, W//4)
        x4 = self.stages[0](x)     # 8x下采样 → (B, 96, H//8, W//8)
        x8 = self.stages[1](x4)    # 16x下采样 → (B, 160, H//16, W//16)
        x16 = self.stages[2](x8)   # 32x下采样 → (B, 304, H//32, W//32)
        x32 = self.stages[3](x16)  # 保持32x → (B, 304, H//32, W//32)

        # ================= 特征解码与融合（对应论文3.1节的层级融合） =================
        x16 = self.deconv32_16(x32, x16)    # 1/32 → 1/16 → (B, 320, H//16, W//16)
        x8 = self.deconv16_8(x16, x8)       # 1/16 → 1/8 → (B, 96, H//8, W//8)
        x4 = self.deconv8_4(x8, x4)         # 1/8 → 1/4 → (B, 48, H//4, W//4)

        # ViT特征融合（对应论文图3右侧STA效果）
        x4 = torch.cat([x4, vit_feat], dim=1) # 通道拼接 → (B, 48+D, H//4, W//4) 对应论文的C操作
        x4 = self.conv4(x4)                          # 残差精炼 → 保持形状 (B, 48+D, H//4, W//4) 对应论文的
        return [x4, x8, x16, x32], vit_feat


