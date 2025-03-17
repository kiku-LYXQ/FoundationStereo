# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch,pdb,logging,timm
import torch.nn as nn
import torch.nn.functional as F
import sys,os
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from core.update import *
from core.extractor import *
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import *
from core.utils.utils import *
from Utils import *
import time


try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def normalize_image(img):
    '''
    @img: (B,C,H,W) in range 0-255, RGB order
    '''
    tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    return tf(img/255.0).contiguous()


class hourglass(nn.Module):
    def __init__(self, cfg, in_channels, feat_dims=None):
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))

        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17))

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*6, in_channels*6, kernel_size=3, kernel_disp=17))


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, in_channels, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv_out = nn.Sequential(
          Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
          Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
        )

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))
        self.atts = nn.ModuleDict({
          "4": CostVolumeDisparityAttention(d_model=in_channels, nhead=4, dim_feedforward=in_channels, norm_first=False, num_transformer=4, max_len=self.cfg['max_disp']//16),
        })
        self.conv_patch = nn.Sequential(
          nn.Conv3d(in_channels, in_channels, kernel_size=4, stride=4, padding=0, groups=in_channels),
          nn.BatchNorm3d(in_channels),
        )

        self.feature_att_8 = FeatureAtt(in_channels*2, feat_dims[1])
        self.feature_att_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_32 = FeatureAtt(in_channels*6, feat_dims[3])
        self.feature_att_up_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_up_8 = FeatureAtt(in_channels*2, feat_dims[1])

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)
        x = self.conv_patch(x)
        x = self.atts["4"](x)
        x = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=False)
        conv = conv + x
        conv = self.conv_out(conv)

        return conv



class FoundationStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        context_dims = args.hidden_dims
        self.cv_group = 8
        volume_dim = 28

        self.cnet = ContextNetDino(output_dim=[args.hidden_dims, context_dims], downsample=args.n_downsample)
        self.update_block = BasicSelectiveMultiUpdateBlock(self.args, self.args.hidden_dims[0], volume_dim=volume_dim) # GRU核心模块
        self.sam = SpatialAttentionExtractor()
        self.cam = ChannelAttentionEnhancement(self.args.hidden_dims[0])

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, kernel_size=3, padding=3//2) for i in range(self.args.n_gru_layers)])

        self.feature = Feature()
        self.proj_cmb = nn.Conv2d(self.feature.d_out[0], 12, kernel_size=1, padding=0)

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )


        self.spx_2_gru = Conv2x(32, 32, True, bn=False)
        self.spx_gru = nn.Sequential(
          nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),
          )


        self.corr_stem = nn.Sequential(
            nn.Conv3d(32, volume_dim, kernel_size=1),
            BasicConv(volume_dim, volume_dim, kernel_size=3, padding=1, is_3d=True),
            ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
            ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
            )
        self.corr_feature_att = FeatureAtt(volume_dim, self.feature.d_out[0])
        self.cost_agg = hourglass(cfg=self.args, in_channels=volume_dim, feat_dims=self.feature.d_out)
        self.classifier = nn.Sequential(
          BasicConv(volume_dim, volume_dim//2, kernel_size=3, padding=1, is_3d=True),
          ResnetBasicBlock3D(volume_dim//2, volume_dim//2, kernel_size=3, stride=1, padding=1),
          nn.Conv3d(volume_dim//2, 1, kernel_size=7, padding=3),
        )

        r = self.args.corr_radius
        dx = torch.linspace(-r, r, 2*r+1, requires_grad=False).reshape(1, 1, 2*r+1, 1)
        self.dx = dx


    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        with autocast(enabled=self.args.mixed_precision):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)   # 1/2 resolution
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp.float()

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False, low_memory=False, init_disp=None):
        """
        立体匹配前向传播核心方法，通过迭代优化估计左右图像视差图

        参数说明:
        image1: 左视图图像张量，形状(B,C,H,W)
        image2: 右视图图像张量，形状(B,C,H,W)
        iters: 迭代优化次数，默认12次
        test_mode: 测试模式标志，True时只返回最终结果，False时返回中间结果
        low_memory: 低内存模式标志，启用特殊优化策略减少显存占用
        init_disp: 初始视差图，可选预估值用于热启动

        返回:
        test_mode=True时: 返回最终上采样后的视差图 (B,1,H,W)
        test_mode=False时: 返回(初始视差, 各迭代阶段视差预测列表)
        """

        # -------------------------- 初始化设置 --------------------------
        B = len(image1)  # 批大小
        low_memory = low_memory or self.args.get('low_memory', False)  # 内存优化模式

        # 图像标准化处理 (归一化到特定范围)
        image1 = normalize_image(image1)
        image2 = normalize_image(image2)

        # 混合精度计算上下文管理
        with autocast(enabled=self.args.mixed_precision):
            # -------------------------- 特征提取 --------------------------
            # 并行提取左右图像多尺度特征
            # out: 多尺度特征列表，包含左右视图拼接后的特征
            #      结构为[features_level4, features_level8, features_level16, features_level32]
            #      每个元素的形状为 (2B, C_scale, H/scale, W/scale)
            #      例如：
            #      - features_level4: (2B, 224, H//4, W//4)  通道数 chans[0]*2+128 = 48 * 2+128=224
            #      - features_level8: (2B, 192, H//8, W//8)  通道数 chans[1]*2 = 96 * 2=192
            #      - features_level16: (2B, 320, H//16, W//16) 通道数 chans[2]*2 = 160 * 2=320
            #      - features_level32: (2B, 304, H//32, W//32) 通道数 chans[3] = 304
            #
            # vit_feat: 仅左视图的ViT特征（因右视图未输入DepthAnything模型）
            #          形状为 (B, 1024, H//4, W//4)  # ViT-L的通道数1024，分辨率与原图1/4对齐
            out, vit_feat = self.feature(torch.cat([image1, image2], dim=0))
            vit_feat = vit_feat[:B]  # 提取ViT特征(CNN作为骨干网络)

            # 分割左右视图特征列表(每个元素为不同尺度的特征图)
            features_left = [o[:B] for o in out]  # 左视图多级特征
            features_right = [o[B:] for o in out]  # 右视图多级特征

            # 提取2x下采样特征 (用于后续上采样)
            stem_2x = self.stem_2(image1)  # (B,C,H/2,W/2)

            # -------------------------- 构建代价体积 --------------------------
            # 组相关代价体积 (Group-wise Correlation Volume)
            gwc_volume = build_gwc_volume(
                features_left[0], features_right[0],
                self.args.max_disp // 4, self.cv_group  # max_disp为最大视差搜索范围
            )  # 形状: (B, G, D', H', W')
            # 其中:
            # - B: 批次大小
            # - G: 分组数量（self.cv_group）
            # todo: Why max_disp should be divided by 4 ?
            # [Answer] Original max_disp is defined for input image resolution,
            #          but feature maps are typically downsampled by a factor (e.g., 1/4).
            #          Scale max_disp according to feature map resolution:
            #          adjusted_max_disp = max_disp / scale_factor
            #          where scale_factor = original_height / feature_map_height
            #          Example: if input is 480x640 and feature map is 120x160, scale_factor=4
            #          Add this adjustment before cost volume computation

            # - D' = self.args.max_disp // 4: 当前特征图的最大视差（下采样后的视差级别数）
            # - H' = H_ori // 4: 特征图高度（原图的1/4）
            # - W' = W_ori // 4: 特征图宽度（原图的1/4）
            #
            # 示例:
            # 若输入图像尺寸为 (H,W)=(512,512)，max_disp=256，cv_group=8，
            # 则 gwc_volume.shape = (2, 8, 64, 128, 128)

            # 连接特征代价体积
            left_tmp = self.proj_cmb(features_left[0])  # 投影后的左特征 (B, C, H//4, W//4)
            right_tmp = self.proj_cmb(features_right[0])  # 投影后的右特征 (B, C, H//4, W//4)

            # 构建拼接式代价体积
            # 输入特征分辨率已下采样至原图的1/4，故最大视差需同步缩放为 max_disp//4
            concat_volume = build_concat_volume(
                left_tmp,
                right_tmp,
                maxdisp=self.args.max_disp // 4  # 实际视差搜索范围 = 原始max_disp // 4
            )  # 输出形状：(B, 2*C, max_disp//4, H//4, W//4)

            del left_tmp, right_tmp  # 及时释放临时变量节省显存

            # 融合两种代价体积
            comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)  # 通道维度拼接
            comb_volume = self.corr_stem(comb_volume)  # 通过卷积压缩通道数
            comb_volume = self.corr_feature_att(comb_volume, features_left[0])  # 加入注意力机制
            comb_volume = self.cost_agg(comb_volume, features_left)  # 多尺度代价聚合

            # -------------------------- 初始视差估计 --------------------------
            # 通过分类器生成视差概率分布
            prob = F.softmax(self.classifier(comb_volume).squeeze(1), dim=1)  # (B,D,H,W)

            # 视差回归计算(加权求和概率分布得到连续视差值)
            if init_disp is None:
                init_disp = disparity_regression(prob, self.args.max_disp // 4)  # (B,1,H/4,W/4)

            # -------------------------- 上下文特征提取 --------------------------
            # 通过上下文网络提取多尺度上下文信息
            cnet_list = self.cnet(image1, vit_feat=vit_feat,
                                  num_layers=self.args.n_gru_layers)  # 返回金字塔特征列表

            # 分割隐藏状态和输入特征
            net_list = [torch.tanh(x[0]) for x in cnet_list]  # GRU隐藏状态初始化
            inp_list = [torch.relu(x[1]) for x in cnet_list]  # 上下文特征金字塔
            inp_list = [self.cam(x) * x for x in inp_list]  # 通道注意力调制
            att = [self.sam(x) for x in inp_list]  # 空间注意力图

        # -------------------------- 迭代优化视差 --------------------------
        # 初始化几何编码模块（结合特征和代价体积）
        geo_fn = Combined_Geo_Encoding_Volume(
            features_left[0].float(), features_right[0].float(),
            comb_volume.float(),
            num_levels=self.args.corr_levels,  # 多级相关层数
            dx=self.dx  # 视差采样间隔
        )

        # 生成水平坐标网格（用于几何编码）
        b, c, h, w = features_left[0].shape
        coords = torch.arange(w, device=init_disp.device).reshape(1, 1, w, 1)
        coords = coords.repeat(b, h, 1, 1)  # (B,H,W,1) 水平坐标矩阵

        disp = init_disp.float()  # 当前视差估计
        disp_preds = []  # 存储各迭代阶段结果

        # 3层GRU模块的迭代优化循环
        for itr in range(iters):
            disp = disp.detach()  # 切断梯度计算（仅用于当前迭代）

            # 几何特征计算
            geo_feat = geo_fn(disp, coords, low_memory=low_memory)

            # 混合精度计算块
            with autocast(enabled=self.args.mixed_precision):
                # GRU单元更新隐藏状态和视差增量
                net_list, mask_feat_4, delta_disp = self.update_block(
                    net_list,  # 隐藏状态列表
                    inp_list,  # 上下文特征列表
                    geo_feat,  # 几何编码特征
                    disp,  # 当前视差
                    att  # 注意力图
                )

            # 视差更新（加上增量）
            disp = disp + delta_disp.float()

            # 测试模式下跳过中间结果保存
            if test_mode and itr < iters - 1:
                continue

            # 上采样当前视差到原图分辨率
            disp_up = self.upsample_disp(
                disp.float(),  # 低分辨率视差
                mask_feat_4.float(),  # 上采样掩膜特征
                stem_2x.float()  # 2x下采样特征用于引导上采样
            )
            disp_preds.append(disp_up)  # 记录当前迭代结果

        # -------------------------- 返回结果 --------------------------
        if test_mode:
            return disp_up  # 测试模式返回最终视差图
        return init_disp, disp_preds  # 训练模式返回初始视差和迭代过程结果
        # 立体匹配前向传播全流程（12次迭代优化版）

        # 1.
        # 输入预处理阶段
        # ├─ 1.1
        # 图像归一化
        # │    ├─ 左视图：image1 ∈ (B, 3, H, W) → [0, 1]
        # 标准化
        # │    └─ 右视图：image2 ∈ (B, 3, H, W) → 相同处理
        # │
        # ├─ 1.2
        # 特征金字塔构建（并行提取）
        # │    ├─ 拼接输入：cat([image1, image2]) ∈ (2B, 3, H, W)
        # │    ├─ 多级特征提取：features_level
        # {4, 8, 16, 32}
        # │    │    ├─ level4: (2B, 224, H // 4, W // 4) → 浅层细节
        # │    │    ├─ level8: (2B, 192, H // 8, W // 8) → 中层语义
        # │    │    ├─ level16: (2B, 320, H // 16, W // 16) → 深层抽象
        # │    │    └─ level32: (2B, 304, H // 32, W // 32) → 全局上下文
        # │    │
        # │    └─ ViT特征提取：vit_feat ∈ (B, 1024, H // 4, W // 4)
        #
        # 2.
        # 代价体积构建阶段（核心匹配信息）
        # ├─ 2.1
        # 组相关代价体积(GWC)
        # │    ├─ 输入：level4特征(features_left[0], features_right[0])
        # │    ├─ 操作：分组点积计算
        # │    ├─ 输出：gwc_volume ∈ (B, G=8, D'=max_disp//4,H//4,W//4)
        # │    └─ 特性：保留通道间局部相关性
        # │
        # ├─ 2.2 拼接式代价体积(Concat)
        # │    ├─ 特征投影：proj_cmb降维 → (B, C, H // 4, W // 4)
        # │    ├─ 右特征平移：切片[:-i]
        # 实现视差对齐
        # │    ├─ 输出：concat_volume ∈ (B, 2C, D',H//4,W//4)
        # │    └─ 特性：保留原始特征完整性
        # │
        # ├─ 2.3 双代价体积融合
        # │    ├─ 通道拼接：comb_volume =[GWC; Concat] ∈ (B, 8+2C, D',H//4,W//4)
        # │    ├─ 卷积压缩：corr_stem → 统一通道维度
        # │    ├─ 注意力增强：corr_feature_att融合全局上下文
        # │    └─ 多尺度聚合：cost_agg整合金字塔特征
        #
        # 3. 初始视差估计阶段（粗匹配）
        # ├─ 3.1 概率分布生成
        # │    ├─ 分类器：3D卷积 → (B, 1, D',H//4,W//4)
        # │    └─ Softmax归一化：prob ∈ (B, D',H//4,W//4)
        # │
        # ├─ 3.2 视差回归
        # │    ├─ 加权求和：∑(d * prob) → init_disp ∈ (B, 1, H // 4, W // 4)
        # │    └─ 热启动机制：允许外部初始化(init_disp参数)
        #
        # 4.
        # 上下文特征提取（动态优化基础）
        # ├─ 4.1
        # 多尺度上下文网络
        # │    ├─ 输入：左视图image1 + ViT特征
        # │    ├─ 输出：金字塔特征列表[(h1, x1), (h2, x2), ...]
        # │    │    ├─ h: GRU隐藏状态初始化值
        # │    │    └─ x: 上下文特征
        # │    └─ 层级数：n_gru_layers（通常3 - 4层）
        # │
        # ├─ 4.2
        # 特征增强
        # │    ├─ 通道注意力：CAM模块动态加权特征图
        # │    └─ 空间注意力：SAM生成空间权重掩膜
        #
        # 5.
        # 迭代优化阶段（GRU循环细化）
        # ├─ 5.1
        # 几何编码初始化
        # │    ├─ 构建几何函数：Combined_Geo_Encoding_Volume()
        # │    │    ├─ 输入：左 / 右特征、融合代价体积
        # │    │    ├─ 功能：计算当前视差下的几何一致性特征
        # │    │    └─ 多级相关：corr_levels控制感受野
        # │    │
        # │    └─ 坐标网格生成：coords ∈ (B, H, W, 1)
        # 记录水平坐标
        # │
        # └─ 5.2
        # GRU迭代循环（12次）
        # ├─ 单次迭代流程：
        # │    ├─ 几何特征计算 → geo_feat
        # │    │    ├─ 当前视差disp ∈ (B, 1, H // 4, W // 4)
        # │    │    ├─ 坐标映射：disp → 右视图对应点
        # │    │    └─ 多级相关特征采样
        # │    │
        # │    ├─ GRU状态更新
        # │    │    ├─ 输入：geo_feat + 上下文特征
        # │    │    ├─ 隐藏状态更新：net_list
        # │    │    └─ 输出：delta_disp（视差修正量）
        # │    │
        # │    ├─ 视差更新：disp += delta_disp
        # │    │
        # │    └─ 上采样准备（最后迭代）
        # │         ├─ 掩膜特征：mask_feat_4 ∈ (B, 9, H // 4, W // 4)
        # │         └─ 引导特征：stem_2x ∈ (B, C, H // 2, W // 2)
        # │
        # └─ 5.3
        # 结果上采样
        # ├─ 输入：低分辨率disp + 高维特征
        # ├─ 操作：可变形卷积引导上采样
        # └─ 输出：disp_up ∈ (B, 1, H, W)
        #
        # 6.
        # 输出阶段
        # ├─ 测试模式：直接返回最终上采样结果disp_up
        # └─ 训练模式：返回初始视差 + 所有迭代中间结果（用于多阶段监督）


    def run_hierachical(self, image1, image2, iters=12, test_mode=False, low_memory=False, small_ratio=0.5):
      B,_,H,W = image1.shape
      img1_small = F.interpolate(image1, scale_factor=small_ratio, align_corners=False, mode='bilinear')
      img2_small = F.interpolate(image2, scale_factor=small_ratio, align_corners=False, mode='bilinear')
      padder = InputPadder(img1_small.shape[-2:], divis_by=32, force_square=False)
      img1_small, img2_small = padder.pad(img1_small, img2_small)
      disp_small = self.forward(img1_small, img2_small, test_mode=True, iters=iters, low_memory=low_memory)
      disp_small = padder.unpad(disp_small.float())
      disp_small_up = F.interpolate(disp_small, size=(H,W), mode='bilinear', align_corners=True) * 1/small_ratio
      disp_small_up = disp_small_up.clip(0, None)

      padder = InputPadder(image1.shape[-2:], divis_by=32, force_square=False)
      image1, image2, disp_small_up = padder.pad(image1, image2, disp_small_up)
      disp_small_up += padder._pad[0]
      init_disp = F.interpolate(disp_small_up, scale_factor=0.25, mode='bilinear', align_corners=True) * 0.25   # Init disp will be 1/4
      disp = self.forward(image1, image2, iters=iters, test_mode=test_mode, low_memory=low_memory, init_disp=init_disp)
      disp = padder.unpad(disp.float())
      return disp

