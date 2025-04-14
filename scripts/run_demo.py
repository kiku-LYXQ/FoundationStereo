# -*- coding: utf-8 -*-
# 版权声明 (Copyright notice)
# 版权所有 © 2025 NVIDIA CORPORATION。保留所有权利。

# 导入系统模块
import os
import sys
import time
import argparse
import logging
import imageio  # 用于图像读写
import cv2  # OpenCV库，用于图像处理
import numpy as np
import open3d as o3d  # 3D点云处理库
import torch
from omegaconf import OmegaConf  # 配置管理工具

# 添加项目根目录到系统路径
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

# 导入项目自定义模块
from core.utils.utils import InputPadder  # 输入填充工具类
from Utils import *  # 自定义工具函数
from core.foundation_stereo import *  # 核心立体匹配模型

if __name__ == "__main__":
    # -------------------------- 初始化设置 --------------------------
    code_dir = os.path.dirname(os.path.realpath(__file__))

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='FoundationStereo立体匹配演示程序')

    # 添加命令行参数及说明
    parser.add_argument('--left_file', default=f'{code_dir}/../assets/left.png', type=str,
                        help='左视图图像路径')
    parser.add_argument('--right_file', default=f'{code_dir}/../assets/right.png', type=str,
                        help='右视图图像路径')
    parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str,
                        help='相机内参矩阵文件路径，包含3x3内参矩阵和基线距离')
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str,
                        help='预训练模型检查点路径')
    parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str,
                        help='结果输出目录')
    parser.add_argument('--scale', default=1, type=float,
                        help='图像缩放因子(<=1)，用于下采样高分辨率图像')
    parser.add_argument('--hiera', default=0, type=int,
                        help='分层推理模式标志(0关闭/1开启)，用于处理超高分辨率图像(>1K)')
    parser.add_argument('--z_far', default=10, type=float,
                        help='点云最大有效深度值(米)，超过该值的点将被过滤')
    parser.add_argument('--valid_iters', type=int, default=32,
                        help='前向传播时流场更新的迭代次数')
    parser.add_argument('--get_pc', type=int, default=1,
                        help='是否保存点云输出(0不保存/1保存)')
    parser.add_argument('--remove_invisible', default=1, type=int,
                        help='移除左右视图不可见区域点云(0关闭/1开启)，提高点云可靠性')
    parser.add_argument('--denoise_cloud', type=int, default=1,
                        help='是否进行点云去噪(0关闭/1开启)')
    parser.add_argument('--denoise_nb_points', type=int, default=30,
                        help='半径离群点去除的邻域点数阈值')
    parser.add_argument('--denoise_radius', type=float, default=0.03,
                        help='半径离群点去除的搜索半径(米)')

    # 解析命令行参数
    args = parser.parse_args()

    # -------------------------- 环境配置 --------------------------
    set_logging_format()  # 设置日志格式
    set_seed(0)  # 固定随机种子保证可重复性
    torch.autograd.set_grad_enabled(False)  # 禁用梯度计算
    os.makedirs(args.out_dir, exist_ok=True)  # 创建输出目录

    # -------------------------- 模型加载 --------------------------
    # 将参数转换为OmegaConf配置对象
    cfg = OmegaConf.create(vars(args))
    logging.info(f"运行参数:\n{args}")
    logging.info(f"正在加载预训练模型: {args.ckpt_dir}")

    # 加载模型配置文件
    cfg_file = OmegaConf.load(f'{os.path.dirname(args.ckpt_dir)}/cfg.yaml')

    # 初始化立体匹配模型
    model = FoundationStereo(cfg_file)

    # 加载预训练权重
    ckpt = torch.load(args.ckpt_dir, weights_only=False)  # PyTorch 2.6 开始，torch.load 默认启用 weights_only=True
    logging.info(f"检查点信息 - 全局步数: {ckpt['global_step']}, 训练轮次: {ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])  # 加载模型参数

    # 模型部署到GPU并设为评估模式
    model.cuda()
    model.eval()

    # -------------------------- 数据预处理 --------------------------
    # 读取左右视图图像
    # todo: v3 和 v2 版本imread有区别，需要实验对比
    img0 = imageio.v3.imread(args.left_file)  # 左视图
    img1 = imageio.v3.imread(args.right_file)  # 右视图

    # 图像缩放处理
    scale = args.scale
    assert scale <= 1, "缩放因子必须小于等于1"
    img0 = cv2.resize(img0, None, fx=scale, fy=scale)  # 双线性插值缩放
    img1 = cv2.resize(img1, None, fx=scale, fy=scale)
    H, W = img0.shape[:2]  # 获取缩放后图像尺寸
    img0_ori = img0.copy()  # 保留原始图像用于可视化
    logging.info(f"左视图尺寸: {img0.shape}")

    # 将图像转换为PyTorch张量并传输到GPU
    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)  # 增加批次维度并调整通道顺序
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)

    # 输入填充处理（使尺寸符合模型要求）
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    infer_start = time.time()
    # -------------------------- 模型推理 --------------------------
    with torch.cuda.amp.autocast(True):  # 启用混合精度推理
        if not args.hiera:
            # 标准推理模式
            disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
        else:
            # 分层推理模式（处理超高分辨率）
            disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)

    infer_end = time.time()
    print(f"耗时: {infer_end - infer_start:.2f} 秒")

    # 后处理：移除填充并转换到CPU
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W)  # 转换为numpy数组

    # -------------------------- 结果可视化 --------------------------
    # 生成视差图可视化
    vis = vis_disparity(disp)  # 伪彩色映射
    vis = np.concatenate([img0_ori, vis], axis=1)  # 拼接原图和视差图
    imageio.imwrite(f'{args.out_dir}/vis.png', vis)  # 保存可视化结果
    logging.info(f"可视化结果已保存至: {args.out_dir}")

    # -------------------------- 点云生成 --------------------------
    if args.get_pc:
        # 加载相机内参和基线
        with open(args.intrinsic_file, 'r') as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].split()))).reshape(3, 3).astype(np.float32)
            baseline = float(lines[1])  # 基线距离（米）

        # 调整内参矩阵的缩放
        K[:2] *= scale

        # 计算深度图（基于视差和三角测量原理）
        depth = K[0, 0] * baseline / (disp + 1e-6)  # 避免除以零
        np.save(f'{args.out_dir}/depth_meter.npy', depth)  # 保存深度图

        # 生成XYZ点云图
        xyz_map = depth2xyzmap(depth, K)  # 坐标转换

        # 创建Open3D点云对象
        pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3))

        # 移除无效点（深度超出范围）
        keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (np.asarray(pcd.points)[:, 2] <= args.z_far)
        keep_ids = np.where(keep_mask)[0]
        pcd = pcd.select_by_index(keep_ids)

        # 保存原始点云
        o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
        logging.info(f"点云已保存至: {args.out_dir}")

        # 点云去噪处理
        if args.denoise_cloud:
            logging.info("正在进行点云去噪...")
            # 半径离群点去除算法
            cl, ind = pcd.remove_radius_outlier(
                nb_points=args.denoise_nb_points,  # 邻域最小点数阈值
                radius=args.denoise_radius  # 搜索半径
            )
            inlier_cloud = pcd.select_by_index(ind)  # 选择内点
            o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)
            pcd = inlier_cloud  # 更新点云

        # 可视化点云
        logging.info("正在可视化点云，按ESC退出...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='3D点云可视化')
        vis.add_geometry(pcd)

        # 设置可视化参数
        vis.get_render_option().point_size = 1.0  # 点大小
        vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])  # 灰色背景

        # 启动可视化循环
        vis.run()
        vis.destroy_window()