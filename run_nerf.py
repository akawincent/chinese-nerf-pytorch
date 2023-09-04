import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

# 调用GPU资源使用CUDA计算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False



############### fn的mini-batch实现 ###############
# parameters: 
# 		fn: 网络
# 		chunk: mini-batch大小
# return:	
# 		ret: 处理mini-batch形式输入的fn函数
def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret



############### NeRF网络的前向处理 ###############
# parameters:
# 		inputs: 样本的原始位置向量
#       viewdirs: 样本的原始方向向量
#       fn: 网络
#       embed_fn: 对原始位置向量进行positional encode的函数
#       embeddirs_fn: 对原始方向向量进行positional encode的函数
# returns:
#		outputs： 网络fn的前向处理输出
def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    # 对原始位置向量进行编码
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)
    # 对原始方向向量进行编码
    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        # 将编码后的位置向量和方向向量拼接在一起
        embedded = torch.cat([embedded, embedded_dirs], -1)

    # 经过网络处理此时的output还是inflatten形式的
    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs



################# 体渲染的并行处理 #################
# parameters:
# 		rays_flat: 光线起点 光线方向 光线近端 光线远端 视角方向
# 		chunk: 并行处理数量
# 		**kwargs: 训练时的配置
# returns；
# 		all_ret: 所有batch的光线渲染后的结果
def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        # 对光线ray_flat执行具体的渲染后得到的结果
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    # 得到所有batch的光线渲染后的结果
    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret



##################### 体渲染 #####################
# parameters:
# 		H: 图像的高
# 		W: 图像的宽
# 		K: 相机内参矩阵
# 		chunk: 并行处理rays的数量
# 		ray: 光线
# 		c2w: 视角的相机位姿
# 		ndc: 是否使用NDC坐标
# 		near: 光线近端
# 		far: 光线远端
# 		use_viewdirs: 是否使用方向向量
# 		c2w_staticcam: 固定的相机位姿来生成对应的光线
# 		**kwargs: 训练时的配置
# returns:
#       rgb_map: [batch_size, 3]. Predicted RGB values for rays.
#       disp_map: [batch_size]. Disparity map. Inverse of depth.
#       acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
#       extras: dict with everything returned by render_rays().
def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    
	# 如果给定了新的c2w序列那就重新生成光线
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    # 使用已经生成过的光线
    else:
        # use provided ray batch
        rays_o, rays_d = rays
	
	# 输入中包含光线方向向量
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        # 使用c2w_staticcam来生成光线 但是原先的视角观察方向的光线仍然保留
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

	# 生成光线的远近端，设置渲染的边界框
	# 使用默认的参数 near=0 far=1
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    # 光线起点 光线方向 光线远端 光线近端放在一起
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    # 若使用视角方向 则将视角方向也加入到rays
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]



def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps



############### 创建NeRF网络和PositionalEncoder并配置训练和测试时的关键参数 ###############
# parameters:
#		args: 关于数据集的参数
# returns: 
#		render_kwargs_train: 训练的设置  起始点 模型参数 优化器 
# 		render_kwargs_test: 测试的设置
# 		start: 训练开始时当前迭代的epoch数
# 		grad_vars: MLP网络参数(权重和偏置)的梯度 
# 		optimizer: 训练时的优化器
def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    # embed_fn是对postion向量进行编码的函数   input_ch是经过编码后的position向量维数 
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    # embeddirs_fn是对direction向量进行编码的函数 input_ch_views是讲过编码后的direction向量维数
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    # 代码作者说这语句其实没什么用  output_ch就是4
    output_ch = 5 if args.N_importance > 0 else 4
    # 在第五层网络上inject 位置输入  因此索引从0开始 所以是4
    skips = [4]
    # 创建coarse nerf network
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    
    # 获取到模型中每一层的权重和偏置参数
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        # 创建fine nerf netowek 
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        # 把fine network的权重和偏置和coarse的放在一起
        grad_vars += list(model_fine.parameters())

    # 创建了一个匿名函数 将位置向量inputs和方向向量viewdirs进行positionalcode处理再送入网络fn中
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
	# 训练开始时当前迭代的epoch数
    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # 加载先前的训练权重
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ########################## 训练时的设置 ##########################
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,	# 运行网络模型的函数
        'perturb' : args.perturb,				# 是否有抖动
        'N_importance' : args.N_importance,		# 精细网络中光线上的采样个数
        'network_fine' : model_fine,			# fNeRF的精细网络模型
        'N_samples' : args.N_samples,			# 粗网络中光线上的采样个数
        'network_fn' : model,					# NeRF的粗网络模型
        'use_viewdirs' : args.use_viewdirs,		# 是否使用方向向量
        'white_bkgd' : args.white_bkgd,			# 背景是否为白色
        'raw_noise_std' : args.raw_noise_std,	# 添加在输出的sigama上的噪声标准差
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    ########################## 测试时的设置 ##########################
    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    # 最终返回训练的设置 测试的设置 起始点 模型参数的梯度 优化器
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer



##################### 对采样的离散空间点进行体渲染的积分操作 #####################
# parameters:
# 		raw: 采样空间点的RGB和体密度
# 		z_vals: 采样的等差间隔
# 		rays_d: 光线方向
# 		raw_noise_std: 输出体密度时加入的噪声方差
# returns:
#		rgb_map: 返回最终渲染得到的图像RGB
# 		disp_map: 返回最终渲染得到的视差图
# 		acc_map: 累加的权重
# 		weights: 光线上每个采样点对应的权重  
# 		depth_map: 返回最终渲染得到的深度图
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    
	# 定义一个匿名函数 返回论文中离散体渲染公式中的 1-exp(-sigma * delta) delta是空间点的采样间隔距离
	# 代码中的raw就是论文中的体密度sigma 代码中的dists就是论文中的采样间隔距离delta
	# 论文中提到sigma不能非负,sigma需要经过ReLU处理
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    
	# 计算相机坐标系Z轴上采样点之间的距离
    dists = z_vals[...,1:] - z_vals[...,:-1]
    # 将光线采样点在相机坐标系Z轴上的间隔距离换算到世界坐标系下
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    
	# 论文中提到输出最终的三维点rgb需要经过sigmoid处理
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        # 生成以raw_noise_std为噪声方差的高斯噪声
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

	# alpha就是 1-exp(-sigma * delta) 这里的sigma已经加入了高斯噪声
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]

	# weights就是论文中提到的体渲染离散形式的权重  weights即为 T * (1-exp(-sigma * delta))
	# T就是由torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]计算得来
	# torch.cumprod的操作 [1,x1,x2,x3,x4,x5] ——> [1, x1, x1*x2, x1*x2*x3, x1*x2*x3*x4, x1*x2*x3*x4*x5]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    
	# 最终得到了渲染后到图像上的RGB值
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
	# 得到深度图
    depth_map = torch.sum(weights * z_vals, -1)
    # 得到视差图
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    # 累加权重
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map



##################### 在光线上进行分层采样策略并执行体渲染的具体数学过程 #####################
# parameters:
# 		ray_batch: 一个batch中光线的相关信息,其中包含光线起点 光线方向 视角方向  光线远端 光线近端
# 		network_fn: 粗网络
# 		network_quert_fn: 执行网络前向传递操作的匿名函数
# 		N_samples: 每条rays上采样的coarse样本数量
#       N_importance: 每条rays上采样的fine样本数量
#       network_fine: 精细网络
#       white_bkgd: 数据集图像是否为白色背景
#       raw_noise_std: 加入在体密度上高斯噪声的方差
# returns:
#       ret: rgb_map: 精细网络得到的空间点RGB sigma最终渲染输出的图像 
#            rgb0: 粗网络得到的空间点RGB sigma渲染输出的图像
def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    
	# 从ray_batch中分离出这一个batch的光线起点 光线方向 视角方向 光线远端 光线近端
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    # near = 0, far = 1
    near, far = bounds[...,0], bounds[...,1] # [-1,1]
    
	# 在[0,1]内生成N_samples个等差点
    t_vals = torch.linspace(0., 1., steps=N_samples)
    
    # 在深度图上采样 
    if not lindisp:
        # 相机坐标的z轴[0,1]这个范围采样N_samples个点
        z_vals = near * (1.-t_vals) + far * (t_vals)
    # 在视差图上采样
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
	# 扩张z_vals到batch中的每一条光线上 数据维度[N_rays,N_samples]
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

	# 空间中的点:光线原点 + 光线方向*距离
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

#     raw = run_network(pts)
	# 将采样的空间点和方向向量输入到粗网络中进行前向处理 得到每个空间点的density以及不同观察方向的RGB值
    raw = network_query_fn(pts, viewdirs, network_fn)
    # 对粗网络的离散点的RGB和体密度进行体渲染得到图像上的颜色
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

	# 分层采样中的细采样环节
    if N_importance > 0:
        # 保存粗网络的渲染结果
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        # 新的采样点 在相机坐标系z轴上的分布距离
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()
        # 将粗网络的采样点和精细网络的采样点合并在一起
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        # 新的空间点(世界坐标系)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        # 运行精细网络的前向处理 得到了每个空间点的RGB和体密度
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        # 对精细网络的离散点的RGB和体密度进行体渲染得到图像上的颜色
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        # 计算每条射线上采样点分布的方差
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")
    # 返回结果
    return ret



############### 终端命令行参数说明 ###############
# parameters:
# 		None
# returns:
#		parser: 参数解释器
def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    # 指定config文件生成路径
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    # 指定实验的名字
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    # 指定实验结果的输出路径
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    # 指定要使用的数据集的目录
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    ###################### training options ######################
	# 指定网络的层数
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    # 指定每层网络中有多少神经元
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    # 指定fine network的网络层数
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    # 指定fine netowrk中每层网络中的神经元个数
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    # 设定mini-batch大小，这里的batch是每次梯度下降时用于计算loss的随机射线数量
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    # 设置学习率
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    # 设定在1000次迭代中的指数学习率衰减
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    # 并行处理射线的数量  一个chunk的射线同时处理
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    # 并行处理的空间点数量  一个chunk的空间点同时处理
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    # 设定每次只从一张图像中随机选取射线
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    # 设定不从保存的模型ckpt(训练迭代一定次数后checkpoint)文件中载入权重
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    # 为coarse network载入权重文件
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    ###################### rendering options ######################
    # 每条ray中采样的coarse样本数量
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    # 每条ray中采样的fine样本数量
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    # 设为0是没有抖动 1为有抖动
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    # 使用视角+位置的5D coordinate输入
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    # 设定是否使用positionalcode操作
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    # 设定对position向量进行positionalcode的最大频率
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    # 设定对direction向量进行positionalcode的最大频率
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    # 设置添加在输出的sigma上的噪声标准差 为了作标准化
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    # 在训练集上只进行渲染 不训练 
    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    # 对测试集进行渲染
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    # 设置渲染时的下采样因子 减少采样加快速度
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    ###################### training options ######################
    # 训练次数
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops') 
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    ###################### dataset options ######################
    # 数据类型包括llff\blender\deepvoxels
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    # 测试集与验证集的数据比例 1：N
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    # 选择deepvoxels中的某个物体
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    # 设置在白色背景中进行渲染
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    # 设置载入的图像大小为400x400
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
	# 设置llff数据集中图像的下采样因子
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    # 设置是否使用标准化坐标系
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    # 基于视差图采样而不是深度图
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    # 360度场景进行渲染
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    # 在llff数据集中每N个图像采用1个图像进行测试，默认N=8
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    ###################### logging/saving options ######################
    # 在终端上打印训练指标的输出频率 默认每100次打印一次
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    # tensorboard图像的记录频率
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    # 每10000次保存一次网络的权重 checkpoint
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    # 每50000次保存测试集结果
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    # 每50000次保存渲染食品
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser



############### 训练NeRF网络 ###############
def train():
    
	# 获取终端中命令行参数
    parser = config_parser()
    args = parser.parse_args()

    # 加载数据集
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
        
	# 当数据集为blender
    elif args.dataset_type == 'blender':
        
        # images是数据集中的图片 [N,H,W,3] N = 训练集、测试集和验证集的总数 
		# pose是数据集中的位姿 [N,H,W,3]
		# render_poses是新视角的姿态
		# hwf分别是图片的高 宽 焦距
		# i_split 分割索引
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        
		# i_train 训练集的索引序列
		# i_val 验证集的索引序列
		# i_test 测试集的索引序列
        i_train, i_val, i_test = i_split

		# 确定渲染时的远端和近端
        near = 2.
        far = 6.

		# 将RGBA图像转换成RGB图像
        if args.white_bkgd:
            # 使用白色背景
            # 将RGBA原图绘制在背景上的公式为：原图RGB * A + 背景RGB * (1-A)
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # 将图像的宽和高转换成整型
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # 构建相机内参矩阵
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

	# 如果render_test为真 则新视角渲染用的相机位姿使用测试集中的而不是用pose_spherical(在load_blender.py中)生成的
    if args.render_test:
        render_poses = np.array(poses[i_test])

    # 创建训练日志，并复制config.txt文件
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # 创建NeRF模型 得到训练的设置 测试的设置 训练开始时当前迭代的epoch数 模型参数 优化器
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    # 训练时当前迭代到的步数
    global_step = start
    # 将渲染的距离加入到训练和测试的配置
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
	# 网络模型训练好后只进行渲染 得到渲染视频
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    #################### 生成每张图像下的每个像素对应的光线起点和光线方向 ####################
	# N_rand: 随机选取射线mini-batch的大小
    N_rand = args.N_rand
    # 是否使用mini-batch
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        # 获取到每个视角下的每个像素发出来的光线 
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        # 打乱光线的顺序
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        # mini-batch的起始索引
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

	# 训练迭代次数
    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
	# 本次训练开始于迭代中的第start步
    start = start + 1
    
	#################### 开始训练 ####################
    for i in trange(start, N_iters):
        time0 = time.time()

        # 如果使用了mini-batch
        if use_batching:
            # 按照batch大小加载光线
			# batch: [batchsize,3(ro,rd,imgrgb),3(对于ro,rd来说是向量,对于imgrgb是三通道)]
            batch = rays_rgb[i_batch:i_batch+N_rand]
            # batch；[3,batchsize,3]
            batch = torch.transpose(batch, 0, 1)
            # 将光线和图像rgb值分开 
			# batch_rays:[2,batchsize,3]
			# target_s:[1,batchsize,3]
            batch_rays, target_s = batch[:2], batch[2]

			# 更新batch的索引起点
            i_batch += N_rand
            # 如果下一次选取的batch超过了光线的范围 这以为这所有的光线都遍历过一次了
            if i_batch >= rays_rgb.shape[0]:
                # 重新调整下一次选取batch的索引起点
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
                
		# 如果没有使用mini-batch
        else:
            # Random from one image
			# 从训练集中选取一张图片
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            # 从训练集中选取图片对应的视角
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                # 生成选择出来的这张图片中所有像素点对应的光线起点和方向
				# get_rays 和 get_rays_np的处理一样
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

				############### 生成图像上每个像素的坐标 #############
                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)

				############# 像素坐标 光线起点 光线方向 像素RGB值 这些都是一一对应的 #############
                # 从所有的像素(Groundtruth)中选取N_rand大小的batch
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                # 得到这个batch的像素坐标
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                # 得到这个batch的光线起点
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                # 得到这个batch的光线方向
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                # 将光线起点和光线方向组合在一起
                batch_rays = torch.stack([rays_o, rays_d], 0)
                # 得到这个batch的像素RGB值 Groundtruth
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        # 渲染！
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        # 计算渲染得到的rgb与训练集target_s Grondtruth之间的损失
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        # 把粗网络的损失也加入到损失列表中
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        # 同时优化粗网络和精细网络
        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        # 调整Adam优化算法的参数
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """
        # 迭代步数前进一次
        global_step += 1



if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
