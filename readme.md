# Chinese-NeRF-Pytorch

> 本项目是对nerf-pytorch实现的详细中文注释

## 1.示例

```Python
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
```