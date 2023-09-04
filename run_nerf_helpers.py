import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    # 只接受不定数量的key-value形式的参数
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        # d = 3 'input_dims'对应的value是3
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            # 把一个什么都不做的匿名函数添加到embed_fns
            embed_fns.append(lambda x : x)
            out_dim += d
        
        # 获取到最高频率
        max_freq = self.kwargs['max_freq_log2']
        # 获取频率总数
        N_freqs = self.kwargs['num_freqs']
        # 产生一系列频率
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        # 将所有频率下的三角函数都保存在embed_fns
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
        
        # 位置编码的所有三角函数 存储在列表中
        self.embed_fns = embed_fns
        # 编码后的维数
        self.out_dim = out_dim
    
    # 位置编码
    def embed(self, inputs):
        # inputs就是原始位置坐标数据
        # fn遍历了embed_fns中的所有函数 完成了对原始输入的高低频编码
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

# Set configuration of positional coder
def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    # 传入到Embedder类实例的key-value参数
    embed_kwargs = {
                'include_input' : True,                     # 若为True，则PosEncode后的结果包括了原始的位置坐标
                'input_dims' : 3,                           # 输入到PosEncode的原始位置坐标维数
                'max_freq_log2' : multires-1,               # 编码器产生的最大频率 就是论文中的2^(L-1)  
                'num_freqs' : multires,                     # 编码器总共能产生出L个频率的位置坐标数据
                'log_sampling' : True,                      # 位置编码
                'periodic_fns' : [torch.sin, torch.cos],    # 产生高频数据的函数
    }
    
    # 只传入key-value参数
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
# NeRF类继承自torch.nn.Module类的  因此NeRF类也继承了相应的基本方法
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D                              # 网络层数
        self.W = W                              # 每一层的神经元个数
        self.input_ch = input_ch                # 5D coordinate 位置坐标 [60,1]
        self.input_ch_views = input_ch_views    # 5D coordinate 视角方向 [24,1]
        self.skips = skips                      # 网络第五层再次输入位置坐标
        self.use_viewdirs = use_viewdirs

        #################################### 构建MLP ####################################
        # 网络结构：
        #                                      60(pos)              sigma  
        #                                        +                    ^(ReLU)                      
        # 60(pos) -> 256-> 256 -> 256 -> 256 -> 256 -> 256 -> 256 -> 256 ->(no activation) 256 -> 128 ->(sigmoid) RGB
        #                                                                                   +
        #                                                                                  24
        #
        # 位置坐标输入 to 第一层(input_position,W)  
        # 在网络的第五层中除了feature map 还注入了位置坐标 第五层 to 第六层(input_position+W,W)
        # 1-2 2->3 3->4 4->5 6->7 7->8 层与层之间连接均为(W,W) 
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        # (视角方向输入 + 第九层feature map) to 最后一层(十层)(input_direction+W,W/2)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        # 加入视角方向的输入
        if use_viewdirs:
            # 第八层 to 第九层 注意这里的连接是没有激活函数的
            self.feature_linear = nn.Linear(W, W)
            # 第九层 to 体密度输出
            self.alpha_linear = nn.Linear(W, 1)
            # 第十层 to RGB颜色输出  注意这里的连接的激活函数是sigmoid
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        # input_pts = x[0:59]  即经过postional encode后的位置向量 
        # input_views = x[60:84] 即经过postional encode后的方向向量
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        # 拿到位置向量
        h = input_pts

        #################################### 前向传递 ####################################
        # 这里的for循环将输入一直前向传递到了第八层网络
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            # 第五层时的输出要inject输入的位置向量 传给第六层
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            # 将方向向量inject到网络第九层中
            h = torch.cat([feature, input_views], -1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            rgb = self.rgb_linear(h)
            #最终得到RGBsigma
            outputs = torch.cat([rgb, alpha], -1)
        else:
            #
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



############### 生成一个视角下每个像素对应的光线方向 ###############
# parameters:
#       H: 该视角下图像的高
#       W：该视角下图像的宽
#       K：相机内参矩阵
#       c2w: 该视角下的相机位姿
# returns:
#       rays_o: 世界坐标系下光线的起点(相机光新)
#       rays_d: 世界坐标系下光线的方向
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

############### 生成一个视角下每个像素对应的光线方向 ###############
# parameters:
#       H: 该视角下图像的高
#       W：该视角下图像的宽
#       K：相机内参矩阵
#       c2w: 该视角下的相机位姿
# returns:
#       rays_o: 世界坐标系下光线的起点(相机光新)
#       rays_d: 世界坐标系下光线的方向
def get_rays_np(H, W, K, c2w):
    # i,j得到图片的x,y坐标序列
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')

    # 得到光线方向(从相机光心出发到成像平面像素方向向量 并归一化为单位向量)
    # 图像坐标:[i,j] ——> 成像平面坐标[i-cx, j-cy] ——> 相机坐标系中像素的空间点坐标[i-cx,j-cy,f] ——> 归一化平面坐标[(i-cx)/f,(j-cy)/f,1]
    # NeRF相机坐标系中 x轴朝右 y轴朝上 z轴朝里 
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    
    # Rotate ray directions from camera frame to the world frame
    # 将光线方向转换到世界坐标系下
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # 将相机光心的空间坐标转换到世界坐标系下 也就是光线的起点
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))

    # 返回世界坐标系下光线起点 和光线方向
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    # 将权重归一化到[0,1]，这样就可以理解为概率了
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    # 累加概率密度分布得到分布函数
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))
  
    # 均匀采样
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        # 在[0,1]上的均匀采样 u的大小为[batch_rays_size,N_samples]
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
