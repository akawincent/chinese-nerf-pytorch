import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

## 到球心距离转换到变换矩阵
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

## 方位角phi转换到变换矩阵
rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

## 仰角theta转换到变换矩阵
rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

############### 生成用于新视角合成的相机位姿 ###############
# parameters:
#       theta: 仰角
#       phi: 方位角
#       radius: 球体半径 到球心的距离
# returns:
#       c2w: 生成的一系列相机位姿 共40个4x4变换矩阵
def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

############### 加载blender数据集 ###############
# parameters:
#       basedir: 数据集路径
#       half_res: 设置载入的图像大小 False为800x800 True为400x400
#       testskip: 测试集和验证集数据的比例 1:testskip
# returns:
#       imgs: 数据集中train val和test中的图像
#       poses: 数据集中trans_train trans_val和trans_test的相机位姿 与imgs中的图像一一对应
#       render_poses: 用于新视角合成的相机位姿
#       [H, W, focal]: 图像的高 宽 相机的焦距
#       i_split: 分割train val和test的索引
def load_blender_data(basedir, half_res=False, testskip=1):
    # 将数据集中的transforms json文件加载到metas
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    # 遍历transforms_train transforms_val transforms_test
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
        # 遍历frames 如果是训练集那就遍历每个frame 如果是测试集或是验证集那就每隔skip个遍历transform矩阵
        for frame in meta['frames'][::skip]:
            # 遍历frame对应的图像
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            # 将遍历的图像添加到imgs存储起来
            imgs.append(imageio.imread(fname))
            # 遍历frame对应的姿态并添加到poses存储起来
            poses.append(np.array(frame['transform_matrix']))
        # 将图像的像素值归一化到[0,1]
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        # 存储train test val的图像个数
        # [0,num_train,num_train+num_val,num_train+num_val+num_test]
        counts.append(counts[-1] + imgs.shape[0])
        # 存储train test val的图像
        all_imgs.append(imgs)
        # 存储train test val的姿态
        all_poses.append(poses)
    # 将train val 和 test分开的索引
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    # 将train val test的数据在0维拼接
    # imgs = [ num_train + val_train + num_test, H, W, 4 ]
    # poses = [ num_train + val_train + num_test, 4, 4 ]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    # 图像的宽和高
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    # 得到焦距
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    # 生成用于新视角合成的相机位姿  np.linspace(-180,180,40+1)[:-1] = [-180,-171,162,...,171]
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    # 如果是处理400x400的输入图像
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    # 返回数据集提供的图像 位姿 新视角合成的位姿势 [图像的高 图像的宽 焦距] 分割索引
    return imgs, poses, render_poses, [H, W, focal], i_split


