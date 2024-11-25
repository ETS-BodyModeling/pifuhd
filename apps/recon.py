# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json 
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from numpy.linalg import inv


from lib.options import BaseOptions
from lib.mesh_util import save_obj_mesh_with_color, reconstruction
from lib.data import EvalWPoseDataset, EvalDataset
from lib.model import HGPIFuNetwNML, HGPIFuMRNet
from lib.geometry import index

from PIL import Image
import trimesh
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
parser = BaseOptions()

def gen_mesh(res, net, cuda, data, save_path, thresh=0.5, use_octree=True, components=False):
    image_tensor_global = data['img_512'].to(device=cuda)
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    net.filter_global(image_tensor_global)
    net.filter_local(image_tensor[:,None])

    try:
        if net.netG.netF is not None:
            image_tensor_global = torch.cat([image_tensor_global, net.netG.nmlF], 0)
        if net.netG.netB is not None:
            image_tensor_global = torch.cat([image_tensor_global, net.netG.nmlB], 0)
    except:
        pass
    
    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor_global.shape[0]):
            save_img = (np.transpose(image_tensor_global[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        cv2.imwrite(save_img_path, save_img)

        verts, faces, _, _ = reconstruction(
            net, cuda, calib_tensor, res, b_min, b_max, thresh, use_octree=use_octree, num_samples=50000)
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        # if 'calib_world' in data:
        #     calib_world = data['calib_world'].numpy()[0]
        #     verts = np.matmul(np.concatenate([verts, np.ones_like(verts[:,:1])],1), inv(calib_world).T)[:,:3]

        color = np.zeros(verts.shape)
        interval = 50000
        for i in range(len(color) // interval + 1):
            left = i * interval
            if i == len(color) // interval:
                right = -1
            else:
                right = (i + 1) * interval
            net.calc_normal(verts_tensor[:, None, :, left:right], calib_tensor[:,None], calib_tensor)
            nml = net.nmls.detach().cpu().numpy()[0] * 0.5 + 0.5
            color[left:right] = nml.T

        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)


def gen_mesh_imgColor(res, net, cuda, data, save_path, thresh=0.5, use_octree=True, components=False):
    # transform = T.ToPILImage()
    # transform1 = T.ToTensor()
    
    image_tensor_global = data['img_512'].to(device=cuda)
    image_tensor = data['img'].to(device=cuda)
    image_tensor_rm = data['image_rm'].to(device=cuda)
    
    
    calib_tensor = data['calib'].to(device=cuda)

    net.filter_global(image_tensor_global)
    net.filter_local(image_tensor[:,None])

    try:
        if net.netG.netF is not None:
            image_tensor_global = torch.cat([image_tensor_global, net.netG.nmlF], 0)
        if net.netG.netB is not None:
            image_tensor_global = torch.cat([image_tensor_global, net.netG.nmlB], 0)
    except:
        pass
    
    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_pose_path=save_path[:-4] + '.npy'
        save_calib_path=save_path[:-4] + '_calib.npy'
        save_img_list = []
        for v in range(image_tensor_global.shape[0]):
            save_img = (np.transpose(image_tensor_global[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        cv2.imwrite(save_img_path, save_img)

        verts, faces, _, _ = reconstruction(
            net, cuda, calib_tensor, res, b_min, b_max, thresh, use_octree=use_octree, num_samples=100000)
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()

        # if this returns error, projection must be defined somewhere else
        xyz_tensor = net.projection(verts_tensor, calib_tensor[:1])
        # print('xyz_tensor',xyz_tensor,xyz_tensor.shape)
        uv = xyz_tensor[:, :2, :]
        
        xx=uv.detach().cpu().squeeze()
        yy=torch.zeros((1,xx.shape[1]))
        yy=torch.concat([xx,yy])
        yy=np.transpose(yy.numpy(),(1,0))
        # mesh_optim = trimesh.Trimesh(vertices=yy,faces=None)
        # mesh_optim.export('/content/uv.obj')
        
        joints_2d = data['keypoints']
        hand_left_keypoints = data['hand_left_keypoints']
        hand_right_keypoints = data['hand_right_keypoints']
        face_keypoints_2d=data['face_keypoints_2d']
        # print('///////////////////',joints_2d.shape,hand_right_keypoints.shape)
        target=np.concatenate((joints_2d, hand_left_keypoints, hand_right_keypoints,face_keypoints_2d),  axis=0)/512-1
        # target=joints_2d/512-1
        source=np.transpose(uv.detach().cpu().squeeze().numpy() ,(1,0)) 

        # neigh =NearestNeighbors(n_neighbors=10,radius=0.005)
        # neigh.fit(source)
        # _, indices = neigh.radius_neighbors(np.asarray(target))
        # # print('indices',indices)
        
        neigh = NearestNeighbors(n_neighbors=10)
        neigh.fit(source)
        _, indices = neigh.kneighbors(np.asarray(target), return_distance=True)
        
        # mesh_pose = trimesh.Trimesh(vertices=yy[indices],faces=None)
        # mesh_pose.export('/content/uv_pose.obj')
        
        
        
        
        val_temp=indices[0]
        for val in indices[1:]:
            val_temp=np.concatenate((val_temp, val), axis=0)
        val_temp=np.unique(val_temp)
        
        
        
        save_img1 = (np.transpose(image_tensor[:1][0].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        for y,x in joints_2d:
          save_img1[x-3:x+3, y-3:y+3, :] = np.array([161, 215, 106])

        
        for y,x in hand_left_keypoints:
          save_img1[x-3:x+3, y-3:y+3, :] = np.array([255, 0, 0])

        for y,x in hand_right_keypoints:
          save_img1[x-3:x+3, y-3:y+3, :] = np.array([0, 0, 255])         

        for y,x in face_keypoints_2d:
          save_img1[x-3:x+3, y-3:y+3, :] = np.array([0, 255, 255])     
        # save_img_512 = cv2.resize(save_img1, (512, 512))    
        # path = os.path.join("/content/output", os.path.basename(save_path)[:-4])
        # if not os.path.exists(path):
        #   os.mkdir(path)       
        # cv2.imwrite(path+'/'+ os.path.basename(save_path)[:-4] +'.png', save_img_512)
        # cv2.imwrite('/content/fares.png', save_img1)
        
        # print('uv', uv, uv.shape)
        
        # save_img1 = (np.transpose(image_tensor[:1][0].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        # cv2.imwrite('/content/fares.png', save_img1)
        
        # print('image_tensor', image_tensor[:1], image_tensor[:1].shape)
        color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T
        color = color * 0.5 + 0.5

        color1 = index(image_tensor_rm[:1], uv).detach().cpu().numpy()[0].T
        color1 = color1 * 0.5 + 0.5
        
        mask_c=color!=color1
        # mask_c = np.abs(color - color1)>0.01
        indices_c = np.nonzero(np.any(mask_c, axis=-1))
        color[indices_c]=[0,0,0]
        # print('color', color, color.shape)

        if 'calib_world' in data:
            calib_world = data['calib_world'].numpy()[0]
            verts = np.matmul(np.concatenate([verts, np.ones_like(verts[:,:1])],1), inv(calib_world).T)[:,:3]
            # np.save(save_calib_path, calib_world)
        
        # print('calib est = ',calib_tensor,' et calib world = {}',calib_world)

        prior_pose=[]
        for ii,ids in enumerate(indices):
          kmeans = KMeans(n_clusters=2, random_state=0).fit(verts[ids])
          if ii in(  [0,15,16,19,20,22,23]+ list(np.arange(67,137,1)) ):
              prior_pose.append(kmeans.cluster_centers_.max(0))
          elif ii in [21,24]:
              prior_pose.append(kmeans.cluster_centers_.min(0))
          else:
              prior_pose.append(kmeans.cluster_centers_.mean(0))
        prior_pose=np.stack(prior_pose)
        np.save(save_pose_path, prior_pose)
        
        # np.save('/content/utils/prior3d.npy', prior_pose)
            
        #   ///////////////////////////////
        
        xx=uv.detach().cpu().squeeze()
        yy=torch.zeros((1,xx.shape[1]))
        yy=torch.concat([xx,yy])
        yy=np.transpose(yy.numpy(),(1,0)) 

        mesh_optim = trimesh.Trimesh(vertices=yy,faces=None)
        # mesh_optim.export('/content/fares.obj')

        mesh_optim = trimesh.Trimesh(vertices=yy[val_temp],faces=None)
        # mesh_optim.export('/content/fares1.obj')
        
        mesh_optim = trimesh.Trimesh(vertices=verts[val_temp],faces=None)
        # mesh_optim.export('/content/fares2.obj')

        mesh_optim = trimesh.Trimesh(vertices=prior_pose,faces=None)
        # mesh_optim.export('/content/fares3.obj')
        #   ///////////////////////////////

        save_obj_mesh_with_color(save_path, verts, faces, color)
        save_obj_mesh_with_color(save_path[:-4] + '-rm.obj', verts, faces, color1)

    except Exception as e:
        print(e)


def recon(opt, use_rect=False):
    # load checkpoints
    state_dict_path = None
    if opt.load_netMR_checkpoint_path is not None:
        state_dict_path = opt.load_netMR_checkpoint_path
    elif opt.resume_epoch < 0:
        state_dict_path = '%s/%s_train_latest' % (opt.checkpoints_path, opt.name)
        opt.resume_epoch = 0
    else:
        state_dict_path = '%s/%s_train_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
    # print('state_dict_path', state_dict_path)
    start_id = opt.start_id
    end_id = opt.end_id

    cuda = torch.device('cuda:%d' % opt.gpu_id if torch.cuda.is_available() else 'cpu')

    state_dict = None
    if state_dict_path is not None and os.path.exists(state_dict_path):
        # print('Resuming from ', state_dict_path)
        state_dict = torch.load(state_dict_path, map_location=cuda)    
        print('Warning: opt is overwritten.')
        dataroot = opt.dataroot
        resolution = opt.resolution
        results_path = opt.results_path
        loadSize = opt.loadSize
        
        opt = state_dict['opt']
        opt.dataroot = dataroot
        opt.resolution = resolution
        opt.results_path = results_path
        opt.loadSize = loadSize
    else:
        raise Exception('failed loading state dict!', state_dict_path)
    
    # parser.print_options(opt)

    if use_rect:
        test_dataset = EvalDataset(opt)
    else:
        test_dataset = EvalWPoseDataset(opt)

    # print('test data size: ', len(test_dataset))
    projection_mode = test_dataset.projection_mode

    opt_netG = state_dict['opt_netG']
    netG = HGPIFuNetwNML(opt_netG, projection_mode).to(device=cuda)
    netMR = HGPIFuMRNet(opt, netG, projection_mode).to(device=cuda)

    def set_eval():
        netG.eval()

    # load checkpoints
    netMR.load_state_dict(state_dict['model_state_dict'])

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s/recon' % (opt.results_path, opt.name), exist_ok=True)

    if start_id < 0:
        start_id = 0
    if end_id < 0:
        end_id = len(test_dataset)

    ## test
    with torch.no_grad():
        set_eval()

        print('generate mesh (test) ...')
        for i in tqdm(range(start_id, end_id)):
            if i >= len(test_dataset):
                break
            
            # for multi-person processing, set it to False
            if True:
                test_data = test_dataset[i]
                # print(test_data['img'].shape , test_data['img_512'].shape , test_data['calib'].shape , test_data['calib_world'].shape)

                save_path = '%s/%s/recon/result_%s_%d.obj' % (opt.results_path, opt.name, test_data['name'], opt.resolution)

                # print(save_path)
                gen_mesh_imgColor(opt.resolution, netMR, cuda, test_data, save_path, components=opt.use_compose)
            else:
                for j in range(test_dataset.get_n_person(i)):
                    test_dataset.person_id = j
                    test_data = test_dataset[i]
                    save_path = '%s/%s/recon/result_%s_%d.obj' % (opt.results_path, opt.name, test_data['name'], j)
                    # print(test_data,test_data.shape)
                    gen_mesh_imgColor(opt.resolution, netMR, cuda, test_data, save_path, components=opt.use_compose)

def reconWrapper(args=None, use_rect=False):
    opt = parser.parse(args)
    recon(opt, use_rect)

if __name__ == '__main__':
    reconWrapper()
  
