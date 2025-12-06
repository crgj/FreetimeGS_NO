#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration, mkdir_p
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.colmap_loader import read_points3D_binary
from utils.graphics_utils import BasicPointCloud
import numpy as np
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # WDD注释: 尝试从4DGS数据集加载场景信息
        if any(f.startswith("frame") and f[5:].isdigit() and os.path.isdir(os.path.join(args.source_path, f)) for f in os.listdir(args.source_path)): #WDD注释
            print("Found frame... folder, assuming 4DGS data set!") #WDD注释
            scene_info = sceneLoadTypeCallbacks["4DGS"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp) #WDD注释
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            # WDD [2024-08-07] [修复：处理 points3D.bin 的情况]
            # 检查 .ply 文件是否存在，如果不存在，则尝试从 .bin 文件加载
            ply_path = scene_info.ply_path
            bin_path = ply_path.replace(".ply", ".bin")
            if not os.path.exists(ply_path) and os.path.exists(bin_path):
                print(f"Could not find {ply_path}, attempting to load {bin_path}")
                xyz, rgb, _ = read_points3D_binary(bin_path)
                # COLMAP的RGB是0-255，需要归一化到0-1
                pcd = BasicPointCloud(points=xyz, colors=rgb / 255.0, normals=np.zeros_like(xyz))
                scene_info = scene_info._replace(point_cloud=pcd) # 使用 _replace 创建新的不可变对象
            else:
                with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                       "point_cloud",
                                                       "iteration_" + str(self.loaded_iter),
                                                       "point_cloud.ply"), args.train_test_exp)
        else:
            # 获取最大时间索引作为帧数
            frame_count = max(camera.time_idx for camera in scene_info.train_cameras) + 1
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent, frame_count)

    def save(self, iteration):
        # WDD [2024-08-02] [修改保存逻辑以支持按帧保存PLY]
        point_cloud_iter_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        mkdir_p(point_cloud_iter_path)
        # 获取总帧数，通过查找训练数据中最大的时间索引+1
        frame_count = max(camera.time_idx for camera in self.getTrainCameras()) + 1
        # 为每个时间帧保存一个ply文件
        for t in range(frame_count):
            ply_path = os.path.join(point_cloud_iter_path, f"point_cloud_t{t}.ply")
            # 将当前时间索引t传递给save_ply，以保存该帧的动态透明度
            self.gaussians.save_ply(ply_path, time_idx=t)
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
