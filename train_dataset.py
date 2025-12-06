
import os
import torch
import time
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
# SUMO
from torch.utils.data import Dataset, DataLoader
import random
from typing import Dict
import numpy as np
from utils.camera_utils import cameraList_from_camInfos
def custom_collate_fn(batch):
    """
    自定义整理函数，用于处理批次大小为 1 且内容为自定义对象列表的情况。
    
    当 batch_size=1 时，输入的 batch 结构通常是：
    [[List[Camera objects]]]
    我们直接返回内部的 List[Camera objects]。
    """
    return batch[0]
  
class ColmapDataset(Dataset):
    def __init__(self,
                 cameras_info,
                 resolution,
                 resolution_scale,args):
        self.cameras_info_dataset=list(self.shuffle_camera_dict(cameras_info).values())
        self.resolution=resolution
        self.resolution_scale=resolution_scale
        self.args=args
    

    def shuffle_camera_dict(self,camera_dict):
        """
        打乱一个 {key: [CameraInfo, ...]} 格式的 dict，
        保持 key 和 list 长度不变，但随机重组 camerainfo 分配。
        """
        # 1. 收集所有 CameraInfo
        all_items = []
        for lst in camera_dict.values():
            all_items.extend(lst)

        # 2. 打乱整体列表
        random.shuffle(all_items)

        # 3. 按原 list 长度重新分配
        result = {}
        start = 0
        for key, lst in camera_dict.items():
            length = len(lst)
            result[key] = all_items[start:start + length]
            start += length

        return result


    def __getitem__(self, idx):
        cams_info=self.cameras_info_dataset[idx]
        train_cameras= cameraList_from_camInfos(cams_info, self.resolution_scale, self.args, False, False)
        return train_cameras
    
    def __len__(self):
        
        return len(self.cameras_info_dataset)

def need_to_double_training(
    losses_data: Dict[str, float], 
    target_frame_id: str, 
    top_percent: float = 0.2
) -> bool:

    
    # 1. 提取所有 Loss 值并计算均值
    # 注意：losses_data.values() 似乎已经是 float
    all_values = np.array(list(losses_data.values()))
    if len(all_values) == 0:
        return False
        
    mean_value = np.mean(all_values)

    # 2. 计算每个样本与均值的绝对距离（方差贡献度）
    contributions = {
        view_id: abs(loss - mean_value) 
        for view_id, loss in losses_data.items()
    }

    # 3. 确定需要筛选的数量 (N%)
    total_count = len(contributions)
    num_to_select = max(1, int(total_count * top_percent))

    # 4. 获取贡献度的值列表，并降序排序
    sorted_contributions = sorted(contributions.values(), reverse=True)
    
    # 5. 确定前 N% 样本中最小的贡献度阈值
    # 只要目标贡献度大于或等于这个阈值，它就在前 N%
    threshold_contribution = sorted_contributions[num_to_select - 1]
    
    # 6. 获取目标视角的贡献度并进行比较
    target_contribution = contributions.get(target_frame_id)
    
    if target_contribution is None:
        # 如果目标 ID 不在当前记录中，则不加倍训练
        return False

    return target_contribution >= threshold_contribution

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=int(opt.iterations)*int(opt.epochs))

    # ----------------------------------------------------------------------
    # 🌟 修改 1: 移除手动 frame_stack 逻辑，创建 DataLoader
    
    # frame_stack = scene.train_cameras_info.copy() # <--- 移除
    # frame_indices=list(range(len(frame_stack))) # <--- 移除

    # 1. 准备数据集信息 (将字典值转换为列表)
    frame_cameras_list = scene.train_cameras_info
    
    # 2. 初始化 Dataset (假设 scene.args 可用)
    train_dataset = ColmapDataset(
        cameras_info=frame_cameras_list,
        resolution=dataset.resolution,
        resolution_scale=1.0,
        args=scene.args
    )

    # 3. 初始化 DataLoader (batch_size=1, 每次加载一帧/一组相机)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1, 
        shuffle=True, 
        num_workers=4, 
        # pin_memory=True ,
        collate_fn=custom_collate_fn
    )
    # ----------------------------------------------------------------------
   
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # WDD [2024-07-31] [为GUI动态播放初始化时间和时间索引]
    last_time_update = time.time()
    current_time_idx = 0
    # frame_count 现在从 DataLoader 中获取长度
    frame_count = len(train_dataset)
        
    global_iteration = 0
    first_iter+=1

    all_frame_loss={}

    print("Starting training")
    
    # ----------------------------------------------------------------------
    # 🌟 修改 2: 替换外层循环的数据加载逻辑

    for epoch in range(0, opt.epochs):
        
        # 🌟 替换：使用 DataLoader 迭代所有帧 (相机组)
        for frames_cams_batch in train_dataloader:
            st=time.time()

            # 原始代码中的加载和转移到 CUDA 逻辑
            # frames_cams=scene.load_cameras(frames_cams_info) # <--- 移除
            viewpoint_cams= [cam.cuda() for cam in frames_cams_batch]
            print(f"load time: {time.time()-st}s")
            # 确定当前帧的时间索引
            current_frame_time_idx = viewpoint_cams[0].time_idx if viewpoint_cams else None
            
            random.shuffle(viewpoint_cams)
            viewpoint_stack = viewpoint_cams.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
            
            # if global_iteration< opt.densify_until_iter:
            #     base_iterations=opt.iterations//10 +1
            # else:
            #     base_iterations=opt.iterations + 1
            base_iterations=opt.iterations + 1
            # SUMO 用于存储每一个视角的loss
            images_loss={}
            
            # loss大的视角加倍训练
            # if epoch > 1: 
            #     if need_to_double_training(all_frame_loss, target_frame_id=current_frame_time_idx, top_percent=0.1):
            #         print(f"current_frame_time_idx: {current_frame_time_idx}, need to double training")
            #         base_iterations=(opt.iterations + 1)*2

            progress_bar = tqdm(range(first_iter, base_iterations-1), desc="Training progress")
            
            # ----------------------------------------------------------------------
            # 🌟 CAMERA LOOP (内层循环保持不变)
            for iteration in range(first_iter, base_iterations):

                if network_gui.conn == None:
                    network_gui.try_connect()
                while network_gui.conn != None:
                    try:
                        net_image_bytes = None
                        custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                        if custom_cam != None:
                            # WDD [2024-07-31] [实现GUI中每0.2秒自动切换时间索引，以循环播放]
                            if time.time() - last_time_update > 0.2:
                                last_time_update = time.time()
                                current_time_idx = (current_time_idx + 1) % frame_count
                            custom_cam.time_idx = current_time_idx

                            net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                        network_gui.send(net_image_bytes, dataset.source_path)
                        if do_training and ((global_iteration < int(opt.iterations)*int(opt.epochs)*frame_count) or not keep_alive):
                            break
                    except Exception as e:
                        network_gui.conn = None
                

                global_iteration+=1
                if not viewpoint_stack:
                    viewpoint_stack =viewpoint_cams.copy()
                    viewpoint_indices = list(range(len(viewpoint_stack)))
                rand_idx = randint(0, len(viewpoint_indices) - 1)
                viewpoint_cam = viewpoint_stack.pop(rand_idx)
                vind = viewpoint_indices.pop(rand_idx)

                iter_start.record()

                gaussians.update_learning_rate(global_iteration)

                # Every 1000 its we increase the levels of SH up to a maximum degree
                if global_iteration % 1000 == 0:
                    gaussians.oneupSHdegree()

                # Render
                if (global_iteration - 1) == debug_from:
                    pipe.debug = True

                bg = torch.rand((3), device="cuda") if opt.random_background else background

                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                if viewpoint_cam.alpha_mask is not None:
                    alpha_mask = viewpoint_cam.alpha_mask.cuda()
                    image *= alpha_mask

                # Loss
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                if FUSED_SSIM_AVAILABLE:
                    ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                else:
                    ssim_value = ssim(image, gt_image)

                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

                # Depth regularization
                Ll1depth_pure = 0.0
                if depth_l1_weight(global_iteration) > 0 and viewpoint_cam.depth_reliable:
                    invDepth = render_pkg["depth"]
                    mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                    depth_mask = viewpoint_cam.depth_mask.cuda()

                    Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
                    Ll1depth = depth_l1_weight(global_iteration) * Ll1depth_pure 
                    loss += Ll1depth
                    Ll1depth = Ll1depth.item()
                else:
                    Ll1depth = 0

                # SUMO
                # images_loss[viewpoint_cam.image_name] = loss.item()

                loss.backward()

                iter_end.record()

                with torch.no_grad():
                    # Progress bar
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

                    if global_iteration % 10 == 0:
                        # WDD [2024-08-01] [在进度条上显示当前的总高斯点数]
                        postfix_dict = {"Loss": f"{ema_loss_for_log:.{7}f}","iter":f"{global_iteration}","Points": f"{gaussians.get_xyz.shape[0]}"}
                        progress_bar.set_postfix(postfix_dict)
                        progress_bar.update(10)
                    if iteration == base_iterations-1:
                        progress_bar.close()

                    # Log and save
                    # training_report(tb_writer, global_iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
                    if (global_iteration in saving_iterations):
                        print("\n[ITER {}] Saving Gaussians".format(global_iteration))
                        scene.save(global_iteration)

                    # Densification
                    if global_iteration < opt.densify_until_iter:
                    # if epoch<2: #SUMO 稠密化2轮
                        # Keep track of max radii in image-space for pruning
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if global_iteration > opt.densify_from_iter and global_iteration % opt.densification_interval == 0:
                            size_threshold = 20 if global_iteration > opt.opacity_reset_interval else None
                            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                        
                        if global_iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and global_iteration == opt.densify_from_iter):
                            gaussians.reset_opacity()

                    # Optimizer step
                    if global_iteration < opt.iterations*int(opt.epochs)*frame_count:
                        gaussians.exposure_optimizer.step()
                        gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                        if use_sparse_adam:
                            visible = radii > 0
                            gaussians.optimizer.step(visible, radii.shape[0])
                            gaussians.optimizer.zero_grad(set_to_none = True)
                        else:
                            gaussians.optimizer.step()
                            gaussians.optimizer.zero_grad(set_to_none = True)

                    if (global_iteration in checkpoint_iterations):
                        print("\n[ITER {}] Saving Checkpoint".format(global_iteration))
                        torch.save((gaussians.capture(), global_iteration), scene.model_path + "/chkpnt" +str(global_iteration) + ".pth")

            # SUMO 记录这一帧的最大loss
            # if viewpoint_cams:
            #     current_time_idx_loss=max(images_loss.values())
            #     all_frame_loss[viewpoint_cams[0].time_idx]=current_time_idx_loss
            
            # 确保下一帧 (下一个 batch) 的内部迭代从 0 开始
            first_iter = 0

        
        print("\n[ITER {}] Saving Gaussians".format(global_iteration))
        scene.save(global_iteration)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
