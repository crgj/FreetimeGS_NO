# WDD [2024-08-09] [创建第二阶段训练脚本，用于优化运动模型和进行竞争性剪枝]
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams

def training_motion(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, start_checkpoint, debug_from):
    # WDD [2024-08-09] [设置第二阶段特定的优化参数]
    # WDD [2024-08-09] [为运动参数设置学习率]
    opt.velocity_lr = 0.0001 
    opt.angular_velocity_lr = 0.0001
    
    # WDD [2024-08-09] [竞争机制参数]
    COMPETITION_START_ITER = 500  # WDD [2024-08-09] [多少次迭代后开始竞争性剪枝]
    COMPETITION_INTERVAL = 100    # WDD [2024-08-09] [每隔多少次迭代进行一次剪枝]
    MIN_CONTRIBUTION_SCORE = 0.05 # WDD [2024-08-09] [贡献度得分阈值，低于此值将被剪枝]

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    
    # WDD [2024-08-09] [必须从第一阶段的检查点开始]
    if not start_checkpoint:
        sys.exit("错误：运动模型训练必须通过 --start_checkpoint 从第一阶段的检查点开始。")
    
    print(f"从检查点加载模型: {start_checkpoint}")
    # WDD [2024-08-09] [原因: 修复PyTorch 2.x中因 `weights_only` 默认为True导致的反序列化错误。由于检查点是自生成的，可以安全地设置为False以加载完整的模型和优化器状态。]
    (model_params, first_iter) = torch.load(start_checkpoint, map_location="cuda", weights_only=False)
    gaussians.restore(model_params, opt)
    
    # WDD [2024-08-09] [激活运动模型，并将运动参数加入优化器]
    gaussians.enable_motion_model(opt)
    print(f"运动模型已激活。从迭代次数 {first_iter} 开始训练。")

    scene = Scene(dataset, gaussians, load_iteration=first_iter)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    ema_loss_for_log = 0.0
    
    # WDD [2024-08-09] [初始化贡献度得分记录器]
    contribution_scores = torch.zeros(gaussians.get_xyz.shape[0], device="cuda")

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="训练运动模型")
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # WDD [2024-08-09] [第二阶段不增加SH阶数，保持模型复杂度]
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # WDD [2024-08-09] [更新贡献度得分]
            # WDD [2024-08-09] [一个简单的贡献度计算：可见点的不透明度 * 位置梯度范数]
            # WDD [2024-08-09] [这能奖励那些在图像上有显著影响（高不透明度）且对损失有贡献（高梯度）的点]
            if viewspace_point_tensor.grad is not None:
                # WDD [2024-08-09] [计算位置梯度]
                pos_grad_norm = torch.norm(viewspace_point_tensor.grad[visibility_filter, :2], dim=-1)
                # WDD [2024-08-09] [获取当前帧的透明度]
                opacity_at_time = gaussians.get_opacity_at_time(viewpoint_cam.time_idx)[visibility_filter].squeeze()
                # WDD [2024-08-09] [计算当前帧的得分]
                current_score = opacity_at_time * pos_grad_norm
                # WDD [2024-08-09] [用EMA（指数移动平均）方式平滑地更新总分]
                contribution_scores[visibility_filter] = 0.9 * contribution_scores[visibility_filter] + 0.1 * current_score

            # WDD [2024-08-09] [第二阶段不进行稠密化，只进行竞争性剪枝]
            if iteration > COMPETITION_START_ITER and iteration % COMPETITION_INTERVAL == 0:
                # WDD [2024-08-09] [计算每个点的生命周期长度]
                lifetime_length = torch.abs(gaussians._lifetime_w) * 2
                
                # WDD [2024-08-09] [最终得分 = 平滑后的贡献度 * 生命周期长度]
                # WDD [2024-08-09] [这会奖励那些在长时间内都有高贡献的点]
                final_score = contribution_scores * lifetime_length.squeeze()
                
                # WDD [2024-08-09] [找出得分低于阈值的点]
                prune_mask = final_score < MIN_CONTRIBUTION_SCORE
                
                # WDD [2024-08-09] [保护机制：如果一个点的基础透明度很高，我们倾向于保留它，因为它可能是某个时刻的关键“孤点”]
                high_base_opacity_mask = gaussians.get_base_opacity().squeeze() > 0.7
                prune_mask = torch.logical_and(prune_mask, ~high_base_opacity_mask)

                if prune_mask.any():
                    print(f"\n[ITER {iteration}] 竞争性剪枝: 移除了 {prune_mask.sum().item()} 个低贡献度的点。")
                    gaussians.prune_points(prune_mask)
                    # WDD [2024-08-09] [剪枝后需要重置贡献度得分张量]
                    contribution_scores = contribution_scores[~prune_mask]
                    torch.cuda.empty_cache()

            # WDD [2024-08-09] [更新进度条]
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Points": f"{gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)

            # WDD [2024-08-09] [优化器步骤]
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # WDD [2024-08-09] [保存检查点和最终模型]
            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] 保存运动模型。")
                scene.save(iteration)
            if (iteration in checkpoint_iterations):
                print(f"\n[ITER {iteration}] 保存运动模型检查点。")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_motion_" + str(iteration) + ".pth")

    print("\n运动模型训练完成。")


if __name__ == "__main__":
    parser = ArgumentParser(description="Motion training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000, 20_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training_motion(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)