# FreeTimeGS（非官方实现）

## 分支说明
当前分支，共享主要的高斯参数，而为每个时刻（time_idx）创建一个独立的opacity数值，用来学习。


## 方法概览
本仓库是 CVPR 2024《FreeTimeGS: Free Gaussian Primitives at Anytime Anywhere for Dynamic Scene Reconstruction》的非官方复现版本，目标是高质量、实时的动态场景渲染。



- **任意时空的 4D 高斯表示**：相比只在 canonical 空间放置高斯的方案，这里允许高斯在任意时间步和空间位置出现，显著提升对快速、大幅运动的建模能力。
- **每个高斯的显式运动函数**：为每个高斯分配一个线性速度 `v`，位置随时间 `t` 更新为 `μx(t)=μx+v·(t-μt)`，用少量参数即可覆盖邻域运动，降低长程对应的优化难度。
- **时间不透明度调制**：通过高斯形状的 `σ(t)` 控制单个高斯在时间轴上的影响范围，配合早期的高不透明度正则（4D regularization）避免梯度被少数高斯“堵塞”。
- **周期性重定位与 4D 初始化**：迭代中根据梯度与不透明度得分，把低贡献的高斯迁移到需求更高的区域；初始化阶段利用 ROMA 匹配与三角测量估计初始位置、时间与速度，加速收敛。

整体流程：利用多视角视频输入 → 初始化 4D 高斯 → 通过 CUDA 加速的高斯光栅化渲染 → 最小化图像/SSIM/感知损失及 4D 正则 → 导出可实时渲染的动态场景。

## 环境准备

```bash
conda env create -f environment.yml
conda activate freetime_gs
```

## 数据准备

将多视角视频或处理后的帧放入 `data/`，目录结构需符合 `train.py` 的数据加载约定（示例可参考 `data/sample_scene`，或将自有数据软链到该目录下）。

## 训练

```bash
python train.py -s <path_to_scene>
```

常见可选参数：

- `--iterations`：训练迭代次数，默认 30k，可依据序列长度与动作复杂度调整。
- `--downsample`：图像缩放比例，长序列可先以 0.5 训练再微调。

## 渲染 / 导出

```bash
python render.py -m <path_to_output>
```

该脚本会加载训练好的高斯簇，按照给定的时间帧与相机轨迹生成视频或图像；也可修改脚本以导出 `.ply` 方便调试。

## 参考

- 论文：FreeTimeGS（CVPR 2024）
- 若需更多实现细节，请阅读仓库根目录中的 `FreeTimeGS.pdf`。
