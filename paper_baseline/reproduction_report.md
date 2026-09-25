# Liu & Lu 二值 FCN 真实数据复现报告

## 1. 复现对象

目标论文为 Liu 与 Lu 的 On-Sensor Binarized Fully Convolutional Neural Network for Localisation and Coarse Segmentation（CVPRW 2022）。当前实现复现论文三层二值 FCN 拓扑：64×64 输入、16 个 4×4 二值卷积核、8 组 3×3 grouped binary convolution（128 通道融合为 64 通道）、64 个 1×1 二值卷积核，以及热图输出头。

这是面向 PC/GPU 的软件复现，不是 SCAMP-5 硬件逐周期复现。BatchNorm 和训练阈值保留在 PyTorch 图中；论文部署阶段将相关参数折叠为偏置/阈值。

## 2. 真实数据集

- 数据集：Oxford-IIIT Pet
- 下载目录：../data/oxford-iiit-pet
- 训练划分：官方 trainval，3680 张
- 测试划分：官方 test，3669 张
- 分割标签：Oxford 像素级 trimap 的宠物前景区域
- 定位标签：由真实前景掩码质心生成 64×64 高斯热图
- 适配器：realdata.py

论文原始 Webots 鸟瞰道路/草地数据没有随 PDF 公开，因此本报告不把 Oxford-IIIT Pet 结果宣称为论文原始数据结果。

## 3. 软件与硬件环境

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- NVIDIA 驱动：580.88（驱动报告 CUDA 13.0）
- PyTorch：2.14.0+cu126
- CUDA runtime：12.6
- torch.cuda.is_available()：True
- 训练设备：CUDA

## 4. 训练命令

分割任务（15 epochs）：

    .venv\\Scripts\\python.exe -m paper_baseline.train --dataset oxford_pet --download --device cuda --epochs 15 --samples 0 --batch-size 64 --task segmentation --output artifacts/paper_oxford_seg_gpu_15ep.pt

定位任务（10 epochs）：

    .venv\\Scripts\\python.exe -m paper_baseline.train --dataset oxford_pet --device cuda --epochs 10 --samples 0 --batch-size 64 --task localisation --output artifacts/paper_oxford_loc_gpu_10ep.pt

训练入口：train.py。模型 checkpoint：

- ../artifacts/paper_oxford_seg_gpu_15ep.pt
- ../artifacts/paper_oxford_loc_gpu_10ep.pt

## 5. 官方 test 完整评估

| 任务 | 训练轮数 | 测试样本 | IoU | Dice/F1 | 其他 |
|---|---:|---:|---:|---:|---:|
| 前景分割 | 15 | 3669 | 0.0000020 | 0.0000041 | 预测趋向全背景 |
| 质心定位热图 | 10 | 3669 | 0.967565 | 0.983513 | 10 像素内准确率 0.105751 |

分割结果显示当前论文轻量二值 FCN 在 Oxford-IIIT Pet 上出现严重类别不平衡/域迁移问题，不能作为成功的自然图像分割结果。该失败结果已如实保留，后续应使用前景加权 BCE 或 Dice/Focal loss、数据增强和更长训练重新优化。

定位热图的像素 IoU/Dice 较高，但质心 top-1 的 10 像素准确率只有 10.58%，说明热图整体回归与峰值位置指标并不一致，后续应增加 soft-argmax 或坐标回归损失。

## 6. 模型效率指标

- 总参数量：7217
- 二值卷积权重参数：7168
- 二值权重理论存储：约 896 bytes（7168 bit）
- 分层二值卷积项/帧：约 20,484,224
  - conv1：952,576
  - grouped conv2：4,286,592
  - conv3：15,245,056
- 理论乘法数：0；主要操作为符号/XNOR 类二值运算与累加

以上为网络拓扑的分析代理指标，不是 SCAMP-5 实测功耗、延迟或 FPS。

## 7. 复现边界与结论

本次已完成：论文相关架构复现、真实公开数据下载、官方 trainval/test 划分、多轮 GPU 训练、完整 test 评估、参数量和二值运算量统计。结果表明定位任务可形成可用基线，分割任务需要进一步处理类别不平衡和域差异后才能作为有竞争力基线。不能将本报告的 Oxford-IIIT Pet 指标与论文 Webots/SCAMP-5 指标直接横向比较。

## 8. 论文风格 Webots 鸟瞰数据复现

由于论文原始 Webots 数据未公开，本仓库新增了 [webots_like.py](webots_like.py)，按论文描述生成 64×64 灰度鸟瞰帧、道路粗分割 mask、车辆位置高斯热图和车辆中心点。训练/测试使用不同随机种子，测试集 1024 张；这属于程序化协议复现，不是作者原始 Webots 场景。

训练命令：

    .venv\\Scripts\\python.exe -m paper_baseline.train --dataset webots_like --device cuda --epochs 10 --samples 4096 --batch-size 128 --task segmentation --output artifacts/webots_like_seg_gpu.pt
    .venv\\Scripts\\python.exe -m paper_baseline.train --dataset webots_like --device cuda --epochs 10 --samples 4096 --batch-size 128 --task localisation --output artifacts/webots_like_loc_gpu.pt

独立测试结果：

| 任务 | 测试样本 | IoU | Dice/F1 | 10 像素定位准确率 |
|---|---:|---:|---:|---:|
| 道路粗分割 | 1024 | 0.997766 | 0.998882 | 0.301758 |
| 车辆定位热图 | 1024 | 0.870655 | 0.930855 | 0.595703 |

该结果与论文报告的仿真道路 IoU 74.0%、草地 IoU 76.6%、定位 10 像素准确率约 88% 不应直接比较，因为程序化场景更简单，且没有复现论文 Webots 的真实纹理、树木遮挡、相机噪声和车辆轨迹分布。它的作用是验证论文网络拓扑、标签协议和训练流程可以端到端运行。

## 9. 增强困难场景与改进损失

为缩小与论文场景的差距，webots_like.py 增加了道路/草地灰度重叠、空间光照梯度、树木/阴影遮挡、随机传感器噪声；训练脚本改用前景加权 BCE+Dice，定位任务增加 soft-argmax 坐标损失。8 epochs、4096 训练样本、1024 独立测试样本的 GPU 结果如下：

| 任务 | IoU | Dice/F1 | 其他 |
|---|---:|---:|---:|
| 道路分割 | 0.560907 | 0.718684 | 草地 IoU 0.254700 |
| 车辆定位热图 | 0.633856 | 0.775899 | 平均中心误差 11.71 px；10 px 命中率 0.391602 |

增强场景下道路 IoU 低于论文仿真道路 74.0%，草地 IoU 也低于论文仿真草地 76.6%，定位 10 像素命中率低于论文约 88%。这说明当前二值 FCN 在加入纹理混淆和遮挡后仍存在容量/训练不足问题；但该结果比原先过于简单场景的 99.78% 道路 IoU 更能反映真实复现难度。后续可继续增加训练轮数、优化阈值参数和使用多头道路/草地输出。

## 10. 30 轮双输出头结果

分割头改为 2 通道，分别预测 road 和 grass，并在增强困难场景上训练 30 epochs。测试集为独立随机种子的 1024 张图像。

checkpoint：../artifacts/webots_dual_seg_30ep.pt

| 指标 | 结果 |
|---|---:|
| Road IoU | 0.582088 |
| Grass IoU | 0.585453 |
| Macro IoU | 0.583771 |
| Road Dice/F1 | 0.735836 |
| Grass Dice/F1 | 0.738504 |
| Macro Dice/F1 | 0.737170 |

双输出头使道路和草地指标趋于平衡，相比 8 轮单头结果的道路 IoU 0.560907、草地 IoU 0.254700 有明显改善，尤其改善了草地分支；但 Macro IoU 仍低于论文仿真结果（道路 74.0%、草地 76.6%），说明增强场景下仍存在域复杂度和二值模型容量差距。

## 11. 50 轮与余弦退火

进一步使用 50 epochs、余弦退火学习率和双输出头训练。测试结果为：Road IoU 0.578279、Grass IoU 0.614544、Road Dice/F1 0.732787、Grass Dice/F1 0.761252，Macro IoU 0.596412。相比 30 轮结果，草地 IoU 提升约 2.9 个百分点，宏平均 IoU 提升约 1.3 个百分点；道路 IoU 基本持平，说明当前瓶颈主要来自二值模型容量和困难场景域差异，而不是单纯训练轮数不足。
## 12. 定位分支 50 轮坐标损失复训

定位分支使用归一化 soft-argmax 坐标损失权重 0.1、余弦退火学习率训练 50 epochs。checkpoint：../artifacts/webots_loc_50ep_coord.pt。

独立测试结果：热图 IoU 0.762875，热图 Dice/F1 0.865488，平均中心误差 12.31 px，10 像素内准确率 0.368164。该结果低于此前 8 轮定位实验的 0.633856 热图 IoU、11.71 px 中心误差和 0.391602 命中率，说明当前坐标损失权重和优化设置不理想，不能仅靠增加 epochs 缩小与论文约 88% 定位准确率的差距。
