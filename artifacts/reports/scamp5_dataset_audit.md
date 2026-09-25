# SCAMP-5 MNIST 数据一致性审计

当前权重文件是 `original_repo/WEIGHTS_MNIST_4x4_16CONVOLS_64x64INPUT_95ACC.hpp`，固件入口是 `MAIN_MNIST_SINGLE_LAYER_16.cpp`。因此可确认：

- 网络输入工作平面为 64×64，但不是把整张 28×28 MNIST 画布直接拉伸到 64×64。
- 固件先以阈值 100 做前景提取和 flood-fill，再按 bounding box 居中；`extracted_digit_img_size` 默认是 45。
- SCAMP 的数字缩放由 `IMG_SCALING_DIGITAL.cpp` 中的 DNEWS 操作完成，和普通 PIL/torchvision 插值不等价。
- 卷积后是 `maxpool=4`，所以每个 64×64 特征图变成 16×16；FC 权重布局为 10×16×16×16。
- 固件注释中的 `%8800` 表明演示使用过一个约 8800 张图片的本地评估目录；仓库没有提供这份目录或训练划分。因此不能把 torchvision 官方 MNIST test split 直接称为原论文/固件的同一数据集。

已新增 `python -m baseline.evaluate_scamp_mnist`：它使用公开 torchvision MNIST test split、固件原始 ternary 权重和匹配的“阈值—裁剪—居中—45px—64×64”CPU 近似预处理。输出是可复现的软件对照结果，不宣称复现 SCAMP 硬件的 95%。

示例（已在 100 个测试样本上运行）：准确率 0.50。这个数值反映公开 MNIST 与固件数据/模拟器语义仍存在差异，不能与 README 中的 95% 直接比较；若要严格复现，需取得作者的 8800 张输入图及其生成/划分脚本，或在公开 MNIST 上重新训练一套与该预处理对应的权重。
