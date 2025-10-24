# ViT CIFAR-10 训练示例

这是一个基于 Vision Transformer (ViT) 在 CIFAR-10 数据集上训练与评估的简易示例工程，包含训练脚本、模型定义与实时训练曲线可视化工具。

## 项目结构

- `ViT.py` — ViT 模型实现（包含位置嵌入、Encoder、分类头等）。
- `config.json` — 模型和参数选择。
- `ViT_train.py` — 训练主脚本：数据加载、训练/验证循环、学习率调度、模型保存、最终测试评估。
- `绘图小工具.py` — 实时绘图工具 `TrainingPlotter`，使用 matplotlib 的 WebAgg 后端在浏览器中显示训练曲线，并保存图片。
- `CIFAR10_balanced/` — 用于测试/验证的平衡 CIFAR10 子集（按类分文件夹）。
- `CIFAR10_imbalanced/` — 用于训练的非平衡 CIFAR10 子集（按类分文件夹）。

## 环境与依赖

推荐使用 Python 3.8+，并安装以下主要依赖：

- torch（与您的 CUDA 版本匹配）
- torchvision
- matplotlib
- tqdm

可以使用 pip 安装（在 Windows PowerShell 中运行）：

```powershell
python -m pip install torch torchvision matplotlib tqdm
```

注意：如果您有 NVIDIA GPU，请根据您的 CUDA 版本安装对应的 PyTorch 二进制（参见 https://pytorch.org/get-started/locally/）。

## 数据准备

本项目使用的是按文件夹组织的图像分类数据结构（与 torchvision.datasets.ImageFolder 兼容）。

仓库中已包含两个示例目录：

- `CIFAR10_imbalanced/CIFAR10_unbalance/` — 训练集，每个子文件夹 `0`..`9` 包含对应类别图像。
- `CIFAR10_balanced/CIFAR10_balance/` — 测试/验证集，每个子文件夹 `0`..`9` 包含对应类别图像。

如果你想使用原始 CIFAR-10 数据集，可以将图像以同样的目录结构解压到上述目录，或修改 `ViT_train.py` 中的路径变量 `train_data_path` 与 `test_data_path`。

注意：训练脚本默认会把训练集划分为 90% 训练、10% 验证（基于 ImageFolder 的随机拆分），并使用完整的 `test_data_path` 作为最终测试集。

## 快速运行

在 PowerShell 中，进入项目根目录后直接运行：

```powershell
# 进入项目目录
cd d:\00_Course\03_DeepLearn\ViT_cifar10_test

# 运行训练脚本
python ViT_train.py
```

脚本默认配置（可在 `ViT_train.py` 中修改）：

- batch_size = 32
- epochs = 500
- learning_rate = 5e-5
- weight_decay = 1e-4
- model: Vit(patch_size=8, embed_dim=384, num_heads=6, encoder_num=6, num_classes=10)

训练过程中会显示进度条与终端日志，并尝试启动 `TrainingPlotter`（使用 matplotlib 的 WebAgg 或 TkAgg 后端）。如果成功，浏览器会在本地打开实时曲线（常见地址为 http://localhost:8987）。

## 可视化

可视化由 `绘图小工具.py` 中的 `TrainingPlotter` 提供。默认使用 WebAgg 后端：

- 确保 matplotlib 支持 WebAgg（通常默认安装即可）。
- 训练脚本会在启动时尝试创建对象并输出本地访问地址（示例：`http://localhost:8987`）。
- 如果在无 GUI 的服务器上运行或后端初始化失败，脚本会回退到仅在控制台输出日志并保存训练图片到 `training_plots/`。

训练过程中 `TrainingPlotter` 会保存 `training_latest.png` 和每个 epoch 的快照 `training_epoch_{epoch}.png` 到 `training_plots/`。

## 模型保存与加载

训练脚本会在验证精度提升时保存最佳模型为 `best_vit_model.pth`，并每 10 个 epoch 保存一次检查点 `checkpoint_epoch_{epoch}.pth`。

这些 checkpoint 包含以下字段（可用于继续训练或推理）：

- `epoch` — 保存时的 epoch 编号
- `model_state_dict` — 模型参数
- `optimizer_state_dict` — 优化器状态
- `scheduler_state_dict` — 学习率调度器状态（若有）
- `accuracy` — 当时验证集精度

示例：在脚本末尾加载最佳模型并在测试集上评估：

```python
checkpoint = torch.load('best_vit_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## 常见问题与调优建议

- 如果显存不足：减小 `batch_size` 或 `embed_dim`、`encoder_num`。
- 加速训练：在具备多 GPU 的环境中可以用 DistributedDataParallel（当前脚本为单卡实现）。
- 学习率调整：默认使用 CosineAnnealingLR；不同设置可改成 StepLR 或 ReduceLROnPlateau。
- 数据增强：当前训练使用 RandomCrop 与 RandomHorizontalFlip；可按需增加 ColorJitter、Mixup 等。

## 开发者说明 / 代码阅读指引

- `ViT.py` 中实现了基础的 ViT 结构：patch embedding、cls token、位置编码、多个 Encoder（包含自注意力和 MLP）。
- `ViT_train.py` 包含训练/验证循环、类不平衡权重计算、日志输出与 checkpoint 逻辑。
- `绘图小工具.py` 用于实时可视化并保存训练曲线。

## 后续改进建议

- 添加命令行参数解析（argparse）以便外部传参（路径、超参数、是否使用 GPU 等）。
- 增加单元测试与小样本运行脚本，方便快速调试。
- 支持从 torchvision 直接下载并转换 CIFAR-10 数据集以减少手动预处理步骤。

---

如果你希望我把 README 中的某个部分改成更详细的安装步骤（例如不同 CUDA 版本的安装命令），或为训练脚本添加 argparse 支持并提交修改，我可以继续完成这些工作.
