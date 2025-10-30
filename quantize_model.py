import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from ViT import Vit


def get_test_loader(test_data_path, batch_size=64, num_workers=2):
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_dataset = datasets.ImageFolder(root=test_data_path, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return test_loader


def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    total_time = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            start = time.time()
            outputs = model(inputs)
            total_time += (time.time() - start)
            _, predicted = outputs.max(1)
            correct += predicted.cpu().eq(targets).sum().item()
            total += targets.size(0)

    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc, total_time


def file_size_mb(path):
    return os.path.getsize(path) / (1024.0 * 1024.0)


def main():
    # 配置
    test_data_path = 'CIFAR10_balanced/CIFAR10_balance'
    batch_size = 64
    device_cpu = torch.device('cpu')

    # 构建与训练时相同的模型结构（与 ViT_train.py 中保持一致）
    model = Vit(
        patch_size=8,
        embed_dim=384,
        num_heads=6,
        max_seq_length=100,
        encoder_num=6,
        dropout=0.1,
        num_classes=10
    )

    # 加载检查点
    ckpt_path = 'best_vit_model.pth'
    if not os.path.exists(ckpt_path):
        print(f"找不到权重文件: {ckpt_path}。请确认在仓库根目录下有训练好的 'best_vit_model.pth'。")
        return

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    # 兼容 checkpoint 可能直接保存了 model.state_dict() 或者包含字典
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    # 准备测试集
    test_loader = get_test_loader(test_data_path, batch_size)

    print("\n== 在 CPU 上评估原始（未量化）模型 ==")
    orig_acc, orig_time = evaluate(model, test_loader, device_cpu)
    print(f"原始模型准确率: {orig_acc:.2f}%，总推理时间: {orig_time:.3f}s")

    # 保存原始模型大小信息（如果需要）
    tmp_orig_path = 'tmp_orig_model.pth'
    torch.save(model.state_dict(), tmp_orig_path)
    orig_size = file_size_mb(tmp_orig_path)
    os.remove(tmp_orig_path)
    print(f"原始模型体积(近似): {orig_size:.2f} MB")

    # 应用动态量化（只量化 Linear 层）
    print("\n== 对模型应用动态量化 (Linear -> qint8) ==")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    # 保存量化后模型
    quantized_path = 'best_vit_model_quantized.pth'
    torch.save(quantized_model.state_dict(), quantized_path)
    quant_size = file_size_mb(quantized_path)
    print(f"量化模型已保存: {quantized_path} (约 {quant_size:.2f} MB)")

    print("\n== 在 CPU 上评估量化后模型 ==")
    q_acc, q_time = evaluate(quantized_model, test_loader, device_cpu)
    print(f"量化后模型准确率: {q_acc:.2f}%，总推理时间: {q_time:.3f}s")

    # 对比信息
    print("\n== 比较总结 ==")
    print(f"准确率变化: {q_acc - orig_acc:+.2f}%")
    print(f"推理时间变化 (CPU，总计): {q_time - orig_time:+.3f}s")
    print(f"体积变化: {quant_size - orig_size:+.2f} MB ({quant_size/orig_size:.2f}x) ")

    print("\n完成：量化模型文件保存在仓库根目录。可将其用于 CPU 上部署以节省内存/存储并通常加速 CPU 推理。")


if __name__ == '__main__':
    main()
