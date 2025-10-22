from ViT import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from tqdm import tqdm
from 绘图小工具 import *
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
import matplotlib.pyplot as plt

# 数据预处理
def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    return train_transform, val_transform


# 数据加载
def get_dataloaders(train_data_path, test_data_path, batch_size=64, num_workers=2):
    train_transform, val_transform = get_transforms()

    try:
        # 训练集
        train_dataset = datasets.ImageFolder(
            root=train_data_path,
            transform=train_transform
        )

        # 测试集
        test_dataset = datasets.ImageFolder(
            root=test_data_path,
            transform=val_transform
        )

        print(f"✅ 成功加载训练集，总样本数: {len(train_dataset)}")
        print(f"✅ 成功加载测试集，总样本数: {len(test_dataset)}")
        print(f"✅ 类别: {train_dataset.classes}")

        # 分析训练集类别分布
        train_class_counts = {}
        for _, label in train_dataset.samples:
            train_class_counts[label] = train_class_counts.get(label, 0) + 1

        print("训练集类别分布:")
        for class_idx in sorted(train_class_counts.keys()):
            print(f"  类别 {class_idx}: {train_class_counts[class_idx]} 个样本")

        # 训练集分割 (90% 训练, 10% 验证)
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        # 验证集使用训练集的分割部分
        val_dataset = datasets.ImageFolder(
            root=train_data_path,
            transform=val_transform
        )
        val_subset = torch.utils.data.Subset(val_dataset, val_subset.indices)

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )

        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        # 测试集使用完整的平衡数据集
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        return train_loader, val_loader, test_loader, train_dataset.classes

    except Exception as e:
        print(f"❌ 数据加载错误: {e}")
        return None, None, None, None


# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} Training')

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{running_loss / (batch_idx + 1):.3f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


# 验证函数
def validate_epoch(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} Validation')

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'Loss': f'{running_loss / (batch_idx + 1):.3f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


# 学习率调度器
def get_scheduler(optimizer, epochs):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


# 保存检查点
def save_checkpoint(model, optimizer, scheduler, epoch, acc, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'accuracy': acc,
    }, path)

mode = "new"
# mode = "load"
# 主训练函数
def main():
    # 配置参数
    train_data_path = 'CIFAR10_imbalanced/CIFAR10_unbalance'
    test_data_path = 'CIFAR10_balanced/CIFAR10_balance'
    batch_size = 32
    epochs = 500
    learning_rate = 5e-5
    weight_decay = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"🎯 使用设备: {device}")

    # 创建模型
    model = Vit(
        patch_size=8,
        embed_dim=384,
        num_heads=6,
        max_seq_length=100,
        encoder_num=6,
        dropout=0.1,
        num_classes=10
    ).to(device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数: {total_params:,} (可训练: {trainable_params:,})")

    # 数据加载
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        train_data_path, test_data_path, batch_size
    )
    if train_loader is None:
        print("❌ 数据加载失败，退出训练")
        return

    print(f"🏷️  类别: {classes}")
    print(f"📚 训练样本: {len(train_loader.dataset):,}")
    print(f"🔍 验证样本: {len(val_loader.dataset):,}")
    print(f"🧪 测试样本: {len(test_loader.dataset):,}")

    # 计算类别权重
    def calculate_class_weights(dataloader, num_classes=10):
        class_counts = torch.zeros(num_classes)
        for _, targets in dataloader:
            for target in targets:
                class_counts[target] += 1
        class_counts = torch.clamp(class_counts, min=1)
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * num_classes
        return weights

    class_weights = calculate_class_weights(train_loader).to(device)
    print("⚖️  类别权重:", class_weights.cpu().numpy())

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    scheduler = get_scheduler(optimizer, epochs)

    # 初始化Web实时绘图器
    try:
        plotter = TrainingPlotter()
        use_web_plotter = True
        print("🌐 Web实时绘图已启用!")
        print("💡 请打开浏览器查看实时训练进度")
        print("📱 通常访问: http://localhost:8987")
        print("⏳ 等待浏览器连接...")
        time.sleep(2)  # 给浏览器一些连接时间
    except Exception as e:
        print(f"❌ Web绘图初始化失败: {e}")
        print("📊 将使用控制台输出模式")
        use_web_plotter = False

    # 训练记录
    best_acc = 0.0

    print("\n" + "=" * 80)
    print("🚀 开始训练 ViT 模型!")
    print("=" * 80)

    try:
        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # 训练
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )

            # 验证
            val_loss, val_acc = validate_epoch(
                model, val_loader, criterion, device, epoch
            )

            # 学习率调度
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            epoch_time = time.time() - start_time

            # 更新Web实时图表
            if use_web_plotter:
                plotter.update(epoch, train_loss, val_loss, train_acc, val_acc)

            # 计算准确率变化和喵喵消息
            delt = val_acc - best_acc

            # 喵喵激励系统
            if delt < 0:
                if delt < -0.003:
                    color_msg = "\033[91m气死我了喵 😠\033[0m"
                    emoji = "💢"
                else:
                    color_msg = "\033[96m怎么回事喵？ 🤔\033[0m"
                    emoji = "❓"
            else:
                if delt > 0.003:
                    color_msg = "\033[92m干的漂亮喵！ 🎉\033[0m"
                    emoji = "✨"
                else:
                    color_msg = "\033[93m就是这样喵~ 😊\033[0m"
                    emoji = "👍"

            # 彩色输出
            print(f'\n┌{"─" * 70}┐')
            print(f'│ 📅 Epoch {epoch:2d}/{epochs} │ ⏱️  {epoch_time:5.1f}s │ 📊 LR: {current_lr:.2e} │')
            print(f'├{"─" * 70}┤')
            print(f'│ 🚂 Train │ Loss: {train_loss:7.4f} │ Acc: {train_acc:6.2f}% │')
            print(f'│ 🧪 Val   │ Loss: {val_loss:7.4f} │ Acc: {val_acc:6.2f}% │ {emoji}')
            print(f'│ {color_msg:^50} │')

            if delt > 0:
                print(f'│ 🎯 准确率提升: +{delt:.3f}%{" ":>30} │')
            elif delt < 0:
                print(f'│ 📉 准确率下降: {delt:.3f}%{" ":>30} │')
            else:
                print(f'│ ➡️  准确率持平{" ":>40} │')
            print(f'└{"─" * 70}┘')

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_acc,
                    'best_vit_model.pth'
                )
                print(f'\n✨ \033[92m🎊 新的最佳模型已保存! 准确率: {best_acc:.2f}%\033[0m\n')

            # 每10个epoch保存一次检查点
            if epoch % 10 == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_acc,
                    f'checkpoint_epoch_{epoch}.pth'
                )
                print(f"💾 检查点已保存: checkpoint_epoch_{epoch}.pth")

    except KeyboardInterrupt:
        print("\n🛑 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
    finally:
        # 保存最终绘图
        if use_web_plotter:
            plotter.save_final_plot()
            plotter.close()
        print(f"\n📁 所有训练图表已保存到: {os.path.abspath('training_plots')}")

    print(f"\n🎊 训练完成! 最佳验证准确率: {best_acc:.2f}%")

    # 最终在测试集上评估最佳模型
    print("\n🔍 加载最佳模型进行最终测试...")
    checkpoint = torch.load('best_vit_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    final_test_loss, final_test_acc = validate_epoch(
        model, test_loader, criterion, device, "Final Test"
    )

    # 测试结果评价
    if final_test_acc > 80:
        color = 92
        rating = "🎉 优秀!"
    elif final_test_acc > 70:
        color = 93
        rating = "👍 良好"
    elif final_test_acc > 60:
        color = 96
        rating = "✅ 一般"
    else:
        color = 91
        rating = "💪 需改进"

    print(f"\n🎯 最终测试结果:")
    print(f"   📊 准确率: \033[{color}m{final_test_acc:.2f}%\033[0m - {rating}")
    print(f"   📉 损失: {final_test_loss:.4f}")


if __name__ == "__main__":
    main()