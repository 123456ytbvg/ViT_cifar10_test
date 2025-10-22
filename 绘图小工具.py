# 绘图小工具.py
import matplotlib

matplotlib.use('WebAgg')  # 使用WebAgg后端，在浏览器中实时显示
import matplotlib.pyplot as plt
import numpy as np
import os
import threading
import time


class TrainingPlotter:
    def __init__(self, save_dir='training_plots'):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.epochs = []
        self.save_dir = save_dir
        self.fig = None
        self.ax1 = None
        self.ax2 = None

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 初始化图形
        self._init_plot()
        print("🌐 实时图表将在浏览器中打开...")
        print("📊 访问: http://localhost:8987 (或类似地址)")

    def _init_plot(self):
        """初始化绘图窗口"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self.fig.suptitle('ViT Training Progress - Real Time', fontsize=16, fontweight='bold')
        plt.ion()  # 开启交互模式
        plt.show(block=False)

    def update(self, epoch, train_loss, val_loss, train_acc, val_acc):
        # 保存数据
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)

        # 清空图形
        self.ax1.clear()
        self.ax2.clear()

        # 绘制损失曲线
        self.ax1.plot(self.epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
        self.ax1.plot(self.epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
        self.ax1.set_title('Training and Validation Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)

        # 绘制准确率曲线
        self.ax2.plot(self.epochs, self.train_accs, 'b-', label='Train Acc', linewidth=2, marker='o', markersize=4)
        self.ax2.plot(self.epochs, self.val_accs, 'r-', label='Val Acc', linewidth=2, marker='s', markersize=4)
        self.ax2.set_title('Training and Validation Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy (%)')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(0, 100)

        # 添加当前数值标注
        if len(self.epochs) > 0:
            self.ax1.annotate(f'Train: {train_loss:.3f}\nVal: {val_loss:.3f}',
                              xy=(epoch, val_loss), xytext=(10, 10), textcoords='offset points',
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            self.ax2.annotate(f'Train: {train_acc:.1f}%\nVal: {val_acc:.1f}%',
                              xy=(epoch, val_acc), xytext=(10, 10), textcoords='offset points',
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        plt.tight_layout()

        # 更新图形（非阻塞）
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.1)  # 短暂处理事件

        # 同时保存图片备份
        plt.savefig(f'{self.save_dir}/training_latest.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{self.save_dir}/training_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')

        print(f"📈 实时图表已更新 (Epoch {epoch}) - 查看浏览器窗口")

    def save_final_plot(self, filename='final_training_plot.png'):
        """保存最终的高清训练曲线"""
        # 创建新的图形用于保存
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 绘制损失曲线
        ax1.plot(self.epochs, self.train_losses, 'b-', label='Train Loss', linewidth=3, alpha=0.8)
        ax1.plot(self.epochs, self.val_losses, 'r-', label='Val Loss', linewidth=3, alpha=0.8)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # 绘制准确率曲线
        ax2.plot(self.epochs, self.train_accs, 'b-', label='Train Accuracy', linewidth=3, alpha=0.8)
        ax2.plot(self.epochs, self.val_accs, 'r-', label='Validation Accuracy', linewidth=3, alpha=0.8)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)

        # 标记最佳准确率
        if len(self.val_accs) > 0:
            best_idx = np.argmax(self.val_accs)
            best_epoch = self.epochs[best_idx]
            best_acc = self.val_accs[best_idx]
            ax2.annotate(f'Best: {best_acc:.2f}%\n(Epoch {best_epoch})',
                         xy=(best_epoch, best_acc), xytext=(10, 30),
                         textcoords='offset points', fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"💾 最终图表已保存: {self.save_dir}/{filename}")

    def close(self):
        """关闭图形"""
        if self.fig:
            plt.close(self.fig)
        plt.ioff()