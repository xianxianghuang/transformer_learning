"""
工具函数模块

本模块提供一些通用的辅助函数，用于：
1. 计算损失曲线统计信息
2. 数据批次迁移到设备
3. 平均值计量器

这些是深度学习训练中常用的工具函数。
"""

import numpy as np


# ==================== 损失曲线统计 ====================

def compute_loss_curve(losses):
    """
    计算损失曲线统计信息

    用于分析训练过程中损失值的变化情况

    参数：
    - losses: 损失值列表
      * 如：[3.5, 3.2, 2.9, 2.6, 2.3]（每个epoch的损失）

    返回值：
    - 字典，包含以下统计信息：
      * min: 最小损失
      * max: 最大损失
      * mean: 平均损失
      * std: 标准差
      * final: 最终损失
      * improvement: 改进百分比

    知识点：
    - 损失曲线可以反映训练是否收敛
    - 损失下降说明模型在学习
    - 损失不下降可能需要调整超参数
    - 改进百分比 = (初始损失 - 最终损失) / 初始损失 * 100%
    """
    return {
        # 最小损失
        'min': np.min(losses),

        # 最大损失
        'max': np.max(losses),

        # 平均损失
        'mean': np.mean(losses),

        # 标准差（反映损失的波动程度）
        'std': np.std(losses),

        # 最终损失（最后一个epoch的损失）
        'final': losses[-1] if losses else None,

        # 改进百分比
        # 计算公式：(初始损失 - 最终损失) / 初始损失 * 100
        # 防止除以0
        'improvement': (losses[0] - losses[-1]) / losses[0] * 100 if losses and losses[0] > 0 else 0
    }


# ==================== 批次迁移到设备 ====================

def batch_to_device(batch, device):
    """
    将批次数据移动到指定设备（CPU或GPU）

    作用：
    在训练循环中，需要把数据从CPU移到GPU（如果使用GPU）
    这个函数统一处理不同类型的数据

    参数：
    - batch: 数据批次
      * 可以是以下类型：
        - torch.Tensor：张量
        - list：列表
        - tuple：元组
        - dict：字典
      * 例如：(context, target, negative) 元组
    - device: 目标设备
      * torch.device('cuda') 或 torch.device('cpu')

    返回值：
    - 移动到目标设备后的数据

    知识点：
    - GPU比CPU快很多，特别是矩阵运算
    - 但数据需要先移到GPU才能使用
    - .to(device) 不会改变原始张量，返回新的张量

    示例：
    # CPU数据
    batch = (context, target, negative)
    # 移到GPU
    batch = batch_to_device(batch, torch.device('cuda'))
    """
    # 判断数据类型
    if isinstance(batch, (list, tuple)):
        # 如果是列表或元组
        # 递归处理每个元素
        # [b.to(device) for b in batch] 把每个元素移到device
        return [b.to(device) for b in batch]

    elif isinstance(batch, dict):
        # 如果是字典
        # 处理每个值
        return {k: v.to(device) for k, v in batch.items()}

    else:
        # 其他类型（通常是单个张量）
        # 直接移动到设备
        return batch.to(device)


# ==================== 平均值计量器 ====================

class AverageMeter:
    """
    计算并存储平均值和当前值

    作用：
    - 在训练过程中跟踪某个指标（如损失、准确率）
    - 自动计算平均值
    - 方便打印训练进度

    使用场景：
    - 跟踪每个batch的损失
    - 跟踪训练/验证准确率

    知识点：
    - 这是深度学习中的常见模式
    - PyTorch的各种示例代码经常用到
    - 类似于指标累计器
    """

    def __init__(self):
        """
        初始化计量器
        """
        # 重置所有值
        self.reset()


    def reset(self):
        """
        重置所有统计值

        通常在以下情况调用：
        - 开始新的epoch时
        - 开始新的训练阶段时
        """
        self.val = 0       # 当前值（最后一个batch的值）
        self.avg = 0       # 平均值
        self.sum = 0       # 累计和
        self.count = 0     # 样本数量


    def update(self, val, n=1):
        """
        更新统计值

        参数：
        - val: 新值（当前batch的值）
        - n: 样本数量（通常是batch_size）
          * 默认为1
          * 如果batch大小不同，需要指定

        示例：
        # 创建计量器
        meter = AverageMeter()

        # 训练循环中
        for batch in dataloader:
            loss = compute_loss()

            # 更新计量器
            # 这里的 n=batch_size
            meter.update(loss, batch.size(0))

        # 打印平均值
        print(f"Average loss: {meter.avg}")
        """
        # 更新当前值
        self.val = val

        # 累计和：sum += val * n
        # 乘以n是因为val是平均值，需要乘以样本数得到总量
        self.sum += val * n

        # 累计样本数
        self.count += n

        # 计算平均值
        # 平均值 = 累计和 / 累计样本数
        self.avg = self.sum / self.count


# ==================== 使用示例 ====================

if __name__ == '__main__':
    """
    演示各个函数/类的用法
    """

    # 示例1：compute_loss_curve
    print("=" * 50)
    print("示例1：损失曲线统计")
    print("=" * 50)

    # 模拟训练损失
    losses = [3.5, 3.2, 2.9, 2.6, 2.4, 2.2, 2.1, 2.0, 1.9, 1.85]

    # 计算统计信息
    stats = compute_loss_curve(losses)

    print(f"最小损失: {stats['min']:.4f}")
    print(f"最大损失: {stats['max']:.4f}")
    print(f"平均损失: {stats['mean']:.4f}")
    print(f"标准差: {stats['std']:.4f}")
    print(f"最终损失: {stats['final']:.4f}")
    print(f"改进百分比: {stats['improvement']:.1f}%")


    # 示例2：AverageMeter
    print("\n" + "=" * 50)
    print("示例2：平均计量器")
    print("=" * 50)

    # 创建计量器
    meter = AverageMeter()

    # 模拟训练过程（10个batch）
    batch_losses = [3.5, 3.2, 2.9, 2.6, 2.4, 2.2, 2.1, 2.0, 1.9, 1.85]
    batch_sizes = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512]

    for i, (loss, n) in enumerate(zip(batch_losses, batch_sizes)):
        meter.update(loss, n)
        print(f"Batch {i+1}: 当前={meter.val:.4f}, 平均={meter.avg:.4f}, 累计={meter.count}")

    print(f"\n最终平均损失: {meter.avg:.4f}")


    # 示例3：batch_to_device
    print("\n" + "=" * 50)
    print("示例3：批次迁移到设备")
    print("=" * 50)

    import torch

    # 假设这是训练数据
    context = torch.randn(32, 10)   # 上下文 (batch_size=32, 10个上下文词)
    target = torch.randint(0, 1000, (32,))  # 目标词 (32,)
    negative = torch.randint(0, 1000, (32, 5))  # 负样本 (32, 5个)

    # 原始数据在CPU上
    batch = (context, target, negative)
    print(f"原始数据类型: {type(batch)}")
    print(f"原始数据设备: {batch[0].device}")

    # 移动到CPU（演示用，实际上通常移到GPU）
    device = torch.device('cpu')
    batch = batch_to_device(batch, device)
    print(f"移动后设备: {batch[0].device}")

    """
    总结：

    1. compute_loss_curve:
       - 用于分析训练曲线
       - 快速了解训练状态

    2. batch_to_device:
       - 统一的数据迁移接口
       - 支持多种数据类型

    3. AverageMeter:
       - 跟踪训练指标
       - 自动计算平均值
       - 代码更简洁

    这些工具函数虽然简单，但能显著提升代码的可读性和可维护性。
    """
