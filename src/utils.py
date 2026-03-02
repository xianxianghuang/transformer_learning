"""
工具函数模块
"""

import numpy as np


def compute_loss_curve(losses):
    """
    计算损失曲线统计信息

    Args:
        losses: 损失列表

    Returns:
        统计信息字典
    """
    return {
        'min': np.min(losses),
        'max': np.max(losses),
        'mean': np.mean(losses),
        'std': np.std(losses),
        'final': losses[-1] if losses else None,
        'improvement': (losses[0] - losses[-1]) / losses[0] * 100 if losses and losses[0] > 0 else 0
    }


def batch_to_device(batch, device):
    """
    将批次数据移动到指定设备

    Args:
        batch: 数据批次
        device: 目标设备

    Returns:
        移动后的批次
    """
    if isinstance(batch, (list, tuple)):
        return [b.to(device) for b in batch]
    elif isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items()}
    else:
        return batch.to(device)


class AverageMeter:
    """
    计算并存储平均值和当前值
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
