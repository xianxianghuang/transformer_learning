"""
CBOW模型实现
使用PyTorch实现带负采样的CBOW模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CBOW(nn.Module):
    """
    Continuous Bag of Words (CBOW) 模型

    模型结构:
    1. Embedding层: 将词索引映射为密集向量
    2. 上下文平均: 将窗口内所有上下文词的向量取平均
    3. 输出层: 使用负采样进行训练，预测目标词
    """

    def __init__(self, vocab_size, embedding_dim, window_size, negative_samples=5):
        """
        初始化CBOW模型

        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词向量维度
            window_size: 上下文窗口大小
            negative_samples: 负采样数量
        """
        super(CBOW, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negative_samples = negative_samples

        # 词嵌入层 (target embeddings)
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # 上下文嵌入层 (context embeddings)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重，使用较小的随机值"""
        init_range = 0.5 / self.embedding_dim
        self.target_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(self, context_indices, target_indices, negative_indices):
        """
        前向传播

        Args:
            context_indices: 上下文词索引 (batch_size, window_size * 2)
            target_indices: 目标词索引 (batch_size,)
            negative_indices: 负采样词索引 (batch_size, negative_samples)

        Returns:
            loss: 损失值
        """
        batch_size = context_indices.size(0)

        # 获取上下文词的嵌入并平均
        # context_indices: (batch_size, window_size * 2)
        context_embeds = self.context_embeddings(context_indices)
        # context_embeds: (batch_size, window_size * 2, embedding_dim)
        context_mean = torch.mean(context_embeds, dim=1)
        # context_mean: (batch_size, embedding_dim)

        # 获取目标词的嵌入
        target_embeds = self.target_embeddings(target_indices)
        # target_embeds: (batch_size, embedding_dim)

        # 获取负采样词的嵌入
        negative_embeds = self.target_embeddings(negative_indices)
        # negative_embeds: (batch_size, negative_samples, embedding_dim)

        # 计算正样本损失 (log-sigmoid)
        out_pos = torch.sum(target_embeds * context_mean, dim=1)
        loss_pos = F.logsigmoid(out_pos)

        # 计算负样本损失
        out_neg = torch.bmm(negative_embeds, context_mean.unsqueeze(2)).squeeze()
        # out_neg: (batch_size, negative_samples)
        loss_neg = torch.sum(F.logsigmoid(-out_neg), dim=1)

        # 总损失 = - (正样本损失 + 负样本损失)
        loss = -(loss_pos + loss_neg)

        return loss.mean()

    def get_embeddings(self):
        """获取训练好的词嵌入"""
        return self.target_embeddings.weight.data.cpu().numpy()


class CBOWDataset(torch.utils.data.Dataset):
    """
    CBOW数据集封装
    """

    def __init__(self, context_ids, target_ids, vocab_size, negative_samples=5,
                 unigram_distribution=None):
        """
        初始化数据集

        Args:
            context_ids: 上下文词索引列表
            target_ids: 目标词索引列表
            vocab_size: 词汇表大小
            negative_samples: 负采样数量
            unigram_distribution: unigram概率分布
        """
        self.context_ids = context_ids
        self.target_ids = target_ids
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples
        self.unigram_distribution = unigram_distribution

    def __len__(self):
        return len(self.context_ids)

    def __getitem__(self, idx):
        context = self.context_ids[idx]
        target = self.target_ids[idx]

        # 负采样
        if self.unigram_distribution is not None:
            negative = np.random.choice(
                self.vocab_size,
                size=self.negative_samples,
                p=self.unigram_distribution
            )
        else:
            negative = np.random.randint(0, self.vocab_size, size=self.negative_samples)

        return (
            torch.LongTensor(context),
            torch.LongTensor([target])[0],
            torch.LongTensor(negative)
        )


def train_epoch(model, dataloader, optimizer, device):
    """
    训练一个epoch

    Args:
        model: CBOW模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 计算设备

    Returns:
        average_loss: 平均损失
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for context, target, negative in dataloader:
        context = context.to(device)
        target = target.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()
        loss = model(context, target, negative)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


if __name__ == '__main__':
    # 简单的测试
    vocab_size = 10000
    embedding_dim = 100
    window_size = 5

    model = CBOW(vocab_size, embedding_dim, window_size)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # 测试前向传播
    batch_size = 32
    context = torch.randint(0, vocab_size, (batch_size, window_size * 2))
    target = torch.randint(0, vocab_size, (batch_size,))
    negative = torch.randint(0, vocab_size, (batch_size, 5))

    loss = model(context, target, negative)
    print(f"Test loss: {loss.item():.4f}")
