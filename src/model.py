"""
CBOW模型实现
使用PyTorch实现带负采样的CBOW模型

CBOW (Continuous Bag of Words) 模型概述：
- 目标：根据上下文词预测目标词
- 输入：周围上下文词的向量表示
- 输出：目标词的预测

模型结构：
1. 词嵌入层 (Embedding Layer)
2. 上下文平均 (Mean Pooling)
3. 负采样损失计算 (Negative Sampling)

核心知识点：
- nn.Embedding：词嵌入层，将索引转换为向量
- Mean Pooling：平均池化，将多个向量合并为一个
- 负采样：一种加速训练的技巧，避免计算全部softmax
- 对数损失 (Log Loss)：二分类交叉熵损失
"""

# ==================== 导入部分 ====================

import numpy as np                    # 数值计算库
import torch                          # PyTorch核心库
import torch.nn as nn                 # 神经网络模块
import torch.nn.functional as F      # 函数式API（如激活函数）
import torch.optim as optim           # 优化器


# ==================== CBOW 模型类 ====================

class CBOW(nn.Module):
    """
    Continuous Bag of Words (CBOW) 模型

    模型结构详解：
    ┌─────────────────────────────────────────────────────────┐
    │                     CBOW 模型                           │
    │                                                         │
    │   输入层              嵌入层              平均池化        │
    │   ┌─────┐           ┌─────┐            ┌─────┐         │
    │   │ w-2 │ ──────►  │ e-2 │ ───────►   │     │          │
    │   ├─────┤           ├─────┤            │     │         │
    │   │ w-1 │ ──────►  │ e-1 │ ───────►   │  v  │          │
    │   ├─────┤           ├─────┤            │     │         │
    │   │ w+1 │ ──────►  │ e+1 │ ───────►   │ avg │ ──► 损失 │
    │   ├─────┤           ├─────┤            │     │         │
    │   │ w+2 │ ──────►  │ e+2 │ ───────►   │     │          │
    │   └─────┘           └─────┘            └─────┘         │
    │                                                         │
    │              目标词嵌入            负采样嵌入            │
    │              ┌─────┐              ┌─────┐               │
    │              │ et  │              │ en  │               │
    │              └─────┘              └─────┘               │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

    训练目标：
    - 正样本：最大化 上下文均值 与 目标词 的相似度
    - 负样本：最小化 上下文均值 与 负采样词 的相似度
    """

    def __init__(self, vocab_size, embedding_dim, window_size, negative_samples=5):
        """
        初始化CBOW模型

        参数说明：
        - vocab_size: 词汇表大小（有多少个不同的词）
        - embedding_dim: 词向量维度（每个词用多少维的向量表示）
          * 常用值：100, 200, 300
          * 维度越大表达能力越强，但需要更多训练数据
        - window_size: 上下文窗口大小（周围看多少个词）
        - negative_samples: 负采样数量
          * 每次训练时采样多少个负样本
          * 常用值：5-10

        知识点：
        - super().__init__()：调用父类 nn.Module 的初始化方法
        - nn.Embedding：创建一个词嵌入表，形状为 (vocab_size, embedding_dim)
        - 两个嵌入层：target_embeddings 和 context_embeddings
          * target：用于目标词的嵌入
          * context：用于上下文词的嵌入
          * 分开可以增加模型容量，但也意味着更多参数
        """
        super(CBOW, self).__init__()  # 初始化父类 nn.Module

        # 保存超参数
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negative_samples = negative_samples

        # 创建词嵌入层
        # nn.Embedding 的作用：把词索引转换为密集向量
        # 参数：(num_embeddings, embedding_dim)
        # num_embeddings = vocab_size：要嵌入的词数量
        # embedding_dim = embedding_dim：每个词的向量维度

        # 目标词嵌入层 (target embeddings)
        # 用于获取目标词的向量表示
        # 形状：(vocab_size, embedding_dim)
        # 例如：vocab_size=10000, embedding_dim=300
        # 则嵌入表大小为 10000 x 300 = 3,000,000 个参数
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # 上下文嵌入层 (context embeddings)
        # 用于获取上下文词的向量表示
        # 形状同上
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # 初始化模型权重
        # 好的初始化可以让模型训练更稳定
        self._init_weights()


    def _init_weights(self):
        """
        初始化权重

        使用均匀分布初始化，范围较小
        公式：uniform(-init_range, init_range)
        其中 init_range = 0.5 / embedding_dim

        为什么要小值初始化？
        - 如果权重太大，sigmoid/tanh函数会进入饱和区
        - 饱和区梯度接近0，导致梯度消失
        - 小值初始化让权重在 sigmoid 的梯度最大区域

        知识点：
        - Xavier初始化：适合 tanh/sigmoid
        - He初始化：适合 ReLU
        - 这里用的是简化的均匀分布初始化
        """
        # 计算初始化范围
        # 使用更大的初始化范围以确保梯度有效传播
        # 经验值：0.1 对于大多数情况效果良好
        init_range = 0.1

        # 初始化目标词嵌入层的权重
        # .weight.data 访问嵌入层的权重参数
        # .uniform_() 原地修改为均匀分布的值
        self.target_embeddings.weight.data.uniform_(-init_range, init_range)

        # 初始化上下文词嵌入层的权重
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)


    def forward(self, context_indices, target_indices, negative_indices):
        """
        前向传播（模型的核心计算逻辑）

        参数说明：
        - context_indices: 上下文词索引
          * 形状：(batch_size, window_size * 2)
          * batch_size：批量大小（一次处理多少样本）
          * window_size * 2：上下文词数量（左右各window_size个）
          * 例如：batch_size=32, window_size=5 → (32, 10)

        - target_indices: 目标词索引
          * 形状：(batch_size,)
          * 例如：(32,)

        - negative_indices: 负采样词索引
          * 形状：(batch_size, negative_samples)
          * 例如：(32, 5)

        返回值：
        - loss: 损失值（标量），越小越好

        知识点：
        - 前向传播：从输入计算输出的过程
        - batch：一次送入模型的一组样本
        - 索引张量：LongTensor类型，不能是FloatTensor
        """
        # 获取批量大小
        batch_size = context_indices.size(0)

        # ========== 第一步：获取上下文词的嵌入并平均 ==========

        # 嵌入层前向传播
        # self.context_embeddings(context_indices)
        # 输入：词索引 (batch_size, window_size * 2)
        # 输出：词向量 (batch_size, window_size * 2, embedding_dim)

        # 示例：
        # context_indices = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]  # 2个样本，各5个上下文词
        # → 嵌入后 → shape = (2, 5, 300)  # 2个样本，5个词，每个300维
        context_embeds = self.context_embeddings(context_indices)

        # Mean Pooling（平均池化）
        # 沿维度1（window_size * 2 维度）取平均
        # 把多个上下文词的向量合并成一个向量

        # torch.mean() 的 dim 参数：
        # dim=1 表示沿着第1维（索引从0开始）计算平均值
        # 输入：(batch_size, window_size * 2, embedding_dim)
        # 输出：(batch_size, embedding_dim)

        # 为什么要平均？
        # CBOW的核心思想是"Bag of Words"
        # 不考虑词的顺序，只把所有上下文词的向量加起来/平均
        context_mean = torch.mean(context_embeds, dim=1)

        # ========== 第二步：获取目标词的嵌入 ==========

        # self.target_embeddings(target_indices)
        # 输入：目标词索引 (batch_size,)
        # 输出：目标词向量 (batch_size, embedding_dim)

        # 示例：
        # target_indices = [10, 11]  # 2个目标词
        # → 嵌入后 → shape = (2, 300)
        target_embeds = self.target_embeddings(target_indices)

        # ========== 第三步：获取负采样词的嵌入 ==========

        # 负采样词的嵌入也使用 target_embeddings 层
        # 这样可以减少参数数量

        # 输入：负样本索引 (batch_size, negative_samples)
        # 输出：负样本向量 (batch_size, negative_samples, embedding_dim)

        # 示例：
        # negative_indices = [[12,13,14,15,16], [17,18,19,20,21]]
        # → 嵌入后 → shape = (2, 5, 300)
        negative_embeds = self.target_embeddings(negative_indices)

        # ========== 第四步：计算正样本损失 ==========

        # 点积计算相似度
        # target_embeds: (batch_size, embedding_dim)
        # context_mean: (batch_size, embedding_dim)
        # * 表示元素-wise乘法（不是矩阵乘法）
        # 结果：(batch_size, embedding_dim)

        # 然后沿 embedding_dim 维度求和
        # torch.sum(..., dim=1) → (batch_size,)
        # 这实际上计算了目标词向量和上下文均值向量的点积

        # 点积的意义：
        # 在向量空间中，相似的词应该有相似的方向
        # 点积越大，表示两个向量越相似
        out_pos = torch.sum(target_embeds * context_mean, dim=1)

        # 计算 log-sigmoid 损失
        # F.logsigmoid(x) = log(sigmoid(x))
        # sigmoid 将值压缩到 (0, 1)
        # log 将范围变为 (-∞, 0)
        # 作用：将点积转化为损失值（越小越好）

        # 公式推导：
        # 我们希望：context 和 target 越相似越好
        # 即：点积越大越好
        # log(sigmoid(x)) 当 x>0 时为负数，x越大越接近0
        # 所以我们要最大化 log(sigmoid(点积))，即最小化 -log(sigmoid(点积))
        loss_pos = F.logsigmoid(out_pos)

        # ========== 第五步：计算负样本损失 ==========

        # 负采样：随机选择一些不是目标的词
        # 我们希望这些负样本和上下文的相似度越低越好

        # torch.bmm：批量矩阵乘法
        # negative_embeds: (batch_size, negative_samples, embedding_dim)
        # context_mean.unsqueeze(2): (batch_size, embedding_dim, 1)
        # 结果：(batch_size, negative_samples, 1)
        # .squeeze() 去掉维度为1的维度 → (batch_size, negative_samples)

        # 示例：
        # negative_embeds: (32, 5, 300)
        # context_mean: (32, 300) → unsqueeze后 (32, 300, 1)
        # bmm 结果: (32, 5, 1) → squeeze后 (32, 5)
        out_neg = torch.bmm(negative_embeds, context_mean.unsqueeze(2)).squeeze()

        # 同样应用 log-sigmoid
        # 但这里用的是 -out_neg
        # 因为我们希望负样本的点积越小越好（越负越好）
        # log(sigmoid(-x)) 当 x>0 时也是负数
        # x越大（负样本越不相似），sigmoid(-x)越接近1，log越接近0
        # 所以 -log(sigmoid(-x)) 越小越好
        loss_neg = torch.sum(F.logsigmoid(-out_neg), dim=1)

        # ========== 第六步：计算总损失 ==========

        # 总损失 = - (正样本损失 + 负样本损失)
        # 负号：因为 log(sigmoid(x)) 本身是负数，我们要最小化损失

        # 公式理解：
        # loss = -[log(sigmoid(pos_score)) + Σ log(sigmoid(-neg_score))]
        # = -log[sigmoid(pos_score) * Π sigmoid(-neg_scores)]
        # 这就是负采样损失（Negative Sampling Loss）

        # 求平均损失
        loss = -(loss_pos + loss_neg)

        # .mean() 对 batch 中所有样本的损失求平均
        return loss.mean()


    def get_embeddings(self):
        """
        获取训练好的词嵌入

        返回值：
        - numpy 数组，形状为 (vocab_size, embedding_dim)

        知识点：
        - .weight.data：访问模型的可学习参数
        - .cpu()：把张量从GPU移到CPU（用于保存/可视化）
        - .numpy()：把PyTorch张量转换为NumPy数组

        为什么要保存嵌入？
        - 用于后续的词相似度计算、类比推理、可视化等
        """
        return self.target_embeddings.weight.data.cpu().numpy()


# ==================== 数据集类 ====================

class CBOWDataset(torch.utils.data.Dataset):
    """
    CBOW数据集封装

    作用：
    - 封装训练数据
    - 提供 PyTorch DataLoader 需要的标准接口
    - 负责负采样

    知识点：
    - Dataset：PyTorch的数据集抽象类
    - 必须实现 __len__ 和 __getitem__ 方法
    - __getitem__ 每次返回一个训练样本
    """

    def __init__(self, context_ids, target_ids, vocab_size, negative_samples=5,
                 unigram_distribution=None):
        """
        初始化数据集

        参数：
        - context_ids: 上下文词索引列表
          * 类型：Python列表
          * 每个元素也是一个列表，表示一个样本的上下文词索引
        - target_ids: 目标词索引列表
          * 类型：Python列表
          * 每个元素是一个整数，表示目标词的索引
        - vocab_size: 词汇表大小
        - negative_samples: 负采样数量
        - unigram_distribution: unigram概率分布（用于加权负采样）
          * 如果为 None，则均匀随机采样
        """
        # 保存参数
        self.context_ids = context_ids
        self.target_ids = target_ids
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples
        self.unigram_distribution = unigram_distribution


    def __len__(self):
        """
        返回数据集大小

        PyTorch DataLoader 需要知道数据集有多少样本
        """
        return len(self.context_ids)


    def __getitem__(self, idx):
        """
        获取指定索引的训练样本

        参数：
        - idx: 样本索引

        返回值：
        - (context, target, negative) 三元组
        - 都是 torch.LongTensor 类型

        知识点：
        - 每次调用时动态进行负采样
        - 这样可以让模型在每个epoch看到不同的负样本
        - 这是一种数据增强技巧
        """
        # 获取索引为 idx 的上下文和目标词
        context = self.context_ids[idx]    # Python列表
        target = self.target_ids[idx]       # Python整数

        # 负采样
        if self.unigram_distribution is not None:
            # 如果提供了unigram分布，按照该分布采样
            # np.random.choice：从给定概率分布中随机选择

            # 参数说明：
            # - self.vocab_size：选择范围（0到词汇表大小-1）
            # - size=self.negative_samples：采样数量
            # - p=self.unigram_distribution：概率分布
            negative = np.random.choice(
                self.vocab_size,
                size=self.negative_samples,
                p=self.unigram_distribution
            )
        else:
            # 如果没有提供分布，使用均匀分布随机采样
            # np.random.randint：随机整数
            negative = np.random.randint(0, self.vocab_size, size=self.negative_samples)

        # 转换为 PyTorch 张量
        # torch.LongTensor：64位整数类型，用于索引
        # 注意：PyTorch 嵌入层要求索引是 LongTensor

        # 返回值：
        # - context: (window_size * 2,) 形状的 LongTensor
        # - target: 标量 LongTensor
        # - negative: (negative_samples,) 形状的 LongTensor
        return (
            torch.LongTensor(context),       # 上下文词索引
            torch.LongTensor([target])[0],   # 目标词索引（标量）
            torch.LongTensor(negative)       # 负样本索引
        )


# ==================== 训练函数 ====================

def train_epoch(model, dataloader, optimizer, device):
    """
    训练一个epoch（遍历一次整个数据集）

    参数：
    - model: CBOW模型
    - dataloader: 数据加载器
    - optimizer: 优化器（如SGD、Adam）
    - device: 计算设备（'cuda' 或 'cpu'）

    返回值：
    - average_loss: 平均损失值

    训练流程（每个batch）：
    1. 把数据从CPU移到GPU（如果使用GPU）
    2. 清零梯度（optimizer.zero_grad()）
    3. 前向传播计算损失（model()）
    4. 反向传播计算梯度（loss.backward()）
    5. 更新模型参数（optimizer.step()）

    知识点：
    - epoch：遍历一次完整数据集
    - batch：一次前向传播处理的一组样本
    - 梯度：损失函数对参数的导数，指向参数应该调整的方向
    - 反向传播：从输出到输入计算梯度的过程
    """
    # 设置模型为训练模式
    # 某些层（如Dropout、BatchNorm）在训练和评估时有不同行为
    model.train()

    # 初始化
    total_loss = 0     # 累计总损失
    num_batches = 0    # 已处理的batch数量

    # 遍历数据加载器
    # dataloader 会自动把数据分成一个个 batch
    for context, target, negative in dataloader:
        # ========== 1. 数据移动到指定设备 ==========

        # .to(device) 把张量移动到 GPU 或 CPU
        # 如果 device='cuda'，数据会移到GPU（加速计算）
        # 如果 device='cpu'，数据保持在CPU
        context = context.to(device)      # 上下文词索引
        target = target.to(device)        # 目标词索引
        negative = negative.to(device)    # 负样本索引


        # ========== 2. 清零梯度 ==========

        # optimizer.zero_grad() 清零所有参数的梯度
        # 原因：PyTorch 默认会累加梯度
        # 如果不清零，梯度会一直累加，导致错误的更新方向
        optimizer.zero_grad()


        # ========== 3. 前向传播 ==========

        # 调用模型的 forward 方法
        # 输入：context, target, negative
        # 输出：损失值 loss
        loss = model(context, target, negative)


        # ========== 4. 反向传播 ==========

        # loss.backward() 计算梯度
        # 反向传播算法：根据损失值，计算每个参数应该调整的方向和幅度
        # 计算结果：每个参数的 .grad 属性被更新
        loss.backward()


        # ========== 5. 更新参数 ==========

        # optimizer.step() 根据梯度更新参数
        # SGD：param = param - learning_rate * gradient
        # 这一步真正改变模型内部的权重
        optimizer.step()


        # ========== 6. 记录损失 ==========

        # loss.item() 获取损失值作为Python数字
        # 避免张量在内存中持续累积
        total_loss += loss.item()

        # 记录已处理的batch数量
        num_batches += 1


    # 返回平均损失
    # 总损失 / batch数量 = 平均每个batch的损失
    return total_loss / num_batches


# ==================== 主程序入口 ====================

if __name__ == '__main__':
    """
    测试代码：验证模型可以正常前向传播
    """
    # 定义参数
    vocab_size = 10000       # 词汇表大小
    embedding_dim = 100       # 嵌入维度
    window_size = 5           # 窗口大小

    # 创建模型
    model = CBOW(vocab_size, embedding_dim, window_size)

    # 打印模型参数量
    # sum(p.numel() for p in model.parameters()) 遍历所有参数
    # numel() 计算参数个数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")

    # 测试前向传播
    # torch.randint：生成随机整数张量
    # 参数：(low, high, size)
    batch_size = 32

    # 随机生成测试数据
    context = torch.randint(0, vocab_size, (batch_size, window_size * 2))
    target = torch.randint(0, vocab_size, (batch_size,))
    negative = torch.randint(0, vocab_size, (batch_size, 5))

    # 前向传播
    loss = model(context, target, negative)

    # 打印损失值
    print(f"Test loss: {loss.item():.4f}")

    """
    知识点总结：

    1. nn.Module：
       - 所有神经网络模型的基类
       - 提供参数管理、梯度计算等功能

    2. nn.Embedding：
       - 词嵌入层
       - 输入：LongTensor（词索引）
       - 输出：(batch_size, embedding_dim)

    3. torch.mean(dim=1)：
       - 平均池化
       - 沿指定维度求平均

    4. 负采样损失：
       - 正样本：希望 context 和 target 相似（点积大）
       - 负样本：希望 context 和 negative 不相似（点积小）

    5. 训练流程：
       - zero_grad → forward → backward → step
    """
