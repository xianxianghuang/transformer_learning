"""
CBOW模型训练脚本

本脚本负责：
1. 解析命令行参数
2. 加载/预处理语料库
3. 创建模型、数据集、数据加载器
4. 执行训练循环
5. 保存模型和词向量

训练流程：
1. 数据准备：加载文本 → 分词 → 构建词汇表 → 生成训练对
2. 模型创建：初始化CBOW模型（随机权重）
3. 训练循环：遍历多个epoch，每个epoch遍历所有batch
4. 模型保存：保存训练好的权重和词汇表

核心知识点：
- 命令行参数解析 (argparse)
- PyTorch DataLoader
- 随机种子 (Random Seed)
- 学习率调度 (Learning Rate Scheduler)
- 模型保存/加载
"""

# ==================== 导入部分 ====================

import os             # 操作系统接口，用于文件路径操作
import argparse        # 命令行参数解析库
import numpy as np    # 数值计算
import torch          # PyTorch核心
import torch.optim as optim  # 优化器（SGD, Adam等）
from torch.utils.data import DataLoader  # 数据加载器

# 导入自定义模块
from data_loader import Corpus, get_unigram_distribution
from model import CBOW, CBOWDataset, train_epoch


# ==================== 参数解析函数 ====================

def parse_args():
    """
    解析命令行参数

    使用 argparse 库解析命令行参数
    允许用户自定义超参数，而不用修改代码

    运行示例：
    python train.py --embedding_dim 300 --epochs 20 --batch_size 512

    知识点：
    - argparse：Python标准库，用于处理命令行参数
    - add_argument：添加一个参数
    - type：参数类型
    - default：默认值
    - help：帮助信息（--help 时显示）
    """
    # 创建ArgumentParser对象
    # description：程序描述
    parser = argparse.ArgumentParser(description='Train CBOW model')

    # ==================== 数据参数 ====================

    # --data_path: 训练数据文件路径
    # 支持三种形式：
    # 1. 单个文件：如 'data/ptb.txt'
    # 2. 多个文件（用逗号分隔）：如 'data/ptb.train.txt,data/ptb.valid.txt'
    # 3. 目录：如 'data/' （目录下所有.txt文件会被合并使用）
    # 注意：路径相对于项目根目录（不是相对于src目录）
    parser.add_argument('--data_path', type=str, default='data/ptb.txt',
                       help='Path to training data (file, comma-separated files, or directory)')

    # --corpus_path: 处理后的语料库文件路径（pickle格式）
    parser.add_argument('--corpus_path', type=str, default='data/corpus.pkl',
                       help='Path to processed corpus')

    # ==================== 模型参数 ====================

    # --embedding_dim: 词向量维度
    # 常用值：100, 200, 300
    # 维度越大表达能力越强，但需要更多训练数据
    parser.add_argument('--embedding_dim', type=int, default=300,
                       help='Embedding dimension')

    # --window_size: 上下文窗口大小
    # 决定用目标词周围多少个词来预测
    parser.add_argument('--window_size', type=int, default=5,
                       help='Context window size')

    # --min_count: 最小词频
    # 词频低于此值的词将被过滤掉
    parser.add_argument('--min_count', type=int, default=5,
                       help='Minimum word count')

    # --negative_samples: 负采样数量
    # 每次训练时采样多少个负样本
    parser.add_argument('--negative_samples', type=int, default=5,
                       help='Number of negative samples')

    # ==================== 训练参数 ====================

    # --batch_size: 批量大小
    # 每次参数更新时处理的样本数
    # 较大的batch_size训练更稳定，但需要更多内存
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size')

    # --learning_rate: 学习率
    # 决定参数更新的步长
    # 学习率太大可能导致不收敛，太小则训练太慢
    # 注意：CBOW模型推荐使用较大的学习率(如10.0)以确保有效训练
    parser.add_argument('--learning_rate', type=float, default=10.0,
                       help='Learning rate (recommended: 10.0 for CBOW)')

    # --epochs: 训练轮数
    # 完整遍历训练数据的次数
    # 更多epoch可以让模型学到更多，但可能过拟合
    # 推荐100个epoch以获得较好的词向量
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (recommended: 100)')

    # --save_dir: 模型保存目录
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save model')

    # ==================== 其他参数 ====================

    # --seed: 随机种子
    # 用于初始化随机数生成器，保证结果可复现
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # --device: 计算设备
    # 'cuda' 使用GPU，'cpu' 使用CPU
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    # 解析并返回参数
    return parser.parse_args()


# ==================== 随机种子设置 ====================

def set_seed(seed):
    """
    设置随机种子，保证实验可复现

    为什么需要随机种子？
    - 神经网络训练涉及很多随机操作：
      * 权重初始化
      * Dropout
      * 数据 shuffle
      * 负采样
    - 设置相同的种子，每次训练结果相同
    - 方便调试和比较

    知识点：
    - numpy.random.seed：NumPy的随机种子
    - torch.manual_seed：PyTorch的CPU随机种子
    - torch.cuda.manual_seed：PyTorch的GPU随机种子
    """
    # 设置NumPy的随机种子
    np.random.seed(seed)

    # 设置PyTorch的随机种子（CPU）
    torch.manual_seed(seed)

    # 如果有GPU，设置GPU的随机种子
    # torch.cuda.is_available() 检查是否有可用的GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# ==================== 主训练函数 ====================

def main():
    """
    主训练函数

    完整的训练流程：
    1. 解析参数
    2. 设置随机种子
    3. 设置计算设备
    4. 创建保存目录
    5. 加载/处理语料库
    6. 创建数据集和数据加载器
    7. 创建模型
    8. 创建优化器和学习率调度器
    9. 训练循环
    10. 保存最终模型
    """

    # ========== 1. 解析参数 ==========
    args = parse_args()


    # ========== 2. 设置随机种子 ==========
    set_seed(args.seed)


    # ========== 3. 设置计算设备 ==========

    # torch.device：PyTorch的设备对象
    # 如果指定 'cuda' 但没有可用GPU，自动回退到 'cpu'
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 打印使用的设备
    print(f"Using device: {device}")

    # 知识点：
    # - CUDA：NVIDIA的GPU计算平台
    # - GPU相比CPU：大规模矩阵运算快很多
    # - 神经网络训练大量使用矩阵运算，适合GPU


    # ========== 4. 创建保存目录 ==========

    # os.makedirs：创建目录
    # exist_ok=True：如果目录已存在，不报错
    os.makedirs(args.save_dir, exist_ok=True)


    # ========== 5. 加载/处理语料库 ==========

    # 逻辑：
    # - 如果已处理过的语料库文件存在，直接加载
    # - 否则，处理原始文本并保存

    # 处理 data_path 参数：
    # 支持三种格式：
    # 1. 目录路径：如 'data/'
    # 2. 逗号分隔的文件列表：如 'data/ptb.train.txt,data/ptb.valid.txt'
    # 3. 单个文件：如 'data/ptb.txt'
    # 注意：这里只需要传递处理后的路径格式给 Corpus 类即可，
    # 因为 Corpus 类内部已经处理了所有这些格式

    if os.path.exists(args.corpus_path):
        # 语料库文件已存在，加载它
        print("Loading preprocessed corpus...")

        # 创建 Corpus 对象（只需提供参数）
        # Corpus 类会自动解析 data_path 的格式
        corpus = Corpus(args.data_path, args.min_count, args.window_size)

        # 从文件加载语料库
        corpus.load(args.corpus_path)
    else:
        # 语料库文件不存在，需要处理
        print("Processing corpus...")

        # 创建 Corpus 对象
        # Corpus 类会自动解析 data_path 的格式：
        # - 如果是目录，自动读取目录下所有.txt文件
        # - 如果是逗号分隔的列表，自动分割成多个文件
        # - 如果是单个文件，直接使用
        corpus = Corpus(args.data_path, args.min_count, args.window_size)

        # 构建词汇表
        corpus.build_vocab()

        # 生成训练数据（上下文-目标词对）
        corpus.generate_training_data()

        # 保存处理后的语料库
        corpus.save(args.corpus_path)


    # ========== 6. 获取训练数据 ==========

    # 从语料库获取训练数据
    # train_data 是一个元组 (context_ids, target_ids)
    context_ids, target_ids = corpus.train_data


    # ========== 7. 创建unigram分布（用于负采样）==========

    # get_unigram_distribution 函数：
    # - 输入：词频字典、词汇表大小、幂次（默认为0.75）
    # - 输出：归一化的概率分布数组
    #
    # 作用：
    # - 负采样时，按照这个概率分布选择负样本
    # - 0.75次幂使分布更平滑，减少高频词的采样优势

    # 转换词频字典格式
    # 原始格式：{词: 频数}
    # 目标格式：{索引: 频数}
    word_counts_dict = {corpus.word2idx[w]: c for w, c in corpus.word_counts.items()}

    unigram_dist = get_unigram_distribution(
        word_counts_dict,      # 词频字典
        corpus.vocab_size,     # 词汇表大小
        power=0.75             # 幂次参数
    )

    # ========== 8. 创建数据集和数据加载器 ==========

    # Dataset：封装训练数据
    dataset = CBOWDataset(
        context_ids,                      # 上下文ID列表
        target_ids,                       # 目标词ID列表
        corpus.vocab_size,                # 词汇表大小
        args.negative_samples,             # 负采样数量
        unigram_dist                      # unigram分布
    )

    # DataLoader：批量加载数据
    # 作用：
    # - 自动把数据分成batch
    # - 可以shuffle（打乱顺序）
    # - 可以多线程加载（num_workers）
    dataloader = DataLoader(
        dataset,               # 数据集对象
        batch_size=args.batch_size,  # 批量大小
        shuffle=True,          # 每个epoch开始时打乱数据
        num_workers=0,         # 数据加载线程数（0表示主线程）
        pin_memory=True if torch.cuda.is_available() else False  # 锁页内存
    )

    # 知识点：
    # - pin_memory：设置为True时，数据加载到锁页内存
    #   锁页内存转移到GPU更快，但占用更多RAM
    # - num_workers：并行加载数据，>0可加速，但可能有问题

    # 打印batch数量
    print(f"Number of batches: {len(dataloader)}")


    # ========== 9. 创建模型 ==========

    # 创建 CBOW 模型实例
    model = CBOW(
        vocab_size=corpus.vocab_size,       # 词汇表大小
        embedding_dim=args.embedding_dim,   # 嵌入维度
        window_size=args.window_size,        # 窗口大小
        negative_samples=args.negative_samples  # 负采样数
    ).to(device)  # 把模型移到指定设备（GPU/CPU）

    # 打印模型参数量
    # sum(p.numel() for p in model.parameters()) 计算总参数数
    # model.parameters() 返回模型所有可学习参数的迭代器
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")


    # ========== 10. 创建优化器 ==========

    # optim.SGD：随机梯度下降优化器
    # 参数：
    # - model.parameters()：要优化的参数
    # - lr：learning rate，学习率
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    # 知识点：
    # - SGD：最基础的优化器
    # - 还有 Adam、RMSprop 等更高级的优化器
    # - Adam 通常收敛更快，是现在的主流选择
    # - 这里用 SGD 是因为原始Word2Vec论文用的SGD


    # ========== 11. 创建学习率调度器 ==========

    # 学习率调度：随着训练进行，动态调整学习率
    # 目的：初期大学习率快速收敛，后期小学习率精细调优

    # optim.lr_scheduler.StepLR：固定间隔衰减学习率
    # 参数：
    # - optimizer：优化器
    # - step_size：每隔多少个epoch衰减一次
    # - gamma：衰减系数（乘以当前学习率）
    #
    # 示例：
    # 初始学习率：0.025
    # 第3个epoch后：0.025 * 0.5 = 0.0125
    # 第6个epoch后：0.0125 * 0.5 = 0.00625
    # 依此类推...
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)


    # ========== 12. 训练循环 ==========

    print("\nStarting training...")
    print("-" * 50)  # 打印分隔线

    # 遍历每个 epoch
    # range(args.epochs) 生成 0, 1, 2, ..., epochs-1
    for epoch in range(args.epochs):

        # 训练一个 epoch
        # train_epoch 函数：
        # - 遍历 dataloader 中的所有 batch
        # - 执行前向传播、反向传播、参数更新
        # - 返回平均损失
        avg_loss = train_epoch(model, dataloader, optimizer, device)

        # 更新学习率调度器
        # scheduler.step() 更新优化器的学习率
        scheduler.step()

        # 获取当前学习率
        # optimizer.param_groups[0]['lr'] 访问当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 打印训练进度
        # epoch + 1：因为epoch从0开始，但通常从1开始计数
        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")


        # ========== 13. 定期保存模型 ==========

        # 每5个epoch保存一次模型
        if (epoch + 1) % 5 == 0:
            # 构造保存路径
            save_path = os.path.join(args.save_dir, f'cbow_epoch_{epoch + 1}.pt')

            # torch.save：保存模型
            # 字典格式，包含：
            # - epoch：当前epoch数
            # - model_state_dict：模型权重
            # - optimizer_state_dict：优化器状态（可继续训练）
            # - loss：当前损失
            # - args：训练参数（方便以后查看）
            # - word2idx, idx2word：词汇表映射
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': args,
                'word2idx': corpus.word2idx,
                'idx2word': corpus.idx2word
            }, save_path)

            print(f"Model saved to {save_path}")


    # ========== 14. 保存最终模型 ==========

    # 训练完成后，保存最终模型
    final_path = os.path.join(args.save_dir, 'cbow_final.pt')

    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'args': args,
        'word2idx': corpus.word2idx,
        'idx2word': corpus.idx2word
    }, final_path)

    print(f"\nFinal model saved to {final_path}")


    # ========== 15. 保存词向量 ==========

    # 获取训练好的词嵌入矩阵
    # shape: (vocab_size, embedding_dim)
    embeddings = model.get_embeddings()

    # 保存为 numpy 数组格式
    # 可以用 np.load 加载
    np.save(os.path.join(args.save_dir, 'embeddings.npy'), embeddings)

    print(f"Embeddings saved to {args.save_dir}/embeddings.npy")


    # 训练完成
    print("\nTraining completed!")


# ==================== 程序入口 ====================

if __name__ == '__main__':
    """
    Python 程序入口

    if __name__ == '__main__' 的作用：
    - 当直接运行此脚本时（python train.py），执行 main()
    - 当作为模块导入时（import train），不自动执行
    - 这是Python的标准做法
    """

    # 调用主函数
    main()


    """
    完整训练流程总结：

    1. 参数解析：
       - 从命令行获取超参数
       - 如：embedding_dim, batch_size, learning_rate 等

    2. 数据准备：
       - 读取文本文件
       - 预处理（分词、清洗）
       - 构建词汇表（过滤低频词）
       - 生成训练对 (context, target)

    3. 模型创建：
       - 创建 CBOW 模型实例
       - 初始化嵌入层权重

    4. 训练循环：
       for epoch in range(epochs):
           for batch in dataloader:
               # 前向传播
               loss = model(context, target, negative)

               # 反向传播
               loss.backward()

               # 更新参数
               optimizer.step()

    5. 保存结果：
       - 保存模型检查点（可继续训练）
       - 保存词嵌入矩阵（用于评估）

    运行示例：
    cd src
    python train.py

    自定义参数：
    python train.py --embedding_dim 300 --epochs 20 --batch_size 512
    """
