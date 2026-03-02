"""
CBOW模型训练脚本
"""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import Corpus, get_unigram_distribution
from model import CBOW, CBOWDataset, train_epoch


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train CBOW model')

    # 数据参数
    parser.add_argument('--data_path', type=str, default='data/ptb.txt',
                       help='Path to training data')
    parser.add_argument('--corpus_path', type=str, default='data/corpus.pkl',
                       help='Path to processed corpus')

    # 模型参数
    parser.add_argument('--embedding_dim', type=int, default=300,
                       help='Embedding dimension')
    parser.add_argument('--window_size', type=int, default=5,
                       help='Context window size')
    parser.add_argument('--min_count', type=int, default=5,
                       help='Minimum word count')
    parser.add_argument('--negative_samples', type=int, default=5,
                       help='Number of negative samples')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save model')

    # 其他
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    # 解析参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载或处理语料库
    if os.path.exists(args.corpus_path):
        print("Loading preprocessed corpus...")
        corpus = Corpus(args.data_path, args.min_count, args.window_size)
        corpus.load(args.corpus_path)
    else:
        print("Processing corpus...")
        corpus = Corpus(args.data_path, args.min_count, args.window_size)
        corpus.build_vocab()
        corpus.generate_training_data()
        corpus.save(args.corpus_path)

    # 获取训练数据
    context_ids, target_ids = corpus.train_data

    # 创建unigram分布（用于负采样）
    unigram_dist = get_unigram_distribution(
        {corpus.word2idx[w]: c for w, c in corpus.word_counts.items()},
        corpus.vocab_size,
        power=0.75
    )

    # 创建数据集和数据加载器
    dataset = CBOWDataset(
        context_ids, target_ids,
        corpus.vocab_size,
        args.negative_samples,
        unigram_dist
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Number of batches: {len(dataloader)}")

    # 创建模型
    model = CBOW(
        vocab_size=corpus.vocab_size,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        negative_samples=args.negative_samples
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 创建优化器
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    # 学习率调度：每3个epoch衰减一半
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # 训练循环
    print("\nStarting training...")
    print("-" * 50)

    for epoch in range(args.epochs):
        # 训练一个epoch
        avg_loss = train_epoch(model, dataloader, optimizer, device)

        # 更新学习率
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        # 保存模型
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(args.save_dir, f'cbow_epoch_{epoch + 1}.pt')
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

    # 保存最终模型
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

    # 保存词向量（用于后续评估）
    embeddings = model.get_embeddings()
    np.save(os.path.join(args.save_dir, 'embeddings.npy'), embeddings)
    print(f"Embeddings saved to {args.save_dir}/embeddings.npy")

    print("\nTraining completed!")


if __name__ == '__main__':
    main()
