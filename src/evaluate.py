"""
CBOW模型评估脚本
提供词相似度计算、词类比任务等功能
"""

import os
import argparse
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def load_model(model_path, device='cpu'):
    """
    加载训练好的模型

    Args:
        model_path: 模型文件路径
        device: 设备

    Returns:
        embeddings: 词嵌入矩阵
        word2idx: 词到索引的映射
        idx2word: 索引到词的映射
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    word2idx = checkpoint['word2idx']
    idx2word = checkpoint['idx2word']

    embeddings = checkpoint['model_state_dict']['target_embeddings.weight'].cpu().numpy()

    print(f"Model loaded from {model_path}")
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    return embeddings, word2idx, idx2word


def cosine_similarity(v1, v2):
    """
    计算两个向量的余弦相似度

    Args:
        v1, v2: 两个向量

    Returns:
        余弦相似度
    """
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot / (norm1 * norm2)


def find_most_similar(word, embeddings, word2idx, idx2word, top_n=10):
    """
    找出与给定词最相似的词

    Args:
        word: 目标词
        embeddings: 词嵌入矩阵
        word2idx: 词到索引的映射
        idx2word: 索引到词的映射
        top_n: 返回前n个最相似的词

    Returns:
        最相似的词列表
    """
    if word not in word2idx:
        print(f"Word '{word}' not in vocabulary!")
        return []

    word_idx = word2idx[word]
    word_vec = embeddings[word_idx]

    similarities = []
    for idx in range(len(embeddings)):
        if idx != word_idx:
            sim = cosine_similarity(word_vec, embeddings[idx])
            similarities.append((idx2word[idx], sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_n]


def word_analogy(a, b, c, embeddings, word2idx, idx2word, top_n=5):
    """
    词类比任务: a is to b as c is to ?

    使用向量运算: b - a + c ≈ d

    Args:
        a, b, c: 已知词
        embeddings: 词嵌入矩阵
        word2idx: 词到索引的映射
        idx2word: 索引到词的映射
        top_n: 返回前n个结果

    Returns:
        类比结果列表
    """
    # 检查词是否在词汇表中
    for word in [a, b, c]:
        if word not in word2idx:
            print(f"Word '{word}' not in vocabulary!")
            return []

    # 计算目标向量: b - a + c
    a_vec = embeddings[word2idx[a]]
    b_vec = embeddings[word2idx[b]]
    c_vec = embeddings[word2idx[c]]

    target_vec = b_vec - a_vec + c_vec

    # 找出最相似的词
    similarities = []
    for idx in range(len(embeddings)):
        if idx not in [word2idx[a], word2idx[b], word2idx[c]]:
            sim = cosine_similarity(target_vec, embeddings[idx])
            similarities.append((idx2word[idx], sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_n]


def visualize_embeddings(embeddings, idx2word, words=None, output_path='visualization.png'):
    """
    使用t-SNE可视化词嵌入

    Args:
        embeddings: 词嵌入矩阵
        idx2word: 索引到词的映射
        words: 要可视化的词列表（None表示可视化所有词）
        output_path: 输出文件路径
    """
    if words is not None:
        # 只可视化指定的词
        word_indices = [i for i, w in idx2word.items() if w in words]
        embeddings = embeddings[word_indices]
        labels = [idx2word[i] for i in word_indices]
    else:
        # 可视化所有词（太多可能看不清）
        labels = [idx2word[i] for i in range(len(embeddings))]

    print(f"Visualizing {len(labels)} words with t-SNE...")

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # 绘制
    plt.figure(figsize=(12, 12))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)

    # 只标注部分词以避免过于拥挤
    for i, label in enumerate(labels):
        if i % max(1, len(labels) // 50) == 0:  # 每隔一定数量标注一个
            plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

    plt.title('Word Embeddings (t-SNE)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to {output_path}")


def evaluate_similarity(embeddings, word2idx, idx2word):
    """
    评估词相似度

    Args:
        embeddings: 词嵌入矩阵
        word2idx: 词到索引的映射
        idx2word: 索引到词的映射
    """
    print("\n" + "=" * 50)
    print("Word Similarity Evaluation")
    print("=" * 50)

    # 测试词列表
    test_words = ['king', 'man', 'woman', 'computer', 'money', 'time', 'year']

    for word in test_words:
        if word not in word2idx:
            continue

        print(f"\nMost similar words to '{word}':")
        similar_words = find_most_similar(word, embeddings, word2idx, idx2word, top_n=5)

        for sim_word, sim_score in similar_words:
            print(f"  {sim_word}: {sim_score:.4f}")


def evaluate_analogy(embeddings, word2idx, idx2word):
    """
    评估词类比任务

    Args:
        embeddings: 词嵌入矩阵
        word2idx: 词到索引的映射
        idx2word: 索引到词的映射
    """
    print("\n" + "=" * 50)
    print("Word Analogy Evaluation")
    print("=" * 50)

    # 经典类比测试
    analogies = [
        # 语义类比
        ('man', 'king', 'woman', 'queen'),
        ('brother', 'sister', 'father', 'mother'),
        ('walked', 'walking', 'played', 'playing'),
        # 语法类比
        ('small', 'smaller', 'big', 'bigger'),
        ('good', 'better', 'bad', 'worse'),
        ('cat', 'cats', 'dog', 'dogs'),
    ]

    correct = 0
    total = 0

    for a, b, c, expected_d in analogies:
        print(f"\n{a} : {b} :: {c} : ?")
        results = word_analogy(a, b, c, embeddings, word2idx, idx2word, top_n=5)

        if not results:
            print("  (One or more words not in vocabulary)")
            continue

        predicted = results[0][0]
        print(f"  Predicted: {predicted} (expected: {expected_d})")
        print(f"  Top 5: {[f'{w}({s:.3f})' for w, s in results]}")

        if predicted == expected_d:
            correct += 1
        total += 1

    if total > 0:
        print(f"\nAnalogy accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Evaluate CBOW model')

    parser.add_argument('--model_path', type=str, default='checkpoints/cbow_final.pt',
                       help='Path to trained model')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate t-SNE visualization')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--visualize_words', type=str, default=None,
                       help='Comma-separated words to visualize')

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 加载模型
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        print("Please train the model first: python train.py")
        return

    embeddings, word2idx, idx2word = load_model(args.model_path, device)

    # 词相似度评估
    evaluate_similarity(embeddings, word2idx, idx2word)

    # 词类比评估
    evaluate_analogy(embeddings, word2idx, idx2word)

    # 可视化（可选）
    if args.visualize:
        if args.visualize_words:
            words = args.visualize_words.split(',')
        else:
            # 使用一些常见词进行可视化
            words = ['king', 'queen', 'man', 'woman', 'computer', 'money',
                    'time', 'year', 'day', 'week', 'month', 'house', 'car',
                    'book', 'school', 'student', 'teacher', 'government']

        visualize_embeddings(embeddings, idx2word, words, output_path='visualization.png')


if __name__ == '__main__':
    main()
