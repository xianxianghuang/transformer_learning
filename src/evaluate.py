"""
CBOW模型评估脚本

本脚本提供以下评估功能：
1. 加载训练好的模型
2. 词相似度计算：找出与给定词最相似的词
3. 词类比任务：解决 "a : b :: c : ?" 形式的问题
4. t-SNE可视化：将高维词向量降到2维进行可视化

评估指标：
- 词相似度：语义相近的词应该有相近的向量
- 词类比：向量运算，如 king - man + woman ≈ queen
- 可视化：直观查看词向量的分布

核心知识点：
- 余弦相似度：衡量两个向量方向相似程度的指标
- 词类比：通过向量运算发现词的语义关系
- t-SNE：非线性降维算法，适合可视化高维数据
"""

# ==================== 导入部分 ====================

import os             # 文件路径操作
import argparse       # 命令行参数解析
import numpy as np    # 数值计算
import torch          # PyTorch（用于加载模型）
from sklearn.manifold import TSNE  # t-SNE降维
import matplotlib.pyplot as plt   # 可视化绘图


# ==================== 模型加载函数 ====================

def load_model(model_path, device='cpu'):
    """
    加载训练好的CBOW模型

    参数：
    - model_path: 模型文件路径（.pt 文件）
    - device: 加载到哪个设备

    返回值：
    - embeddings: 词嵌入矩阵，形状 (vocab_size, embedding_dim)
    - word2idx: 词到索引的映射字典
    - idx2word: 索引到词的映射字典

    知识点：
    - torch.load：从文件加载PyTorch模型
    - map_location：指定加载到的设备
    - weights_only=False：允许加载包含Python对象的检查点
    - 模型保存格式：字典，包含 model_state_dict 等
    """
    # 加载模型检查点
    # weights_only=False：允许加载包含自定义Python对象的字典
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 从检查点提取词汇表映射
    word2idx = checkpoint['word2idx']
    idx2word = checkpoint['idx2word']

    # 提取词嵌入矩阵
    # 从模型权重中获取 target_embeddings 的权重
    # shape: (vocab_size, embedding_dim)
    embeddings = checkpoint['model_state_dict']['target_embeddings.weight'].cpu().numpy()

    # 打印模型信息
    print(f"Model loaded from {model_path}")
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    return embeddings, word2idx, idx2word


# ==================== 余弦相似度 ====================

def cosine_similarity(v1, v2):
    """
    计算两个向量的余弦相似度

    余弦相似度公式：
    cos(θ) = (A · B) / (||A|| * ||B||)

    其中：
    - A · B：向量点积（各元素相乘后求和）
    - ||A||：向量A的L2范数（欧几里得范数），即 sqrt(sum(x^2))

    取值范围：[-1, 1]
    - 1：完全相同方向
    - 0：正交（无关联）
    - -1：完全相反方向

    为什么用余弦相似度？
    - 只关心方向，不关心 magnitude（长度）
    - 词向量的长度可能不同，但方向更重要

    参数：
    - v1, v2: 两个 numpy 一维数组

    返回值：
    - 余弦相似度（float，-1到1之间）
    """
    # 计算点积
    # numpy.dot：对两个向量逐元素相乘后求和
    # 等价于 v1[0]*v2[0] + v1[1]*v2[1] + ...
    dot = np.dot(v1, v2)

    # 计算L2范数（向量长度）
    # numpy.linalg.norm：计算矩阵/向量的范数
    # 默认是L2范数：sqrt(sum(x^2))
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    # 防止除零错误
    # 如果向量长度为0（不应该发生），返回0
    if norm1 == 0 or norm2 == 0:
        return 0

    # 计算余弦相似度
    return dot / (norm1 * norm2)


# ==================== 找相似词 ====================

def find_most_similar(word, embeddings, word2idx, idx2word, top_n=10):
    """
    找出与给定词最相似的词

    算法：
    1. 获取给定词的词向量
    2. 计算该词与所有其他词的余弦相似度
    3. 排序，返回最相似的top_n个词

    参数：
    - word: 目标词（字符串）
    - embeddings: 词嵌入矩阵
    - word2idx: 词到索引映射
    - idx2word: 索引到词映射
    - top_n: 返回前n个最相似的词

    返回值：
    - 列表，每个元素是 (词, 相似度) 元组
    """
    # 检查词是否在词汇表中
    if word not in word2idx:
        print(f"Word '{word}' not in vocabulary!")
        return []

    # 获取词的索引
    word_idx = word2idx[word]

    # 获取词的向量表示
    # embeddings[word_idx] 是一个 (embedding_dim,) 的向量
    word_vec = embeddings[word_idx]

    # 计算与所有词的相似度
    similarities = []

    # 遍历所有词
    for idx in range(len(embeddings)):
        # 跳过自己（相似度肯定是1）
        if idx != word_idx:
            # 计算余弦相似度
            sim = cosine_similarity(word_vec, embeddings[idx])
            # 保存（词，相似度）对
            similarities.append((idx2word[idx], sim))

    # 按相似度降序排序
    # sort(key=lambda x: x[1], reverse=True)
    # key：排序依据（第二个元素，即相似度）
    # reverse=True：降序（从大到小）
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 返回前top_n个
    return similarities[:top_n]


# ==================== 词类比任务 ====================

def word_analogy(a, b, c, embeddings, word2idx, idx2word, top_n=5):
    """
    词类比任务：a is to b as c is to ?

    经典例子：
    - man : king :: woman : ?
    - 期望答案：queen

    原理（向量运算）：
    如果 king - man + woman ≈ queen
    那么 king - man ≈ queen - woman

    这意味着：
    - "国王"和"男人"的关系类似于"王后"和"女人"的关系
    - 去掉"男性"特性，加上"女性"特性

    计算步骤：
    1. 计算目标向量：target = b - a + c
    2. 找与目标向量最相似的词

    参数：
    - a, b, c: 已知词
    - embeddings: 词嵌入矩阵
    - word2idx, idx2word: 映射表
    - top_n: 返回前n个结果

    返回值：
    - 列表，每个元素是 (词, 相似度) 元组
    """
    # 检查词是否在词汇表中
    for word in [a, b, c]:
        if word not in word2idx:
            print(f"Word '{word}' not in vocabulary!")
            return []

    # 获取词的向量
    a_vec = embeddings[word2idx[a]]
    b_vec = embeddings[word2idx[b]]
    c_vec = embeddings[word2idx[c]]

    # 计算目标向量
    # 原理：b - a + c
    # b - a：表示 a→b 的语义关系
    # + c：把这种关系应用到 c 上
    target_vec = b_vec - a_vec + c_vec

    # 找出最相似的词
    similarities = []

    for idx in range(len(embeddings)):
        # 排除已知词 a, b, c
        if idx not in [word2idx[a], word2idx[b], word2idx[c]]:
            sim = cosine_similarity(target_vec, embeddings[idx])
            similarities.append((idx2word[idx], sim))

    # 排序
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_n]


# ==================== t-SNE 可视化 ====================

def visualize_embeddings(embeddings, idx2word, words=None, output_path='visualization.png'):
    """
    使用t-SNE可视化词嵌入

    t-SNE (t-Distributed Stochastic Neighbor Embedding)：
    - 非线性降维算法
    - 保持高维空间中相似点之间的距离
    - 将高维数据降到2-3维用于可视化

    原理：
    - 把高维向量转换为低维向量
    - 尽量保持原始的相似性关系
    - 适合揭示聚类结构

    参数：
    - embeddings: 词嵌入矩阵
    - idx2word: 索引到词的映射
    - words: 要可视化的词列表（None表示全部）
    - output_path: 输出图片路径
    """
    # 选择要可视化的词
    if words is not None:
        # 只可视化指定的词
        # 找出这些词在嵌入矩阵中的索引
        word_indices = [i for i, w in idx2word.items() if w in words]

        # 提取这些词的嵌入
        embeddings = embeddings[word_indices]

        # 对应的词标签
        labels = [idx2word[i] for i in word_indices]
    else:
        # 可视化所有词
        labels = [idx2word[i] for i in range(len(embeddings))]

    print(f"Visualizing {len(labels)} words with t-SNE...")

    # t-SNE降维
    # 参数：
    # - n_components：输出维度（2表示2D）
    # - random_state：随机种子，保证可复现
    # - perplexity：困惑度，与近邻数相关
    #   值越大考虑越多的全局结构
    #   一般设为 5-50，对小数据集要更小
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))

    # 执行降维
    # 输入：(n_samples, embedding_dim)
    # 输出：(n_samples, 2)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 绘制散点图
    plt.figure(figsize=(12, 12))  # 图像大小

    # 绘制散点
    # embeddings_2d[:, 0]：所有点的x坐标
    # embeddings_2d[:, 1]：所有点的y坐标
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)

    # 添加词标签
    # 为了避免太拥挤，只标注部分词
    for i, label in enumerate(labels):
        # 每隔一定数量标注一个
        if i % max(1, len(labels) // 50) == 0:
            # plt.annotate：在指定位置添加文本
            plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

    # 设置标题和坐标轴标签
    plt.title('Word Embeddings (t-SNE)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')

    # 保存图片
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()  # 关闭图片释放内存

    print(f"Visualization saved to {output_path}")


# ==================== 相似度评估 ====================

def evaluate_similarity(embeddings, word2idx, idx2word):
    """
    评估词相似度

    测试语义相似的词是否在向量空间中靠近

    参数：
    - embeddings: 词嵌入矩阵
    - word2idx, idx2word: 映射表
    """
    print("\n" + "=" * 50)
    print("Word Similarity Evaluation")
    print("=" * 50)

    # 测试词列表
    test_words = ['king', 'man', 'woman', 'computer', 'money', 'time', 'year']

    # 遍历每个测试词
    for word in test_words:
        # 跳过不在词汇表中的词
        if word not in word2idx:
            continue

        print(f"\nMost similar words to '{word}':")

        # 找最相似的词
        similar_words = find_most_similar(word, embeddings, word2idx, idx2word, top_n=5)

        # 打印结果
        for sim_word, sim_score in similar_words:
            print(f"  {sim_word}: {sim_score:.4f}")


# ==================== 类比评估 ====================

def evaluate_analogy(embeddings, word2idx, idx2word):
    """
    评估词类比任务

    测试模型能否通过向量运算解决类比问题

    参数：
    - embeddings: 词嵌入矩阵
    - word2idx, idx2word: 映射表
    """
    print("\n" + "=" * 50)
    print("Word Analogy Evaluation")
    print("=" * 50)

    # 类比测试用例
    # 格式：(a, b, c, expected_d)
    # 含义：a is to b as c is to ?
    analogies = [
        # 语义类比
        ('man', 'king', 'woman', 'queen'),      # 性别关系
        ('brother', 'sister', 'father', 'mother'),  # 家庭关系
        ('walked', 'walking', 'played', 'playing'),  # 时态变化

        # 语法类比（形容词/副词比较级）
        ('small', 'smaller', 'big', 'bigger'),
        ('good', 'better', 'bad', 'worse'),

        # 语法类比（单复数）
        ('cat', 'cats', 'dog', 'dogs'),
    ]

    # 统计准确率
    correct = 0  # 正确数
    total = 0    # 总数

    # 遍历每个类比
    for a, b, c, expected_d in analogies:
        print(f"\n{a} : {b} :: {c} : ?")

        # 执行类比推理
        results = word_analogy(a, b, c, embeddings, word2idx, idx2word, top_n=5)

        # 检查结果
        if not results:
            print("  (One or more words not in vocabulary)")
            continue

        # 获取预测结果
        predicted = results[0][0]

        # 打印结果
        print(f"  Predicted: {predicted} (expected: {expected_d})")
        print(f"  Top 5: {[f'{w}({s:.3f})' for w, s in results]}")

        # 统计
        if predicted == expected_d:
            correct += 1
        total += 1

    # 打印准确率
    if total > 0:
        accuracy = 100 * correct / total
        print(f"\nAnalogy accuracy: {correct}/{total} ({accuracy:.1f}%)")


# ==================== 参数解析 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Evaluate CBOW model')

    # 模型路径
    parser.add_argument('--model_path', type=str, default='checkpoints/cbow_final.pt',
                       help='Path to trained model')

    # 是否生成可视化
    # action='store_true'：如果指定此参数，值为True
    parser.add_argument('--visualize', action='store_true',
                       help='Generate t-SNE visualization')

    # 设备
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    # 可视化的词列表（逗号分隔）
    # 如：--visualize_words king,queen,man,woman
    parser.add_argument('--visualize_words', type=str, default=None,
                       help='Comma-separated words to visualize')

    return parser.parse_args()


# ==================== 主函数 ====================

def main():
    """主评估函数"""
    # 解析参数
    args = parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        print("Please train the model first: python train.py")
        return

    # 加载模型
    embeddings, word2idx, idx2word = load_model(args.model_path, device)

    # 1. 词相似度评估
    evaluate_similarity(embeddings, word2idx, idx2word)

    # 2. 词类比评估
    evaluate_analogy(embeddings, word2idx, idx2word)

    # 3. 可视化（可选）
    if args.visualize:
        # 确定要可视化的词
        if args.visualize_words:
            # 从命令行参数获取
            words = args.visualize_words.split(',')
        else:
            # 默认词列表
            words = ['king', 'queen', 'man', 'woman', 'computer', 'money',
                    'time', 'year', 'day', 'week', 'month', 'house', 'car',
                    'book', 'school', 'student', 'teacher', 'government']

        # 生成可视化
        visualize_embeddings(embeddings, idx2word, words, output_path='visualization.png')


# ==================== 程序入口 ====================

if __name__ == '__main__':
    main()

    """
    使用示例：

    1. 基本评估（相似度和类比）：
    python evaluate.py

    2. 生成可视化：
    python evaluate.py --visualize

    3. 可视化指定词：
    python evaluate.py --visualize --visualize_words king,queen,man,woman,computer,money

    4. 指定模型路径：
    python evaluate.py --model_path checkpoints/cbow_epoch_10.pt
    """
