"""
全面评估CBOW词向量模型
包括：词相似度、词类比、聚类分析、可视化

本脚本用于评估训练好的词向量模型的质量，通过多种方式验证模型是否学到了有意义的语义关系。

知识点：
- 余弦相似度：衡量两个向量方向相似程度的指标，范围[-1, 1]，越接近1表示越相似
- 词类比：经典的语言学任务，通过向量运算推理词关系 (king - man + woman ≈ queen)
- t-SNE：降维可视化技术，将高维向量映射到2D空间，保持相似词的邻近关系
- K-means：聚类算法，将词按照语义相似度自动分组
"""

import torch
import numpy as np
import argparse
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os


def load_model(model_path):
    """
    加载训练好的模型

    知识点：
    - PyTorch的torch.load用于加载保存的模型
    - weights_only=False允许加载包含Python对象的检查点（如word2idx字典）
    - 模型权重通常保存在GPU上，需要.cpu()转换到CPU进行计算
    """
    # weights_only=False: 允许加载包含自定义Python对象的检查点
    checkpoint = torch.load(model_path, weights_only=False)

    # 从检查点中提取词到索引的映射字典
    word2idx = checkpoint['word2idx']

    # 从检查点中提取索引到词的映射字典
    idx2word = checkpoint['idx2word']

    # 从模型权重中提取目标词的嵌入矩阵
    # .cpu()将张量从GPU转换到CPU
    # .numpy()将PyTorch张量转换为NumPy数组便于后续计算
    embeddings = checkpoint['model_state_dict']['target_embeddings.weight'].cpu().numpy()

    return embeddings, word2idx, idx2word


def cosine_similarity(v1, v2):
    """
    计算两个向量的余弦相似度

    知识点：
    - 余弦相似度只关注方向，不关注 magnitude（长度）
    - 公式: cos(θ) = (A·B) / (||A|| * ||B||)
    - 范围: [-1, 1]，1表示完全相同方向，-1表示完全相反

    参数:
        v1, v2: 两个向量

    返回:
        余弦相似度值
    """
    # np.dot: 向量点积
    # np.linalg.norm: 向量的L2范数（长度）
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)


def most_similar(word, embeddings, word2idx, idx2word, top_k=10):
    """
    查找与给定词最相似的词

    知识点：
    - 通过计算目标词与所有其他词的余弦相似度来查找相似词
    - 使用矩阵乘法一次性计算所有词的相似度（向量化加速）

    参数:
        word: 要查找的词
        embeddings: 词嵌入矩阵，形状为 (词汇表大小, 嵌入维度)
        word2idx: 词到索引的映射字典
        idx2word: 索引到词的映射字典
        top_k: 返回前k个最相似的词

    返回:
        相似词列表，每个元素为 (词, 相似度) 元组
    """
    # 检查词是否在词汇表中
    if word not in word2idx:
        return f"'{word}' not in vocabulary"

    # 获取词的索引
    idx = word2idx[word]

    # 获取该词的嵌入向量
    word_vec = embeddings[idx]

    # 计算所有词的L2范数（用于归一化）
    norms = np.linalg.norm(embeddings, axis=1)

    # 计算余弦相似度
    # 矩阵乘法 @ 同时计算目标词与所有词的点积
    # 除以各自的范数实现归一化
    cos_sim = (embeddings @ word_vec) / (norms * np.linalg.norm(word_vec) + 1e-10)

    # np.argsort: 返回排序后的索引（从小到大）
    # [::-1]: 逆转数组，实现从大到小排序
    # [1:top_k+1]: 跳过第一个（自己），取前k个
    top_indices = np.argsort(cos_sim)[::-1][1:top_k+1]

    # 将索引转换回词，并返回相似度
    return [(idx2word[i], cos_sim[i]) for i in top_indices]


def word_analogy(a, b, c, embeddings, word2idx, idx2word):
    """
    解决词类比任务: a : b :: c : ?

    知识点：
    - 词类比是Word2Vec的经典评估任务
    - 原理：词的语义关系可以通过向量运算表示
    - 例如：king - man + woman ≈ queen
    - 因为：king : man ≈ queen : woman（性别关系）

    参数:
        a, b, c: 已知词
        embeddings: 词嵌入矩阵
        word2idx, idx2word: 词映射

    返回:
        预测的词列表，最高相似度分数
    """
    # 检查所有词是否都在词汇表中
    if a not in word2idx or b not in word2idx or c not in word2idx:
        return None, "Some words not in vocabulary"

    # 获取三个词的嵌入向量
    vec_a = embeddings[word2idx[a]]
    vec_b = embeddings[word2idx[b]]
    vec_c = embeddings[word2idx[c]]

    # 计算目标词向量
    # 原理：b - a 表示 "a到b的语义关系"
    # 加上c表示将这种关系应用到c上
    vec_result = vec_b - vec_a + vec_c

    # 归一化并计算相似度（与most_similar相同）
    norms = np.linalg.norm(embeddings, axis=1)
    cos_sim = (embeddings @ vec_result) / (norms * np.linalg.norm(vec_result) + 1e-10)

    # 取最相似的5个词
    top_indices = np.argsort(cos_sim)[::-1][:5]

    return [idx2word[i] for i in top_indices], cos_sim[top_indices[0]]


def evaluate_similarity(embeddings, word2idx, idx2word, test_words):
    """
    评估词相似度

    知识点：
    - 词相似度是最直接的词向量质量指标
    - 语义相似的词应该有更高的余弦相似度
    """
    print("\n" + "="*60)
    print("词相似度评估")  # 标题
    print("="*60)

    for word in test_words:
        if word in word2idx:
            # 查找最相似的词
            results = most_similar(word, embeddings, word2idx, idx2word, top_k=5)
            print(f"\n{word}:")  # 打印目标词
            for w, sim in results:
                print(f"  {w}: {sim:.4f}")  # 打印相似词和相似度


def evaluate_analogies(embeddings, word2idx, idx2word, analogy_tests):
    """
    评估词类比任务

    知识点：
    - 词类比任务检验模型是否理解了词之间的关系
    - 常见类比类型：
      * 语义类比：king→queen, man→woman
      * 语法类比：walk→walking, play→playing
    """
    print("\n" + "="*60)
    print("词类比评估")  # 标题
    print("="*60)

    correct = 0
    total = 0

    for a, b, c, expected in analogy_tests:
        # 执行类比推理
        result, score = word_analogy(a, b, c, embeddings, word2idx, idx2word)

        if result is None:
            print(f"\n{a} : {b} :: {c} : ?")
            print(f"  错误: {result}")
            continue

        # 检查期望的词是否在预测结果中
        is_correct = expected in result
        total += 1
        if is_correct:
            correct += 1

        print(f"\n{a} : {b} :: {c} : ?")
        print(f"  预测: {result[0]} (期望: {expected})")
        print(f"  候选: {result}")

    print(f"\n准确率: {correct}/{total} ({100*correct/total:.1f}%)")
    return correct, total


def visualize_tsne(embeddings, word2idx, idx2word, words, save_path):
    """
    使用t-SNE进行降维可视化

    知识点：
    - t-SNE (t-distributed Stochastic Neighbor Embedding)
    - 将高维词向量降到2维用于可视化
    - 原理：保持相似点在低维空间的邻近关系
    - 优点：能很好地展示聚类结构
    - 缺点：计算量大，不保持全局距离
    """
    # 提取指定词的嵌入
    # 列表推导式：过滤出在词汇表中的词
    word_indices = [word2idx[w] for w in words if w in word2idx]
    valid_words = [w for w in words if w in word2idx]

    if len(word_indices) < 2:
        print("Not enough words for visualization")
        return

    # 从嵌入矩阵中提取这些词的向量
    selected_embeds = embeddings[word_indices]

    # 创建t-SNE降维器
    # n_components: 输出维度（2D可视化）
    # random_state: 随机种子，保证结果可复现
    # perplexity: 考虑近邻数量，通常设为5-50
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(valid_words)-1))

    # 执行降维
    coords = tsne.fit_transform(selected_embeds)

    # 绘制散点图
    plt.figure(figsize=(12, 10))
    plt.scatter(coords[:, 0], coords[:, 1], c='blue', s=100, alpha=0.6)

    # 为每个点添加标签
    for i, word in enumerate(valid_words):
        plt.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=12,
                    xytext=(5, 5), textcoords='offset points')

    plt.title('Word Embeddings (t-SNE)', fontsize=16)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"t-SNE可视化已保存到: {save_path}")
    plt.close()


def visualize_pca(embeddings, word2idx, idx2word, words, save_path):
    """
    使用PCA进行降维可视化

    知识点：
    - PCA (Principal Component Analysis) 主成分分析
    - 线性降维方法，计算向量在主要方向上的投影
    - 优点：快速，保持全局结构
    - 缺点：可能不如t-SNE清晰
    - explained_variance_ratio_: 表示每个主成分解释的方差比例
    """
    word_indices = [word2idx[w] for w in words if w in word2idx]
    valid_words = [w for w in words if w in word2idx]

    if len(word_indices) < 2:
        print("Not enough words for visualization")
        return

    selected_embeds = embeddings[word_indices]

    # 创建PCA降维器
    pca = PCA(n_components=2)
    coords = pca.fit_transform(selected_embeds)

    # 绘图
    plt.figure(figsize=(12, 10))
    plt.scatter(coords[:, 0], coords[:, 1], c='green', s=100, alpha=0.6)

    for i, word in enumerate(valid_words):
        plt.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=12,
                    xytext=(5, 5), textcoords='offset points')

    plt.title('Word Embeddings (PCA)', fontsize=16)
    # 显示每个主成分解释的方差比例
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"PCA可视化已保存到: {save_path}")
    plt.close()


def cluster_analysis(embeddings, word2idx, idx2word, words, n_clusters=3):
    """
    使用K-means进行聚类分析

    知识点：
    - K-means是无监督聚类算法
    - 将数据分成k个簇，使簇内距离最小化
    - 原理：
      1. 随机选择k个中心点
      2. 将每个点分配给最近的中心
      3. 更新中心点为簇的均值
      4. 重复2-3直到收敛
    """
    print("\n" + "="*60)
    print("聚类分析")  # 标题
    print("="*60)

    word_indices = [word2idx[w] for w in words if w in word2idx]
    valid_words = [w for w in words if w in word2idx]

    if len(word_indices) < n_clusters:
        print("Not enough words for clustering")
        return

    selected_embeds = embeddings[word_indices]

    # 创建K-means聚类器
    # n_clusters: 要聚成的簇数
    # random_state: 随机种子
    # n_init: 初始化次数，选择最优结果
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    # 执行聚类，返回每个点的簇标签
    labels = kmeans.fit_predict(selected_embeds)

    print(f"\n聚成 {n_clusters} 类:")
    for i in range(n_clusters):
        # 找出属于当前簇的所有词
        cluster_words = [valid_words[j] for j in range(len(valid_words)) if labels[j] == i]
        print(f"\n类别 {i}: {cluster_words}")

    return labels


def evaluate_specific_relations(embeddings, word2idx, idx2word):
    """
    评估特定的语义关系

    知识点：
    - 针对特定领域或关系进行评估
    - 可以自定义测试词组来验证模型学到的知识
    - 金融领域、数值单位、动作时态等都是常见的测试维度
    """
    print("\n" + "="*60)
    print("特定语义关系评估")  # 标题
    print("="*60)

    # 定义测试关系：(词列表, 关系名称)
    relations = [
        # 金融相关
        (["million", "billion", "thousand", "hundred", "trillion"], "数值单位"),
        (["bank", "stock", "market", "money", "dollar"], "金融"),
        (["company", "corporation", "firm", "business", "enterprise"], "公司"),
        (["president", "chairman", "ceo", "director", "executive"], "职位"),
        # 动作时态
        (["walk", "walked", "walking", "run", "running"], "动作"),
        (["eat", "ate", "eating", "drink", "drinking"], "动作"),
        # 复数形式
        (["dog", "dogs", "cat", "cats", "bird"], "单复数"),
        # 词性
        (["quick", "quickly", "slow", "slowly", "fast"], "形容词/副词"),
    ]

    for word_group, relation_name in relations:
        print(f"\n--- {relation_name} ---")

        # 过滤出在词汇表中的词
        valid_words = [w for w in word_group if w in word2idx]
        if len(valid_words) < 2:
            print(f"  词汇不在词表中: {word_group}")
            continue

        # 提取这些词的嵌入
        embeddings_subset = embeddings[[word2idx[w] for w in valid_words]]

        # 导入余弦相似度函数
        from sklearn.metrics.pairwise import cosine_similarity

        # 计算相似度矩阵
        sim_matrix = cosine_similarity(embeddings_subset)

        print(f"  词汇: {valid_words}")
        print(f"  相似度矩阵:")
        for i, w1 in enumerate(valid_words):
            # 格式化输出相似度值
            row = [f"{sim_matrix[i][j]:.2f}" for j in range(len(valid_words))]
            print(f"    {w1}: {row}")


def main():
    """
    主函数：执行全面评估

    知识点：
    - argparse: Python标准库，用于解析命令行参数
    - 允许用户自定义模型路径、输出目录等
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='全面评估CBOW词向量')

    # 添加命令行参数
    parser.add_argument('--model_path', type=str, default='checkpoints/cbow_final.pt',
                       help='模型路径')
    parser.add_argument('--visualize', action='store_true',
                       help='生成可视化')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='输出目录')
    args = parser.parse_args()

    # 创建输出目录（如果不存在）
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print(f"加载模型: {args.model_path}")
    embeddings, word2idx, idx2word = load_model(args.model_path)
    print(f"词汇表大小: {len(word2idx)}")
    print(f"嵌入维度: {embeddings.shape[1]}")

    # 1. 词相似度评估
    test_words = ["king", "man", "woman", "computer", "money", "time", "year",
                  "good", "bad", "small", "big", "new", "old"]
    evaluate_similarity(embeddings, word2idx, idx2word, test_words)

    # 2. 词类比评估
    analogy_tests = [
        ("man", "king", "woman", "queen"),
        ("brother", "sister", "father", "mother"),
        ("walked", "walking", "played", "playing"),
        ("small", "smaller", "big", "bigger"),
        ("good", "better", "bad", "worse"),
        ("one", "first", "two", "second"),
    ]
    evaluate_analogies(embeddings, word2idx, idx2word, analogy_tests)

    # 3. 特定语义关系
    evaluate_specific_relations(embeddings, word2idx, idx2word)

    # 4. 聚类分析
    cluster_words = ["king", "queen", "man", "woman", "prince", "princess",
                    "dog", "cat", "animal", "bird", "fish",
                    "money", "bank", "stock", "dollar", "gold",
                    "walk", "run", "jump", "swim", "fly"]
    cluster_analysis(embeddings, word2idx, idx2word, cluster_words, n_clusters=4)

    # 5. 可视化（如果指定了--visualize参数）
    if args.visualize:
        # 选择用于可视化的词
        vis_words = ["king", "queen", "man", "woman", "prince", "princess",
                    "dog", "cat", "bird", "fish", "animal",
                    "money", "bank", "stock", "dollar",
                    "walk", "run", "jump", "swim",
                    "good", "bad", "big", "small",
                    "computer", "software", "internet", "data"]

        # 生成t-SNE可视化
        visualize_tsne(embeddings, word2idx, idx2word, vis_words,
                      os.path.join(args.output_dir, 'tsne_visualization.png'))

        # 生成PCA可视化
        visualize_pca(embeddings, word2idx, idx2word, vis_words,
                     os.path.join(args.output_dir, 'pca_visualization.png'))

    print("\n" + "="*60)
    print("评估完成!")
    print("="*60)


if __name__ == '__main__':
    # 知识点：
    # if __name__ == '__main__': 是Python的入口模式
    # 确保代码只在直接运行脚本时执行，而不是被导入时执行
    main()
