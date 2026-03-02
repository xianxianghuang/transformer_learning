"""
数据加载与预处理模块
负责文本分词、词汇表构建、训练数据生成
"""

import re
import pickle
import numpy as np
from collections import Counter


class Corpus:
    """语料库处理类"""

    def __init__(self, path, min_count=5, window_size=5):
        """
        初始化语料库

        Args:
            path: 语料库文件路径
            min_count: 最小词频，低于此频率的词将被过滤
            window_size: 上下文窗口大小
        """
        self.path = path
        self.min_count = min_count
        self.window_size = window_size
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = None
        self.vocab_size = 0
        self.train_data = []

    def preprocess(self, text):
        """
        文本预处理：小写化、去除标点、过滤短词

        Args:
            text: 原始文本

        Returns:
            处理后的词列表
        """
        # 转小写
        text = text.lower()
        # 只保留字母和空格
        text = re.sub(r'[^a-z\s]', '', text)
        # 分词
        words = text.split()
        # 过滤空词和单字符词
        words = [w for w in words if len(w) > 0]
        return words

    def build_vocab(self):
        """
        从语料库构建词汇表
        """
        print("Loading corpus...")
        with open(self.path, 'r', encoding='utf-8') as f:
            text = f.read()

        words = self.preprocess(text)
        print(f"Total words: {len(words)}")

        # 统计词频
        word_counts = Counter(words)
        print(f"Unique words before filtering: {len(word_counts)}")

        # 过滤低频词
        self.word_counts = {word: count for word, count in word_counts.items()
                          if count >= self.min_count}
        print(f"Vocabulary size after filtering (min_count={self.min_count}): {len(self.word_counts)}")

        # 构建词到索引的映射
        self.word2idx = {word: idx for idx, word in enumerate(self.word_counts.keys())}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        return self.word2idx, self.idx2word

    def generate_training_data(self):
        """
        生成训练数据 (context, target) 对

        Returns:
            context_ids: 上下文词索引数组
            target_ids: 目标词索引数组
        """
        print("Generating training data...")

        with open(self.path, 'r', encoding='utf-8') as f:
            text = f.read()

        words = self.preprocess(text)

        # 将词转换为索引（过滤不在词汇表中的词）
        word_ids = [self.word2idx[w] for w in words if w in self.word2idx]

        context_ids = []
        target_ids = []

        # 生成(context, target)对
        for i in range(self.window_size, len(word_ids) - self.window_size):
            target = word_ids[i]
            # 获取上下文词（窗口内除目标词外的所有词）
            context = []
            for j in range(-self.window_size, self.window_size + 1):
                if j != 0:
                    context.append(word_ids[i + j])

            context_ids.append(context)
            target_ids.append(target)

        print(f"Training samples: {len(context_ids)}")
        self.train_data = (context_ids, target_ids)

        return context_ids, target_ids

    def save(self, path):
        """保存词汇表和训练数据"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_counts': self.word_counts,
                'vocab_size': self.vocab_size,
                'train_data': self.train_data
            }, f)
        print(f"Corpus saved to {path}")

    def load(self, path):
        """加载词汇表和训练数据"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.word_counts = data['word_counts']
        self.vocab_size = data['vocab_size']
        self.train_data = data['train_data']

        print(f"Corpus loaded from {path}")
        print(f"Vocabulary size: {self.vocab_size}")

        return self.word2idx, self.idx2word


def get_unigram_distribution(word_counts, vocab_size, power=0.75):
    """
    获取用于负采样的unigram分布

    Args:
        word_counts: 词频字典
        vocab_size: 词汇表大小
        power: 概率分布的幂次

    Returns:
        归一化的概率分布数组
    """
    word_counts_list = []
    for i in range(vocab_size):
        word_counts_list.append(word_counts.get(i, 0))

    word_counts_array = np.array(word_counts_list, dtype=np.float32)

    # 应用幂次变换
    word_counts_array = np.power(word_counts_array, power)

    # 归一化
    word_counts_array = word_counts_array / word_counts_array.sum()

    return word_counts_array


if __name__ == '__main__':
    # 测试数据预处理
    corpus = Corpus('data/ptb.txt', min_count=5, window_size=5)
    corpus.build_vocab()
    corpus.generate_training_data()
    corpus.save('data/corpus.pkl')
