"""
数据加载与预处理模块
负责文本分词、词汇表构建、训练数据生成

本模块是CBOW模型的数据预处理部分，主要功能包括：
1. 文本预处理：清洗、分词
2. 词汇表构建：词频统计、过滤低频词
3. 生成训练数据：上下文-目标词对

知识点：
- One-Hot编码 vs 词索引
- 词汇表（Vocabulary）的概念
- 上下文窗口（Context Window）
- 词频统计与过滤
"""

# ==================== 导入部分 ====================

import os           # 操作系统接口，用于文件路径操作
import re           # 正则表达式库，用于文本清洗
import pickle       # Python对象序列化，用于保存/加载数据
import numpy as np  # 数值计算库
from collections import Counter  # 计数器，用于词频统计


# ==================== Corpus 类 ====================

class Corpus:
    """
    语料库处理类

    作用：封装语料库的所有数据处理功能
    - 加载原始文本
    - 预处理（清洗、分词）
    - 构建词汇表
    - 生成训练数据对（context, target）

    核心属性：
    - word2idx: 词 -> 索引 的映射字典
    - idx2word: 索引 -> 词 的映射字典
    - word_counts: 词频字典
    - vocab_size: 词汇表大小
    - train_data: 训练数据（上下文ID列表, 目标词ID列表）
    """

    def __init__(self, path, min_count=5, window_size=5):
        """
        初始化语料库

        参数说明：
        - path: 语料库文件路径
          * 支持三种形式：
          * 1. 单个文件路径：如 'data/ptb.txt'
          * 2. 文件路径列表：如 ['data/ptb.train.txt', 'data/ptb.valid.txt']
          * 3. 目录路径：如 'data/' （目录下所有.txt文件会被合并使用）
          * 注意：当传入目录时，只会读取以.txt结尾的文件
        - min_count: 最小词频阈值
          * 低于此频率的词将被过滤掉
          * 作用：去除生僻词/噪音词，减少词汇表大小
          * 默认值5表示：词频小于5的词不进入词汇表
        - window_size: 上下文窗口大小
          * 决定CBOW模型看多少个上下文词来预测目标词
          * window_size=5 表示：目标词左右各5个词，共10个上下文词

        知识点：
        - 超参数（Hyperparameter）：需要人工设定的参数
        - 窗口大小影响：更大的窗口捕获语义相关性，但增加计算量
        - 多文件支持：可以同时使用训练集、验证集、测试集来构建更大词汇表
        """
        # 处理path参数，支持四种输入形式
        # 1. 单个字符串：检查是文件还是目录
        # 2. 逗号分隔的字符串：如 'file1.txt,file2.txt'
        # 3. 列表：直接使用多个文件
        # 4. 空字符串或None：用于从pickle加载时的情况
        if isinstance(path, str):
            # 检查是否是逗号分隔的多个文件
            if ',' in path:
                # 逗号分隔：分割成多个文件路径
                potential_paths = [p.strip() for p in path.split(',')]
                self.paths = [p for p in potential_paths if p]  # 过滤空字符串
            elif os.path.isdir(path):
                # 目录：获取目录下所有.txt文件
                self.paths = [os.path.join(path, f) for f in os.listdir(path)
                              if f.endswith('.txt') and os.path.isfile(os.path.join(path, f))]
                self.paths.sort()  # 排序保证顺序一致
            elif os.path.isfile(path):
                # 单个文件
                self.paths = [path]
            elif path == '' or path is None:
                # 空路径：用于从pickle加载时的初始化
                self.paths = []
            else:
                raise FileNotFoundError(f"Path not found: {path}")
        elif isinstance(path, list):
            # 文件列表：检查所有文件是否存在
            self.paths = path
            for p in self.paths:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"File not found: {p}")
        else:
            raise ValueError("path must be a string (file/directory path) or list of file paths")

        # 打印加载的文件信息
        if len(self.paths) == 1:
            print(f"Loading corpus from: {self.paths[0]}")
        else:
            print(f"Loading corpus from {len(self.paths)} files:")
            for p in self.paths:
                print(f"  - {p}")

        self.min_count = min_count          # 最小词频阈值
        self.window_size = window_size      # 上下文窗口大小

        # 初始化词汇表相关变量（空字典）
        self.word2idx = {}   # 词 -> 索引映射，如 {"the": 0, "cat": 1, ...}
        self.idx2word = {}   # 索引 -> 词映射，如 {0: "the", 1: "cat", ...}
        self.word_counts = None   # 词频字典，存储每个词出现的次数
        self.vocab_size = 0        # 词汇表大小（总共有多少个不同的词）

        # 训练数据初始化为空列表
        # 存储格式: ([context_ids], [target_ids])
        # context_ids: 上下文词的索引列表
        # target_ids: 目标词的索引列表
        self.train_data = []


    def preprocess(self, text):
        """
        文本预处理：小写化、去除标点、过滤短词

        输入：原始文本字符串
        输出：处理后的词列表

        处理步骤：
        1. 转小写（统一大小写）
        2. 去除标点符号（只保留字母和空格）
        3. 分词（按空格分割）
        4. 过滤空词

        知识点：
        - 为什么要转小写？避免 "The" 和 "the" 被视为不同的词
        - 为什么要去除标点？标点对语义学习没有帮助，反而增加噪音
        - 正则表达式：r'[^a-z\s]' 表示"不是字母和空格的字符"
        """
        # 第一步：转小写
        # 示例："The Quick Brown Fox" -> "the quick brown fox"
        text = text.lower()

        # 第二步：去除标点和数字
        # re.sub() 是替换函数，将匹配到的字符替换为空字符串
        # r'[^a-z\s]' 匹配规则：不是a-z字母和空白字符的所有字符
        # 示例："hello, world!" -> "hello world"
        text = re.sub(r'[^a-z\s]', '', text)

        # 第三步：按空格分割成词列表
        # split() 默认按空白字符（空格、tab、换行）分割
        # 示例："hello world" -> ["hello", "world"]
        words = text.split()

        # 第四步：过滤空词
        # 列表推导式，过滤长度为0的词（可能是连续空格导致的）
        words = [w for w in words if len(w) > 0]

        # 返回处理后的词列表
        return words


    def build_vocab(self):
        """
        从语料库构建词汇表

        流程：
        1. 读取所有原始文本文件
        2. 预处理（分词）
        3. 统计词频
        4. 过滤低频词
        5. 建立词<->索引映射

        知识点：
        - 词汇表（Vocabulary）：把所有不重复的词收集起来，编号
        - 词频统计：计算每个词在语料库中出现的次数
        - 低频词过滤：减少词汇表大小，降低模型复杂度，过滤噪音
        - 为什么要用索引？神经网络只能处理数字，需要把词转换为数字
        - 多文件合并：所有文件的词会合并在一起进行词频统计
        """
        print("Loading corpus...")

        # 1. 读取所有文本文件并合并
        # 遍历self.paths列表，读取每个文件的内容
        all_words = []  # 存储所有文件的词

        for file_path in self.paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # 预处理当前文件
            words = self.preprocess(text)
            all_words.extend(words)

            # 打印当前文件的词数
            print(f"  {os.path.basename(file_path)}: {len(words)} words")

        # 2. 打印总词数
        print(f"Total words from all files: {len(all_words)}")

        # 3. 统计词频
        # Counter 是 collections 模块提供的计数器类
        # 会自动统计列表中每个元素出现的次数
        # 示例：["the", "cat", "the"] -> Counter({"the": 2, "cat": 1})
        word_counts = Counter(all_words)

        # 4. 打印不重复词的数量
        print(f"Unique words before filtering: {len(word_counts)}")

        # 5. 过滤低频词
        # 字典推导式，只保留词频 >= min_count 的词
        # 低于阈值的词被过滤掉，不进入词汇表
        self.word_counts = {word: count for word, count in word_counts.items()
                          if count >= self.min_count}

        # 6. 打印过滤后的词汇表大小
        print(f"Vocabulary size after filtering (min_count={self.min_count}): {len(self.word_counts)}")

        # 7. 构建词到索引的映射
        # enumerate() 生成索引-词对：(0, "word1"), (1, "word2"), ...
        # 字典推导式：{词: 索引, ...}
        self.word2idx = {word: idx for idx, word in enumerate(self.word_counts.keys())}

        # 8. 构建索引到词的映射（反向映射）
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # 9. 记录词汇表大小
        self.vocab_size = len(self.word2idx)

        # 返回映射表
        return self.word2idx, self.idx2word


    def generate_training_data(self):
        """
        生成训练数据 (context, target) 对

        CBOW模型的任务：
        - 输入：上下文词的索引
        - 输出：目标词的索引

        生成逻辑：
        对于语料库中的每个词，以它为中心词，
        提取它前后 window_size 范围内的词作为上下文

        示例（window_size=2）：
        原始句子: "the quick brown fox jumps"
                         ^
                       中心词

        中心词 "brown" 的：
        - 上下文词: ["the", "quick", "fox", "jumps"]
        - 目标词: "brown"

        知识点：
        - 滑动窗口：遍历整个句子，每次移动一个位置
        - 上下文：中心词周围的词，用于预测中心词
        - 训练样本：(context_words, target_word) 对
        - 多文件处理：所有文件的词会被串接在一起形成连续的文本序列
        """
        print("Generating training data...")

        # 1. 读取并预处理所有文本文件
        # 将所有文件的词串接在一起
        all_words = []

        for file_path in self.paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            words = self.preprocess(text)
            all_words.extend(words)

        # 打印处理后的总词数
        print(f"Total words for training: {len(all_words)}")

        # 2. 将词转换为索引
        # 过滤掉不在词汇表中的词（如被过滤的低频词）
        # 如果词在 word2idx 中，返回其索引；否则跳过
        # 示例：["the", "cat", "sat"] -> [0, 152, 89]
        word_ids = [self.word2idx[w] for w in words if w in self.word2idx]

        # 初始化存储列表
        context_ids = []   # 存储上下文词的索引
        target_ids = []    # 存储目标词的索引

        # 3. 生成(context, target)对
        # 遍历范围：window_size 到 len(word_ids) - window_size
        # 为什么要从 window_size 开始？
        #   因为前面 window_size 个词没有足够的左侧上下文
        # 为什么要到 len(word_ids) - window_size？
        #   因为后面 window_size 个词没有足够的右侧上下文

        for i in range(self.window_size, len(word_ids) - self.window_size):
            # i 是当前中心词的索引位置

            # 目标词：中心词
            target = word_ids[i]

            # 获取上下文词
            # 初始化空列表
            context = []

            # j 表示相对于中心词的偏移量
            # 范围：[-window_size, window_size]，但不包括0（中心词本身）
            # window_size=2 时：j = -2, -1, 1, 2
            for j in range(-self.window_size, self.window_size + 1):
                if j != 0:  # 跳过中心词本身
                    # i + j 是上下文词的索引位置
                    context.append(word_ids[i + j])

            # 保存这一对训练数据
            context_ids.append(context)   # 上下文词索引列表
            target_ids.append(target)     # 目标词索引

        # 4. 打印训练样本数量
        print(f"Training samples: {len(context_ids)}")

        # 5. 保存训练数据
        self.train_data = (context_ids, target_ids)

        return context_ids, target_ids


    def save(self, path):
        """
        保存词汇表和训练数据到文件

        为什么要保存？
        - 避免每次训练前都重新处理数据
        - 方便后续加载使用

        知识点：
        - pickle：Python对象序列化库
        - 可以把Python对象保存为二进制文件
        - 类似于JSON，但支持更多Python对象类型
        """
        # 打开文件（二进制写入模式）
        with open(path, 'wb') as f:
            # pickle.dump() 序列化对象并写入文件
            # 保存的内容：
            # - paths：数据文件路径列表
            # - word2idx：词到索引映射
            # - idx2word：索引到词映射
            # - word_counts：词频统计
            # - vocab_size：词汇表大小
            # - train_data：训练数据
            pickle.dump({
                'paths': self.paths,
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_counts': self.word_counts,
                'vocab_size': self.vocab_size,
                'train_data': self.train_data
            }, f)

        print(f"Corpus saved to {path}")


    def load(self, path):
        """
        从文件加载词汇表和训练数据

        流程：
        1. 打开pickle文件
        2. 反序列化对象
        3. 恢复到各个属性

        注意：
        - 为了向后兼容，旧版本的corpus文件（没有paths字段）也能正常加载
        - 如果没有paths字段，会使用传入的path参数作为单一数据源
        """
        # 打开文件（二进制读取模式）
        with open(path, 'rb') as f:
            # pickle.load() 从文件读取并反序列化为Python对象
            data = pickle.load(f)

        # 恢复到各个属性
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.word_counts = data['word_counts']
        self.vocab_size = data['vocab_size']
        self.train_data = data['train_data']

        # 恢复paths属性（向后兼容：旧版本corpus文件没有此字段）
        self.paths = data.get('paths', [])

        # 打印加载信息
        print(f"Corpus loaded from {path}")
        print(f"Vocabulary size: {self.vocab_size}")

        return self.word2idx, self.idx2word


# ==================== 辅助函数 ====================

def get_unigram_distribution(word_counts, vocab_size, power=0.75):
    """
    获取用于负采样的unigram分布

    什么是unigram分布？
    - 基于词频的概率分布
    - 词频越高，被采样的概率越大

    什么是负采样（Negative Sampling）？
    - 训练时不仅使用正样本，还随机采样一些"负样本"
    - 正样本：正确的(context, target)对
    - 负采样：随机采样的错误配对
    - 作用：加速训练，避免计算整个词汇表的softmax

    参数 power 的作用：
    - 对词频进行幂运算（通常用0.75）
    - 效果：降低高频词的采样概率，提升低频词的采样概率
    - 这是一种"平滑"策略，让词汇表中的词被采样更均匀

    知识点：
    - 负采样是Word2Vec的核心技巧之一
    - 原始论文使用了0.75次幂，效果很好
    - 这个分布用于在训练时随机选择负样本词

    示例：
    假设词频 = [1000, 100, 10]，power=0.75
    - 变换后：1000^0.75 ≈ 178, 100^0.75 ≈ 56, 10^0.75 ≈ 18
    - 归一化后作为概率分布
    """
    # 1. 将词频字典转换为列表
    # 按索引顺序排列，缺失的词频设为0
    word_counts_list = []
    for i in range(vocab_size):
        # word_counts 的键是词（字符串），这里需要按索引查找
        # 如果索引对应词不存在，返回0
        word_counts_list.append(word_counts.get(i, 0))

    # 2. 转换为numpy数组（float32类型，节省内存）
    word_counts_array = np.array(word_counts_list, dtype=np.float32)

    # 3. 应用幂次变换（对每个词频进行 power 次幂运算）
    # 这是Word2Vec论文中的技巧，使分布更平滑
    word_counts_array = np.power(word_counts_array, power)

    # 4. 归一化：使所有值相加为1，变成合法的概率分布
    word_counts_array = word_counts_array / word_counts_array.sum()

    # 返回概率分布数组
    # 长度 = vocab_size，每个元素是对应索引词的采样概率
    return word_counts_array


# ==================== 主程序入口 ====================

if __name__ == '__main__':
    # 这段代码只在直接运行 data_loader.py 时执行
    # 当作为模块导入时不会执行

    # 示例：构建和处理语料库
    # 支持三种输入形式：
    # 1. 目录路径（推荐）：会读取目录下所有.txt文件
    #    corpus = Corpus('data/', min_count=5, window_size=5)
    # 2. 逗号分隔的多个文件：
    #    corpus = Corpus('data/ptb.train.txt,data/ptb.valid.txt', min_count=5, window_size=5)
    # 3. 单个文件（兼容旧版）：
    #    corpus = Corpus('data/ptb.txt', min_count=5, window_size=5)

    # 1. 创建 Corpus 对象（使用目录，会自动读取所有.txt文件）
    corpus = Corpus('data/', min_count=5, window_size=5)

    # 2. 构建词汇表
    corpus.build_vocab()

    # 3. 生成训练数据
    corpus.generate_training_data()

    # 4. 保存到文件
    corpus.save('data/corpus.pkl')

    """
    运行方式：
    cd src
    python data_loader.py

    这会生成 data/corpus.pkl 文件，包含处理好的词汇表和训练数据

    数据文件支持：
    - 默认读取 data/ 目录下的所有 .txt 文件
    - 也可以手动指定文件列表，用逗号分隔
    """
