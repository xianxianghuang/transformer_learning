# CBOW词向量模型

本项目实现了经典的CBOW (Continuous Bag of Words) 模型，使用PyTorch从Penn Treebank语料库学习词向量表示。

## 项目简介

CBOW模型通过上下文词预测目标词，学习到词的分布式表示（词向量）。这些词向量能够捕获词的语义信息，使得语义相似的词在向量空间中也相近。

## 环境要求

- Python 3.8+
- PyTorch 1.9+
- NumPy
- scikit-learn
- matplotlib

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 下载数据

本项目使用Penn Treebank数据集。你可以从以下地址下载：

https://raw.githubusercontent.com/woojinseo/cbow-word2vec/master/ptb.train.txt

将下载的文件保存为 `data/ptb.txt`。

或者运行以下命令自动下载：

```bash
python -c "
import urllib.request
url = 'https://raw.githubusercontent.com/woojinseo/cbow-word2vec/master/ptb.train.txt'
urllib.request.urlretrieve(url, 'data/ptb.txt')
print('Downloaded ptb.txt')
"
```

### 2. 训练模型

```bash
cd src
python train.py
```

训练完成后，模型将保存在 `checkpoints/` 目录下。

### 3. 评估模型

```bash
cd src
python evaluate.py
```

## 训练参数

可以通过命令行参数自定义训练：

```bash
python train.py \
    --embedding_dim 300 \
    --window_size 5 \
    --learning_rate 0.025 \
    --epochs 20 \
    --batch_size 512
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--embedding_dim` | 300 | 词向量维度 |
| `--window_size` | 5 | 上下文窗口大小 |
| `--min_count` | 5 | 最小词频 |
| `--learning_rate` | 0.025 | 学习率 |
| `--epochs` | 20 | 训练轮数 |
| `--batch_size` | 512 | 批次大小 |
| `--negative_samples` | 5 | 负采样数量 |

## 评估功能

### 词相似度

找出与给定词最相似的词：

```bash
python evaluate.py --model_path checkpoints/cbow_final.pt
```

### 词类比

执行词类比任务，如 "man : king :: woman : ?"

评估脚本会自动运行多个类比测试：
- 语义类比：man->king, woman->queen
- 语法类比：small->smaller, big->?

### 可视化

生成词向量的t-SNE可视化：

```bash
python evaluate.py --visualize --visualize_words king,queen,man,woman,computer,money
```

## 模型架构

```
Input: 上下文词索引 (batch_size, window_size*2)
       ↓
Embedding: 词嵌入 (batch_size, window_size*2, embedding_dim)
       ↓
Mean Pooling: 上下文平均向量 (batch_size, embedding_dim)
       ↓
Output: 负采样预测
```

## 训练效果

训练完成后，你应该能够看到：

1. **损失下降**: 损失从高位持续下降，表明模型在学习
2. **词相似度**: 语义相似的词（如 "man" 和 "woman"）具有高相似度
3. **词类比**: 能够正确完成类比推理

示例输出：

```
Most similar words to 'man':
  woman: 0.8234
  person: 0.7562
  child: 0.7231

man : king :: woman : ?
Predicted: queen (expected: queen)
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `src/data_loader.py` | 数据加载与预处理 |
| `src/model.py` | CBOW模型定义 |
| `src/train.py` | 训练脚本 |
| `src/evaluate.py` | 评估脚本 |
| `src/utils.py` | 工具函数 |
| `data/ptb.txt` | 训练数据 |
| `checkpoints/` | 模型保存目录 |

## 参考

- Mikolov et al. "Efficient Estimation of Word Representations in Vector Space" (2013)
- Mikolov et al. "Distributed Representations of Words and Phrases and their Compositionality" (2013)
