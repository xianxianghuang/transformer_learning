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
    --learning_rate 10.0 \
    --epochs 100 \
    --batch_size 512
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--embedding_dim` | 300 | 词向量维度 |
| `--window_size` | 5 | 上下文窗口大小 |
| `--min_count` | 5 | 最小词频 |
| `--learning_rate` | **10.0** | 学习率 (重要: 使用较大值确保训练效果) |
| `--epochs` | **100** | 训练轮数 (推荐100以获得更好效果) |
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

1. **损失下降**: 损失从约3.8持续下降到2.35左右，表明模型在学习
2. **领域语义关系**: 对于金融新闻语料，模型能学到金融领域的语义关系
3. **词类比**: 在特定领域词汇上可能表现较好

### 验证方法

运行评估脚本查看训练效果：

```bash
cd src
python evaluate.py --model_path ../checkpoints/cbow_final.pt
```

### 实际效果示例（Penn Treebank 金融新闻语料）

**损失变化**:
- Epoch 1: ~3.78
- Epoch 50: ~2.37
- Epoch 100: ~2.35

**金融领域词相似度** (验证模型学到了有意义的语义):

```
million  → billion(0.70), cents(0.69), revenue(0.68)
billion  → rose(0.73), yen(0.72), revenue(0.71), million(0.70)
bank     → bancorp(0.39), trillion(0.36)
stock    → trading(0.57), exchange(0.54)
president→ chairman(0.60), executive(0.55), chief(0.53)
shares   → revenue(0.63), rose(0.62), million(0.61)
```

**说明**: Penn Treebank 是金融新闻语料，因此模型主要学到金融领域的语义关系。例如：
- "million" 和 "billion" 高度相关（数值单位）
- "president" 和 "chairman" 高度相关（公司职位）
- "stock" 和 "trading/exchange" 紧密相关

传统类比任务（king:queen, man:woman）在金融语料上效果较差，因为这些词汇在金融新闻中出现频率较低或语义关系不明显。

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