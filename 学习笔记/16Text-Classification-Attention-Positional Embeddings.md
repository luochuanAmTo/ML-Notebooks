```py
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
```





```py
dataset = fetch_20newsgroups(subset='all')

X = pd.Series(dataset['data'])  #原始文本数据,X是一个Series，其中每一行是一个新闻组的文本内容
y = pd.Series(dataset['target']) #原始标签
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, stratify=y, random_state=19)  #将数据划分为训练集和验证集。
y_train = pd.get_dummies(y_train)
y_valid = pd.get_dummies(y_valid) #将训练集的标签（y_train）和验证集进行独热编码（one-hot encoding）。

```

### **`dataset` 的主要结构**

| 字段名 (Key)   | 数据类型        | 描述                                                         |
| :------------- | :-------------- | :----------------------------------------------------------- |
| `data`         | `list` of `str` | 所有新闻文本的列表，每个元素是一篇新闻的原始文本（字符串）。 |
| `target`       | `numpy.ndarray` | 每篇新闻对应的类别标签（数字形式），范围是 `0` 到 `19`（共20个类别）。 |
| `target_names` | `list` of `str` | 类别名称列表，按数字标签顺序排列（如 `target=0` 对应 `target_names[0]`）。 |
| `filenames`    | `numpy.ndarray` | 每个新闻文件的原始路径（如果从磁盘加载时有用，否则可能为虚拟路径）。 |
| `DESCR`        | `str`           | 数据集的描述信息（英文）。                                   |



```py
class TransformerBlock(L.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim #输入向量的维度
        self.dense_dim = dense_dim #前馈神经网络（Feed Forward Network）的隐藏层维度
        self.num_heads = num_heads  #多头注意力机制的头数
        #定义子层
        self.attention = L.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim) #多头注意力层（MultiHeadAttention），用于计算输入序列的自注意力
        self.dense_proj = keras.Sequential([L.Dense(dense_dim, activation='relu'), L.Dense(embed_dim)])#前馈神经网络（FFN），是一个两层的全连接网络 第一层：维度为 dense_dim，使用 ReLU 激活。 第二层：维度为 embed_dim（与输入一致），无激活函数。
        self.layernorm1 = L.LayerNormalization() #对 自注意力子层输出 的残差连接结果归一化。
        self.layernorm2 = L.LayerNormalization() #对 FFN 子层输出 的残差连接结果归一化。
    
    def call(self, inputs, mask=None): #inputs：输入张量，形状为 (batch_size, sequence_length, embed_dim)。 mask：可选掩码张量，用于屏蔽无效位置（如填充位置）
        if mask is not None:
            mask = mask[: tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        #inputs (第一个参数)	作为 query（查询向量），决定要关注哪些位置。
        #inputs (第二个参数)	作为 key 和 value（键和值向量），提供被关注的内容
        #attention_mask=mask	可选掩码，屏蔽无效位置（如填充符 [PAD]）
        proj_input = self.layernorm1(inputs + attention_output) #残差连接：将原始输入 inputs 与注意力输出相加（inputs + attention_output  层归一化：对结果应用 LayerNormalization（沿 embed_dim 归一化）。
        proj_output = self.dense_proj(proj_input) #通过两层全连接网络（FFN）变换特征。
        return self.layernorm2(proj_input + proj_output)#将 FFN 的输入 proj_input 与输出 proj_output 相加（残差连接）。再次应用层归一化。
    
    def get_config(self):
        config = super().get_confog()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim
        })
        return config
```



以![\mathbf{X}\mathbf{X}^\top](https://www.zhihu.com/equation?tex=%5Cmathbf%7BX%7D%5Cmathbf%7BX%7D%5E%5Ctop&consumer=ZHI_MENG)中的第一行第一列元素为例，其实是向量![\mathbf{x}_{0}](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D_%7B0%7D&consumer=ZHI_MENG)与![\mathbf{x}_{0}](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D_%7B0%7D&consumer=ZHI_MENG)自身做点乘，其实就是![\mathbf{x}_{0}](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D_%7B0%7D&consumer=ZHI_MENG)自身与自身的相似度，那第一行第二列元素就是![\mathbf{x}_{0}](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D_%7B0%7D&consumer=ZHI_MENG)与![\mathbf{x}_{1}](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D_%7B1%7D&consumer=ZHI_MENG)之间的相似度。

下面以词向量矩阵为例，这个矩阵中，每行为一个词的词向量。矩阵与自身的转置相乘，生成了目标矩阵，目标矩阵其实就是一个词的词向量与各个词的词向量的相似度。如果再加上Softmax呢？我们进行下面的计算：![Softmax(\mathbf{X}\mathbf{X}^\top)](https://www.zhihu.com/equation?tex=Softmax%28%5Cmathbf%7BX%7D%5Cmathbf%7BX%7D%5E%5Ctop%29&consumer=ZHI_MENG)。Softmax的作用是对向量做归一化，那么就是对相似度的归一化，得到了一个归一化之后的权重矩阵，矩阵中，某个值的权重越大，表示相似度越高

![img](https://pica.zhimg.com/v2-deb3deed7c3bad7ee10b15d7e8cbec96_1440w.webp?consumer=ZHI_MENG)

在这个基础上，再进一步：![Softmax(\mathbf{X}\mathbf{X}^\top)\mathbf{X}](https://www.zhihu.com/equation?tex=Softmax%28%5Cmathbf%7BX%7D%5Cmathbf%7BX%7D%5E%5Ctop%29%5Cmathbf%7BX%7D&consumer=ZHI_MENG)，将得到的归一化的权重矩阵与词向量矩阵相乘。权重矩阵中某一行分别与词向量的一列相乘，词向量矩阵的一列其实代表着不同词的某一维度。经过这样一个矩阵相乘，相当于一个加权求和的过程，得到结果词向量是经过加权求和之后的新表示，而权重矩阵是经过相似度和归一化计算得到的。

![img](https://pic3.zhimg.com/v2-cf1b40bc7b53359d8b540b378e24ef98_1440w.webp?consumer=ZHI_MENG)

Q为Query、K为Key、V为Value。Q、K、V是从哪儿来的呢？Q、K、V其实都是从同样的输入矩阵X线性变换而来的。我们可以简单理解成：

![Q = XW^Q \\ K = XW^K \\ V = XW^V \\](https://www.zhihu.com/equation?tex=Q+%3D+XW%5EQ+%5C%5C+K+%3D+XW%5EK+%5C%5C+V+%3D+XW%5EV+%5C%5C&consumer=ZHI_MENG)

为了增强拟合性能，Transformer对Attention继续扩展，提出了多头注意力（Multiple Head Attention）。刚才我们已经理解了，![Q](https://www.zhihu.com/equation?tex=Q&consumer=ZHI_MENG)、![K](https://www.zhihu.com/equation?tex=K&consumer=ZHI_MENG)、![V](https://www.zhihu.com/equation?tex=V&consumer=ZHI_MENG)是输入![X](https://www.zhihu.com/equation?tex=X&consumer=ZHI_MENG)与![W^Q](https://www.zhihu.com/equation?tex=W%5EQ&consumer=ZHI_MENG)、![W^K](https://www.zhihu.com/equation?tex=W%5EK&consumer=ZHI_MENG)和![W^V](https://www.zhihu.com/equation?tex=W%5EV&consumer=ZHI_MENG)分别相乘得到的，![W^Q](https://www.zhihu.com/equation?tex=W%5EQ&consumer=ZHI_MENG)、![W^K](https://www.zhihu.com/equation?tex=W%5EK&consumer=ZHI_MENG)和![W^V](https://www.zhihu.com/equation?tex=W%5EV&consumer=ZHI_MENG)是可训练的参数矩阵。现在，对于同样的输入![X](https://www.zhihu.com/equation?tex=X&consumer=ZHI_MENG)，我们定义多组不同的![W^Q](https://www.zhihu.com/equation?tex=W%5EQ&consumer=ZHI_MENG)、![W^K](https://www.zhihu.com/equation?tex=W%5EK&consumer=ZHI_MENG)、![W^V](https://www.zhihu.com/equation?tex=W%5EV&consumer=ZHI_MENG)，比如![W^Q_0](https://www.zhihu.com/equation?tex=W%5EQ_0&consumer=ZHI_MENG)、![W^K_0](https://www.zhihu.com/equation?tex=W%5EK_0&consumer=ZHI_MENG)、![W^V_0](https://www.zhihu.com/equation?tex=W%5EV_0&consumer=ZHI_MENG)，![W^Q_1](https://www.zhihu.com/equation?tex=W%5EQ_1&consumer=ZHI_MENG)、![W^K_1](https://www.zhihu.com/equation?tex=W%5EK_1&consumer=ZHI_MENG)和![W^V_1](https://www.zhihu.com/equation?tex=W%5EV_1&consumer=ZHI_MENG)，每组分别计算生成不同的![Q](https://www.zhihu.com/equation?tex=Q&consumer=ZHI_MENG)、![K](https://www.zhihu.com/equation?tex=K&consumer=ZHI_MENG)、![V](https://www.zhihu.com/equation?tex=V&consumer=ZHI_MENG)，最后学习到不同的参数。



```py
class PositionalEmbedding(L.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        #sequence_length：输入序列的最大长度 input_dim：词汇表大小  output_dim：嵌入向量的维度 **kwargs：其他传递给父类的参数（
        super().__init__(**kwargs)
        self.token_embeddings = L.Embedding(input_dim, output_dim)
        self.position_embeddings = L.Embedding(sequence_length, output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def call(self, inputs): #（前向计算）inputs：输入张量，形状为 (batch_size, sequence_length)，包含词汇索引
        length = tf.shape(inputs)[-1] #获取当前输入序列的实际长度 length（动态支持变长序列）。
        positions = tf.range(start=0, limit=length, delta=1)
        #生成位置索引序列 [0, 1, 2, ..., length-1]，形状为 (length,)。length=3，则 positions = [0, 1, 2]。
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config
```

  位置嵌入通过给每个词赋予一个与它在句子中位置相关的独特向量，使得模型不仅能够捕捉到词语的语义，还能理解它们之间的相对顺序，从而更好地建模句子结构和依赖关系。

为了让模型能够学习到位置信息，最直接的方法是为每个位置分配一个固定的、预定义的向量。在原始的 Transformer 模型中，位置嵌入是由**正弦**和**余弦**函数组成的，这样设计的原因在于它具有周期性，可以帮助模型处理比训练时更长的序列，同时保持一定的泛化能力。



步骤一：词嵌入
首先，我们需要将句子中的每个词转换为词嵌入，假设我们得到了如下简化版的词嵌入向量（实际预训练模型的维度远高于此）：

W{The} = [0.1, 0.2]
W{cat} = [0.3, 0.4]
W{sat} = [0.5, 0.6]
W{on}  = [0.7, 0.8]
W{the} = [0.9, 1.0]
W{mat} = [1.1, 1.2]

步骤二：位置嵌入
接下来，我们需要为每个词添加位置嵌入。我们可以根据上述公式计算出每个位置的嵌入向量。假设我们得到了如下位置嵌入向量（同样简化为2 维）：

P_0 = [0.0, 1.0]
P_1 = [0.8, 0.6]
P_2 = [0.5, 0.8]
P_3 = [0.2, 0.9]
P_4 = [0.9, 0.4]
P_5 = [0.7, 0.2]
步骤三：词嵌入 + 位置嵌入
现在，我们将词嵌入和位置嵌入相加，得到最终的输入向量。这一步操作使得每个词的表示不仅包含了其语义信息，还包含了它在句子中的位置信息。具体来说，我们有：

X{The} = W{The} + P_0 = [0.1, 0.2] + [0.0, 1.0] = [0.1, 1.2]
X{cat} = W{cat} + P_1 = [0.3, 0.4] + [0.8, 0.6] = [1.1, 1.0]
X{sat} = W{sat} + P_2 = [0.5, 0.6] + [0.5, 0.8] = [1.0, 1.4]
X{on}  = W{on}  + P_3 = [0.7, 0.8] + [0.2, 0.9] = [0.9, 1.7]
X{the} = W{the} + P_4 = [0.9, 1.0] + [0.9, 0.4] = [1.8, 1.4]
X{mat} = W{mat} + P_5 = [1.1, 1.2] + [0.7, 0.2] = [1.8, 1.4]



```py
vocab_size = 10_000
embed_dim = 256
num_heads = 2
dense_dim = 32
seq_length = 256

#vocab_size：词汇表大小（保留最常见的10,000个词）。
#embed_dim：词嵌入向量的维度（256维）。
#num_heads：Transformer中多头注意力的头数（2头）。
#dense_dim：Transformer前馈网络的隐藏层维度（32维）。

seq_length：输入序列的最大长度（256个词）。
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<unw>')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_train = sequence.pad_sequences(X_train, maxlen=seq_length)
X_valid = tokenizer.texts_to_sequences(X_valid)
X_valid = sequence.pad_sequences(X_valid, maxlen=seq_length)
#输入 X_train[0] = "hello world" → 输出 [23, 56, 0, 0, ..., 0]（长度256）
inputs = keras.Input(shape=(None, ), dtype="int64")
x = PositionalEmbedding(seq_length, vocab_size, embed_dim)(inputs)
x = TransformerBlock(embed_dim, dense_dim, num_heads)(x)
x = L.GlobalMaxPooling1D()(x)
x = L.Dropout(0.5)(x)
outputs = L.Dense(20, activation='softmax')(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```



处理流程：**1. 位置感知嵌入层 (PositionalEmbedding)**

```
x = PositionalEmbedding(seq_length, vocab_size, embed_dim)(inputs)
```

- **输入**：`inputs` 是形状为 `(batch_size, seq_length)` 的整数序列（词汇索引）。
- **处理过程**：
  1. **词嵌入**：通过 `Embedding(vocab_size, embed_dim)` 将每个词汇索引转换为 `embed_dim` 维的稠密向量。
  2. **位置嵌入**：通过 `Embedding(seq_length, embed_dim)` 为序列中的每个位置（0 到 `seq_length-1`）生成位置向量。
  3. **相加**：将词嵌入和位置嵌入逐元素相加，得到最终的位置感知嵌入。
- **输出形状**：`(batch_size, seq_length, embed_dim)`。
- **作用**：
  - 让模型同时捕获词汇的语义信息和其在序列中的位置信息。
  - 解决了Transformer因无循环结构导致的位置不敏感问题。

------

### **2. Transformer编码块 (TransformerBlock)**

```
x = TransformerBlock(embed_dim, dense_dim, num_heads)(x)
```

- **输入**：来自上一层的 `(batch_size, seq_length, embed_dim)` 张量。
- **处理过程**：
  1. **多头自注意力**：
     - 计算序列内每个位置与其他位置的关联权重（动态关注重要部分）。
     - `num_heads` 允许模型从不同角度学习注意力（如局部和全局关系）。
  2. **残差连接 + 层归一化**：
     - 保留原始输入信息，缓解梯度消失：`LayerNorm(x + Attention(x))`。
  3. **前馈神经网络 (FFN)**：
     - 通过两层全连接层（`Dense(dense_dim)` + `Dense(embed_dim)`）进一步非线性变换。
- **输出形状**：`(batch_size, seq_length, embed_dim)`（与输入相同）。
- **作用**：
  - 捕获序列内部的复杂依赖关系（如长距离依赖）。
  - 是Transformer的核心组件，替代了RNN的循环计算。

------

### **3. 全局最大池化 (GlobalMaxPooling1D)**

```
x = L.GlobalMaxPooling1D()(x)
```

- **输入**：`(batch_size, seq_length, embed_dim)`。
- **处理过程**：
  - 沿序列维度（`seq_length`）取每个特征维度（`embed_dim`）的最大值。
  - 例如，对 `embed_dim=256`，会在256个维度上分别取最大值。
- **输出形状**：`(batch_size, embed_dim)`。
- **作用**：
  - 将变长序列压缩为固定长度的向量表示。
  - 突出序列中最显著的特征（类似NLP中的“关键词”作用）。
  - 替代了Flatten层，减少参数量并保留重要信息。

------

### **4. Dropout正则化**

```
x = L.Dropout(0.5)(x)
```

- **输入**：`(batch_size, embed_dim)`。
- **处理过程**：
  - 在训练阶段，随机将50%的神经元输出置为0（测试阶段不生效）。
- **输出形状**：与输入相同 `(batch_size, embed_dim)`。
- **作用**：
  - 防止模型过拟合训练数据，提升泛化能力。
  - 在稠密层（Dense）前使用，是一种常见的正则化手段。

------

### **数据流示例**

假设 `batch_size=2`, `seq_length=5`, `embed_dim=4`：

1. **输入**：

   ```
inputs = [[1, 3, 2, 0, 0], [4, 1, 0, 0, 0]]  # 形状 (2, 5)
   ```

2. **PositionalEmbedding后**：

   ```
   x = [
       [[0.1, 0.3, -0.2, 0.5], [0.4, -0.1, 0.0, 0.2], ..., [0.0, 0.0, 0.0, 0.0]],  # 序列1
       [[0.7, 0.2, 0.1, -0.3], [0.1, 0.3, -0.2, 0.5], ..., [0.0, 0.0, 0.0, 0.0]]   # 序列2
   ]  # 形状 (2, 5, 4)
   ```
   
3. **TransformerBlock后**：

   - 自注意力重新加权后的序列（形状仍为 `(2, 5, 4)`）。

4. **GlobalMaxPooling1D后**：

   

   ```
   x = [
       [0.4, 0.3, 0.0, 0.5],  # 序列1各维度的最大值
       [0.7, 0.3, 0.1, 0.5]   # 序列2各维度的最大值
   ]  # 形状 (2, 4)
   ```

5. **Dropout后**（训练时）：

   

   ```
   x = [
       [0.4, 0.0, 0.0, 0.5],  # 第2、3维被随机置0
       [0.7, 0.3, 0.0, 0.0]    # 第3、4维被随机置0
   ]
   ```



```py
es = keras.callbacks.EarlyStopping(verbose=1, patience=5, restore_best_weights=True) #atience=5：如果验证集指标（默认监控 val_loss）在 5轮 内没有提升，则停止训练。


rlp = keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1) #patience=3：如果验证集指标（默认 val_loss）在 3轮 内未提升，则将学习率乘以默认因子 0.1。
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
    callbacks=[es, rlp], epochs=100
)
```

