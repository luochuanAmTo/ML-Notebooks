#### **embedding**

对于要处理的一串文本，我们要让其能够被计算机处理，需要将其转变为词向量，方法有最简单的one-hot，或者有名的Word2Vec等，甚至可以随机初始化词向量, 经过Embedding后，文本中的每一个字就被转变为一个向量


#### **Positional Encoding**

我们对于RNN在处理文本时，由于天然的顺序输入，顺序处理，当前输出要等上一步输出处理完后才能进行，因此不会造成文本的字词在顺序上或者先后关系出现问题。但对于Transformer来说，由于其在处理时是并行执行，虽然加快了速度，但是忽略了词字之间的前后关系或者先后顺序。同时Transformer基于Self-Attention机制，而self-attention不能获取字词的位置信息，即使打乱一句话中字词的位置，每个词还是能与其他词之间计算出attention值，因此我们需要为每一个词向量添加位置编码。

 得到512维的位置编码后，我们将**512维的位** **置编码与512维的词向量****相加**，得到 **最终的512维词向量** 作为最终的Transformer输入。

![img](https://i-blog.csdnimg.cn/blog_migrate/313e703f63e3094dbd2598eb402e3457.png)



#### **Encoder**

​     放大一个encoder，发现里边的结构是一个自注意力机制加上一个前馈神经网络。

![img](https://i-blog.csdnimg.cn/blog_migrate/acd42debbe17a13d71a8165ff2f673c5.jpeg)

### **self-attention**
​        首先，self-attention的输入就是词向量，即整个模型的最初的输入是词向量的形式。那自注意力机制呢，顾名思义就是自己和自己计算一遍注意力，即对每一个输入的词向量，我们需要构建self-attention的输入。在这里，transformer首先将词向量乘上三个矩阵，得到三个新的向量，之所以乘上三个矩阵参数而不是直接用原本的词向量是因为这样增加更多的参数，提高模型效果。对于输入X1(Thinking)，乘上三个矩阵后分别得到Q1,K1,V1，同样的，对于输入X2(Machines)，也乘上三个不同的矩阵得到Q2,K2,V2。

![img](https://pic2.zhimg.com/v2-b0a11f97ab22f5d9ebc396bc50fa9c3f_1440w.jpg)



在实际中，Self-Attention 接收的是输入(单词的表示向量x组成的矩阵X) 或者上一个 Encoder block 的输出。而**Q,K,V**正是通过 Self-Attention 的输入进行线性变换得到的。Self-Attention 的输入用矩阵X进行表示，则可以使用线性变阵矩阵**WQ,WK,WV**计算得到**Q,K,V**。计算如下图所示，**注意 X, Q, K, V 的每一行都表示一个单词。**

![img](https://pic1.zhimg.com/v2-4f4958704952dcf2c4b652a1cd38f32e_1440w.jpg)

公式中计算矩阵**Q**和**K**每一行向量的内积，为了防止内积过大，因此除以 dk 的平方根。**Q**乘以**K**的转置后，得到的矩阵行列数都为 n，n 为句子单词数，这个矩阵可以表示单词之间的 attention 强度。下图为**Q**乘以 KT ，1234 表示的是句子中的单词。





![img](https://pic4.zhimg.com/v2-9caab2c9a00f6872854fb89278f13ee1_1440w.jpg)

得到Q$K^T$ 之后，使用 Softmax 计算每一个单词对于其他单词的 attention 系数，公式中的 Softmax 是对矩阵的每一行进行 Softmax，即每一行的和都变为 1.

![img](https://pic3.zhimg.com/v2-96a3716cf7f112f7beabafb59e84f418_1440w.jpg)

上图中 Softmax 矩阵的第 1 行表示单词 1 与其他所有单词的 attention 系数，最终单词 1 的输出 Z1 等于所有单词 i 的值 Vi 根据 attention 系数的比例加在一起得到，如下图所示：

![img](https://pic1.zhimg.com/v2-27822b2292cd6c38357803093bea5d0e_1440w.jpg)



![img](https://picx.zhimg.com/v2-b0ea8f5b639786f98330f70405e94a75_1440w.jpg)

Multi-Head Attention 包含多个 Self-Attention 层，首先将输入**X**分别传递到 h 个不同的 Self-Attention 中，计算得到 h 个输出矩阵**Z**。下图是 h=8 时候的情况，此时会得到 8 个输出矩阵**Z**。

得到 8 个输出矩阵 Z1 到 Z8 之后，Multi-Head Attention 将它们拼接在一起 **(Concat)**，然后传入一个**Linear**层，得到 Multi-Head Attention 最终的输出**Z**。

![img](https://picx.zhimg.com/v2-35d78d9aa9150ae4babd0ea6aa68d113_1440w.jpg)

![img](https://picx.zhimg.com/v2-0203e83066913b53ec6f5482be092aa1_1440w.jpg)![img](https://pic1.zhimg.com/v2-a4b35db50f882522ee52f61ddd411a5a_1440w.jpg)

其中 **X**表示 Multi-Head Attention 或者 Feed Forward 的输入，MultiHeadAttention(**X**) 和 FeedForward(**X**) 表示输出 (输出与输入 **X** 维度是一样的，所以可以相加)。

![img](https://pic4.zhimg.com/v2-4b3dde965124bd00f9893b05ebcaad0f_1440w.jpg)



Feed Forward 层比较简单，是一个两层的全连接层，第一层的激活函数为 Relu，第二层不使用激活函数，对应的公式如下

![img](https://pic4.zhimg.com/v2-47b39ca4cc3cd0be157d6803c8c8e0a1_1440w.jpg)。





第一个 Encoder block 的输入为句子单词的表示向量矩阵，后续 Encoder block 的输入是前一个 Encoder block 的输出，最后一个 Encoder block 输出的矩阵就是**编码信息矩阵 C**，这一矩阵后续会用到 Decoder 中。





![img](https://pic1.zhimg.com/v2-45db05405cb96248aff98ee07a565baa_r.jpg)



## Decoder 结构

![img](https://pic3.zhimg.com/v2-f5049e8711c3abe8f8938ced9e7fc3da_1440w.jpg)





Decoder block 的第一个 Multi-Head Attention 采用了 Masked 操作，因为在翻译的过程中是顺序翻译的，即翻译完第 i 个单词，才可以翻译第 i+1 个单词。通过 Masked 操作可以防止第 i 个单词知道 i+1 个单词之后的信息。下面以 "我有一只猫" 翻译成 "I have a cat" 为例，了解一下 Masked 操作。

下面的描述中使用了类似 [Teacher Forcing](https://zhida.zhihu.com/search?content_id=163422979&content_type=Article&match_order=1&q=Teacher+Forcing&zhida_source=entity) 的概念，不熟悉 Teacher Forcing 的童鞋可以参考以下上一篇文章Seq2Seq 模型详解。在 Decoder 的时候，是需要根据之前的翻译，求解当前最有可能的翻译，如下图所示。首先根据输入 "<Begin>" 预测出第一个单词为 "I"，然后根据输入 "<Begin> I" 预测下一个单词 "have"。

Decoder 可以在训练的过程中使用 Teacher Forcing 并且并行化训练，即将正确的单词序列 (<Begin> I have a cat) 和对应输出 (I have a cat <end>) 传递到 Decoder。那么在预测第 i 个输出时，就要将第 i+1 之后的单词掩盖住，**注意 Mask 操作是在 Self-Attention 的 Softmax 之前使用的，下面用 0 1 2 3 4 5 分别表示 "<Begin> I have a cat <end>"。**

**第一步：**是 Decoder 的输入矩阵和 **Mask** 矩阵，输入矩阵包含 "<Begin> I have a cat" (0, 1, 2, 3, 4) 五个单词的表示向量，**Mask** 是一个 5×5 的矩阵。在 **Mask** 可以发现单词 0 只能使用单词 0 的信息，而单词 1 可以使用单词 0, 1 的信息，即只能使用之前的信息。

![img](https://pica.zhimg.com/v2-b26299d383aee0dd42b163e8bda74fc8_1440w.jpg)

**第二步：**接下来的操作和之前的 Self-Attention 一样，通过输入矩阵**X**计算得到**Q,K,V**矩阵。然后计算**Q**和 KT 的乘积 QKT 。

![img](https://pic2.zhimg.com/v2-a63ff9b965595438ed0c0e0547cd3d3b_1440w.jpg)



**第三步：**在得到 QKT 之后需要进行 Softmax，计算 attention score，我们在 Softmax 之前需要使用**Mask**矩阵遮挡住每一个单词之后的信息，遮挡操作如下：



![img](https://picx.zhimg.com/v2-35d1c8eae955f6f4b6b3605f7ef00ee1_1440w.jpg)



使用 **Mask** QKT与矩阵 **V**相乘，得到输出 **Z**，则单词 1 的输出向量 Z1 是只包含单词 1 信息的。



![img](https://picx.zhimg.com/v2-58f916c806a6981e296a7a699151af87_1440w.jpg)

通过上述步骤就可以得到一个 Mask Self-Attention 的输出矩阵 Zi ，然后和 Encoder 类似，通过 Multi-Head Attention 拼接多个输出Zi 然后计算得到第一个 Multi-Head Attention 的输出**Z**，**Z**与输入**X**维度一样。