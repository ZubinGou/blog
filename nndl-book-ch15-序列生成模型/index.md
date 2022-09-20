# 《神经网络与深度学习》第15章 - 序列生成模型


类似一般概率模型，序列概率模型的两个基本问题：
1. 概率密度估计
2. 样本生成

## 15.1 序列概率模型
序列数据的概率密度估计可以转换为单变量的条件概率估计问题：
$$
p\left(x\_{t} \mid \boldsymbol{x}\_{1}:(t-1)\right)
$$

给定N个序列的数据集，序列概率模型学习模型$p\_{\theta}\left(x \mid \boldsymbol{x}\_{1:(t-1)}\right)$来最大化整个数据集的对数似然函数：
$$
\max \_{\theta} \sum\_{n=1}^{N} \log p\_{\theta}\left(\boldsymbol{x}\_{1: T\_{n}}^{(n)}\right)=\max \_{\theta} \sum\_{n=1}^{N} \sum\_{t=1}^{T\_{n}} \log p\_{\theta}\left(x\_{t}^{(n)} \mid \boldsymbol{x}\_{1:(t-1)}^{(n)}\right)
$$

这种每次将前一步的输出作为当前输入的方式称为自回归（AutoRegressive），这类模型称为自动回归生成模型（AutoRegressive Generative Model）

主流自回归生成模型：
- N-gram
- 深度序列模型

### 15.1.1 序列生成
搜索：
- greedy
- beam search

## 15.2 N元统计模型
**N 元模型**（N-Gram Model）：假设每个词只依赖于前面N-1个词：
$$
p\left(x\_{t} \mid \boldsymbol{x}\_{1:(t-1)}\right)=p\left(x\_{t} \mid \boldsymbol{x}\_{(t-N+1):(t-1)}\right)
$$

N=1时，一元（Unigram）模型：
$$
p\left(\boldsymbol{x}\_{1: T} ; \theta\right)=\prod\_{t=1}^{T} p\left(x\_{t}\right)=\prod\_{k=1}^{|\mathcal{V}|} \theta\_{k}^{m\_{k}}
$$

可以证明，其最大似然估计等于频率估计

N元时，条件概率也可以通过最大似然函数得到：
$$
p\left(x\_{t} \mid \boldsymbol{x}\_{(t-N+1):(t-1)}\right)=\frac{\mathrm{m}\left(\boldsymbol{x}\_{(t-N+1): t}\right)}{\mathrm{m}\left(\boldsymbol{x}\_{(t-N+1):(t-1)}\right)}
$$

**平滑技术**
N-gram一个主要问题：数据稀疏
- 直接解法：增加数据

> Zipf 定律（Zipf’s Law）：给定自然语言数据集，单词出现频率与其频率排名成反比。
- 平滑技术：对未出现猜测词组赋予一定先验概率

加法平滑：
$$
p\left(x\_{t} \mid \boldsymbol{x}\_{(t-N+1):(t-1)}\right)=\frac{\mathrm{m}\left(\boldsymbol{x}\_{(t-N+1): t}\right)+\delta}{\mathrm{m}\left(\boldsymbol{x}\_{(t-N+1):(t-1)}\right)+\delta|\mathcal{V}|}
$$

其中 $\delta \in (0, 1]$



## 15.3 深度序列模型
深度序列模型（Deep Sequence Model）：用神经网络估计条件概率：
$$
p\_{\theta}\left(x\_{t} \mid \boldsymbol{x}\_{1:(t-1)}\right)=f\_{k\_{x\_{t}}}\left(\boldsymbol{x}\_{1:(t-1)} ; \theta\right)
$$

### 15.3.1 模型结构
- 嵌入层
- 特征层
	- 简单平均
	- FNN/CNN
	- RNN
- 输出层

### 15.3.2 参数学习
给定训练序列，训练目标是找到参数 $\theta$ （embed、weight、bias等）使得对数似然函数最大：
$$
p\_{\theta}\left(x\_{t} \mid \boldsymbol{x}\_{1:(t-1)}\right)=f\_{k\_{x\_{t}}}\left(\boldsymbol{x}\_{1:(t-1)} ; \theta\right)
$$

一般通过梯度上升法学习：
$$
\theta \leftarrow \theta+\alpha \frac{\partial \log p\_{\theta}\left(\boldsymbol{x}\_{1: T}\right)}{\partial \theta}
$$

## 15.4 评价方法
### 15.4.1 困惑度
困惑度（Perplexity）衡量分布的不确定性，随机变量 $X$ 的困惑度：
$$
2^{H(p)}=2^{-\sum\_{x \in x} p(x) \log \_{2} p(x)}
$$
指数为分布p的熵。

困惑度也可以衡量两个分布的差异，对未知数据分布采样，则模型分布的困惑度为：
$$
2^{H\left(\tilde{p}\_{r}, p\_{\theta}\right)}=2^{-\frac{1}{N} \sum\_{n=1}^{N} \log \_{2} p\_{\theta}\left(x^{(n)}\right)}
$$
指数为经验分布和模型分布交叉熵，也是所有样本的负对数似然函数。

困惑度衡量了模型分布和样本经验分布之间的契合程度，困惑度越低两个分布越接近。

对N个独立同分布的序列，测试集的联合概率为：
$$
\prod\_{n=1}^{N} p\_{\theta}\left(\boldsymbol{x}\_{1: T\_{n}}^{(n)}\right)=\prod\_{n=1}^{N} \prod\_{t=1}^{T\_{n}} p\_{\theta}\left(x\_{t}^{(n)} \mid \boldsymbol{x}\_{1:(t-1)}^{(n)}\right)
$$

模型$p_\theta(x)$的困惑度定义为：
$$
\begin{aligned}
\operatorname{PPL}(\theta) &=2^{-\frac{1}{T} \sum\_{n=1}^{N} \log \_{2} p\_{\theta}\left(x\_{1: T\_{n}}^{(n)}\right)} \\\\
&=2^{-\frac{1}{T} \sum\_{n=1}^{N} \sum\_{t=1}^{T n} \log \_{2} p\_{\theta}\left(x\_{t}^{(n)} \mid x\_{1:(t-1)}^{(n)}\right)} \\\\
&=\left(\prod\_{n=1}^{N} \prod\_{t=1}^{T\_{n}} p\_{\theta}\left(x\_{t}^{(n)} \mid x\_{1:(t-1)}^{(n)}\right)\right)^{-1 / T}
\end{aligned}
$$
其中$T$为测试序列总长度。可以看到，困惑度为每个词的条件概率的几何平均数的倒数。

### 15.4.2 BLEU
BLEU（BiLingual Evaluation Understudy）：衡量生成序列与参考序列之间N-Gram重合度。

N 元组合的精度（Precision）：
$$
P\_{N}(\boldsymbol{x})=\frac{\sum\_{w \in \mathcal{W}} \min \left(c\_{w}(\boldsymbol{x}), \max \_{k=1}^{K} c\_{w}\left(\boldsymbol{s}^{(k)}\right)\right)}{\sum\_{w \in \mathcal{W}} c\_{w}(\boldsymbol{x})},
$$
对每个N元组合$w$，累加$w$在K个参考序列中出现的最多次数，除以总N元组合个数，得到生成序列的N元组合在参考序列出现的比例。

由于生成序列越短，精度会越高，引入长度惩罚因子（Brevity Penalty）：
$$
b(\boldsymbol{x})=\left\{\begin{array}{ccc}
1 & \text { if } & l\_{x}>l\_{s} \\\\
\exp \left(1-l\_{s} / l\_{x}\right) & \text { if } & l\_{x} \leq l\_{s}
\end{array}\right.
$$

BLEU是对不同长度的N元组合精度的几何加权平均：
$$
\operatorname{BLEU-N}(\boldsymbol{x})=b(\boldsymbol{x}) \times \exp \left(\sum\_{N=1}^{N^{\prime}} \alpha\_{N} \log P\_{N}\right)
$$

注：BLEU只计算精度，不关心召回率。

### 15.4.3 ROUGE
ROUGE（Recall-Oriented Understudy for Gisting Evaluation）
- 最早应用与文本摘要
- 计算召回率

$$
\operatorname{ROUGE-N}(\boldsymbol{x})=\frac{\sum\_{k=1}^{K} \sum\_{w \in \mathcal{W}} \min \left(c\_{w}(\boldsymbol{x}), c\_{w}\left(\boldsymbol{s}^{(k)}\right)\right)}{\sum\_{k=1}^{K} \sum\_{w \in \mathcal{W}} c\_{w}\left(\boldsymbol{s}^{(k)}\right)},
$$


## 15.5 序列生成模型中的学习问题



## 15.6 序列到序列 
seq2seq：机器翻译、语音识别、文本摘要、对话系统、图像标题生成等

seq2seq模型目标是估计条件概率：
$$
p\_{\theta}\left(\boldsymbol{y}\_{1: T} \mid \boldsymbol{x}\_{1: S}\right)=\prod\_{t=1}^{T} p\_{\theta}\left(y\_{t} \mid \boldsymbol{y}\_{1:(t-1)}, \boldsymbol{x}\_{1: S}\right)
$$

用最大似然估计训练模型参数：
$$
\hat{\theta}=\underset{\theta}{\arg \max } \sum\_{n=1}^{N} \log p\_{\theta}\left(\boldsymbol{y}\_{1: T\_{n}} \mid \boldsymbol{x}\_{1: S\_{n}}\right)
$$

根据输入序列生成最可能目标序列（greedy / beam search）：
$$
\hat{\boldsymbol{y}}=\underset{\boldsymbol{y}}{\arg \max } p\_{\hat{\theta}}(\boldsymbol{y} \mid \boldsymbol{x})
$$

条件概率 $p\_{\theta}\left(y\_{t} \mid \boldsymbol{y}\_{1:(t-1)}, \boldsymbol{x}\_{1: S}\right)$ 可以通过不同神经网络实现，如RNN、注意力模型等。

### 15.6.1 基于RNN的seq2seq
编码器-解码器（Encoder-Decoder）模型

![858dd81164f0f893cc1c0c2335d8ff8e.png](../../_resources/bccef725d11b426baa2019ee26ffa001.png)

缺点：
1. 编码向量信息容量瓶颈
2. 对长序列存在长程依赖问题，容易丢失输入序列的信息

### 15.6.1 基于注意力的seq2seq
解码过程中，将上一步的隐状态$h^{dec}\_{t-1}$作为查询向量，对所用输入序列的隐状态中选择信息：
$$
\begin{aligned}
\boldsymbol{c}\_{t} &=\operatorname{att}\left(\boldsymbol{H}^{\mathrm{enc}}, \boldsymbol{h}\_{t-1}^{\mathrm{dec}}\right)=\sum\_{i=1}^{S} \alpha\_{i} \boldsymbol{h}\_{i}^{\mathrm{enc}} \\\\
&=\sum\_{i=1}^{S} \operatorname{softmax}\left(s\left(\boldsymbol{h}\_{i}^{\mathrm{enc}}, \boldsymbol{h}\_{t-1}^{\mathrm{dec}}\right)\right) \boldsymbol{h}\_{i}^{\mathrm{enc}}
\end{aligned}
$$

将从输入序列中选择的信息$c_t$也作为解码器的输入，得到t步骤的隐状态：
$$
\boldsymbol{h}\_{t}^{\mathrm{dec}}=f\_{\mathrm{dec}}\left(\boldsymbol{h}\_{t-1}^{\mathrm{dec}},\left[\boldsymbol{e}\_{y\_{t-1}} ; \mathbf{c}\_{t}\right], \theta\_{\mathrm{dec}}\right)
$$

最后，将 $\boldsymbol{h}\_{t}^{\mathrm{dec}}$ 输入到分类器得到每个词的概率。

### 15.6.1 基于自注意力的seq2seq
基于CNN的seq2seq除了长程依赖，还有无法并行计算的缺陷，自注意力模型解决了这个问题。这里主要介绍Transformer。

自注意力：
$$
\begin{array}{l}
\operatorname{self}-\operatorname{att}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V})=\boldsymbol{V} \operatorname{softmax}\left(\frac{\boldsymbol{K}^{\top} \boldsymbol{Q}}{\sqrt{D\_{k}}}\right), \\\\
\boldsymbol{Q}=\boldsymbol{W}\_{q} \boldsymbol{H}, \boldsymbol{K}=\boldsymbol{W}\_{k} \boldsymbol{H}, \boldsymbol{V}=\boldsymbol{W}\_{v} \boldsymbol{H}
\end{array}
$$

多头注意力：
$$
\begin{array}{c}
\operatorname{MultiHead}(\boldsymbol{H})=\boldsymbol{W}\_{o}\left[\text { head }\_{1} ; \cdots ; \text { head }\_{M}\right] \\\\
\qquad \text { head }\_{m}=\operatorname{self}-\operatorname{att}\left(\boldsymbol{Q}\_{m}, \boldsymbol{K}\_{m}, \boldsymbol{V}\_{m}\right) \\\\
\forall m \in\{1, \cdots, M\}, \quad \boldsymbol{Q}\_{m}=\boldsymbol{W}\_{q}^{m} \boldsymbol{H}, \boldsymbol{K}=\boldsymbol{W}\_{k}^{m} \boldsymbol{H}, \boldsymbol{V}=\boldsymbol{W}\_{v}^{m} \boldsymbol{H}
\end{array}
$$

**基于self-attention的序列编码**
由于self-attention忽略了未知信息，需要在初始输入序列中加入位置编码：
$$
\boldsymbol{H}^{(0)}=\left[\boldsymbol{e}\_{x\_{1}}+\boldsymbol{p}\_{1}, \cdots, \boldsymbol{e}\_{x\_{T}}+\boldsymbol{p}\_{T}\right]
$$

其中$p_t$为位置编码，可以作为可学习参数，也可以预定义为：
$$
\begin{aligned}
\boldsymbol{p}\_{t, 2 i} &=\sin \left(t / 10000^{2 i / D}\right) \\\\
\boldsymbol{p}\_{t, 2 i+1} &=\cos \left(t / 10000^{2 i / D}\right),
\end{aligned}
$$

$\boldsymbol{p}\_{t, 2 i}$表示第t个位置编码向量的第$2i$维，D是编码向量的维度。

l层隐状态$H^{(l)}$可以通过l-1层的隐状态$H^{(l-1)}$获得：
$$
\begin{array}{l}
\boldsymbol{Z}^{(l)}=\operatorname{norm}\left(\boldsymbol{H}^{(l-1)}+\operatorname{MultiHead}\left(\boldsymbol{H}^{(l-1)}\right)\right) \\\\
\boldsymbol{H}^{(l)}=\operatorname{norm}\left(\boldsymbol{Z}^{(l)}+\operatorname{FFN}\left(\boldsymbol{Z}^{(l)}\right)\right)
\end{array}
$$

这里的FFN为position-wise：
$$
\operatorname{FFN}(z)=W\_{2} \operatorname{ReLu}\left(\boldsymbol{W}\_{1} \boldsymbol{z}+\boldsymbol{b}\_{1}\right)+\boldsymbol{b}\_{2}
$$

基于self-attention的序列编码可以看作全连接的FNN。

**Transfermer**
基于多头自注意力的seq2seq
1. 编码器：多层的多头注意力，输入序列$\boldsymbol{x}\_{1: S}$，输出隐状态序列$\boldsymbol{H}^{\mathrm{enc}}=\left[\boldsymbol{h}\_{1}^{\mathrm{enc}}, \cdots, \boldsymbol{h}\_{S}^{\mathrm{en}}\right]$，再映射为键值对供解码器使用：
$$
\begin{array}{l}
\boldsymbol{K}^{\text {enc }}=\boldsymbol{W}\_{k}^{\prime} \boldsymbol{H}^{\text {enc }}, \\\\
\boldsymbol{V}^{\text {enc }}=\boldsymbol{W}\_{v}^{\prime} \boldsymbol{H}^{\text {enc }}
\end{array}
$$

2. 解码器：自回归方式，三个部分：
	a. 掩蔽自注意模块：使用自注意力编码已知前缀序列$\boldsymbol{y}\_{0:(t-1)}$，得到$\boldsymbol{H}^{\mathrm{dec}}=\left[\boldsymbol{h}\_{1}^{\mathrm{dec}}, \cdots, \boldsymbol{h}\_{t}^{\mathrm{dec}}\right]$
	b. 解码器到编码器模块：将$h_t^{dec}$线性映射得到$q_t^{dec}$，从编码器得到的键值对查询相关信息
	c. 逐位置的FNN：使用FNN综合所有信息
	
训练时对解码器输入处理的trick：掩蔽自注意力（Masked Self-Attention）
- 将右移的目标序列（Right-Shifted Output）$\mathcal{Y}\_{0}:(T-1)$ 作为输入
- 通过掩码来屏蔽后面的输入信息

![c823725b3998cf94ed2ad5b54bd3f69a.png](../../_resources/879223c576e648ae9fad92d61f187084.png)

