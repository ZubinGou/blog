# 【NLP Papers】word2vec improvement


# Distributed Representations of Words and Phrases and their Compositionality
[Mikolov 2013] negative sampling with Skip-gram

## 1 Abstract & Introduction
several extensions of continuous skip-gram that improve quality and speed:
1. subsampling of the frequent words
	- speedup (around 2x - 10x)
	- improve accuracy of less frequent words
	- Noise Contrastive Estimation (NCE) replace hierarchical softmax
2. nagative sampling (alternative to hierarchical softmax)
3. treat word pairs / phase as one word

interesting property of Skip-gram: simple vector addition can often produce meaningful results
- vec(“Russia”) + vec(“river”) is close to vec(“Volga River”)
- vec(“Germany”) + vec(“capital”) is close to vec(“Berlin”)

## 2 The Skip-gram Model
objective is to maximize:
$$
\frac{1}{T} \sum\_{t=1}^{T} \sum\_{-c \leq j \leq c, j \neq 0} \log p\left(w\_{t+j} \mid w\_{t}\right)
$$

defines $p(w\_{t+j}\mid w_t)$ using softmax:
$$
p\left(w\_{O} \mid w\_{I}\right)=\frac{\exp \left(v\_{w\_{O} }^{\prime}{ }^{\top} v\_{w\_{I} }\right)}{\sum\_{w=1}^{W} \exp \left(v\_{w}^{\prime}{ }^{\top} v\_{w\_{I} }\right)}
$$
- impractical because the cost of computing $\nabla \log p\left(w\_{O} \mid w\_{I}\right)$ is proportional to W

### 2.1 Hierarchical Softmax
1. CBOW
输出层为以词在语料出现次数为权值构造的 Huffman 树，每个词为叶子结点，每个分支视为二分类（假设左负右正），路径上的概率之积：
$$
p(w \mid \operatorname{Context}(w))=\prod\_{j=2}^{l^{w} } p\left(d\_{j}^{w} \mid \mathbf{x}\_{w}, \theta\_{j-1}^{w}\right)
$$
其中：
$$
p\left(d\_{j}^{w} \mid \mathbf{x}\_{w}, \theta\_{j-1}^{w}\right)=\left\{\begin{array}{ll}
\sigma\left(\mathbf{x}\_{w}^{\top} \theta\_{j-1}^{w}\right), & d\_{j}^{w}=0 \\\\
1-\sigma\left(\mathbf{x}\_{w}^{\top} \theta\_{j-1}^{w}\right), & d\_{j}^{w}=1
\end{array}\right.
$$
或者写成整体表达式：
$$
p\left(d\_{j}^{w} \mid \mathbf{x}\_{w}, \theta\_{j-1}^{w}\right)=\left[\sigma\left(\mathbf{x}\_{w}^{\top} \theta\_{j-1}^{w}\right)\right]^{1-d\_{j}^{w} } \cdot\left[1-\sigma\left(\mathbf{x}\_{w}^{\top} \theta\_{j-1}^{w}\right)\right]^{d\_{j}^{w} }
$$

带入对数似然函数，采用随机梯度上升即可

2. SG
同理：
$$
p(u \mid w)=\prod\_{j=2}^{l^{u} } p\left(d\_{j}^{u} \mid \mathbf{v}(w), \theta\_{j-1}^{u}\right)
$$
其中：
$$
p\left(d\_{j}^{u} \mid \mathbf{v}(w), \theta\_{j-1}^{u}\right)=\left[\sigma\left(\mathbf{v}(w)^{\top} \theta\_{j-1}^{u}\right)\right]^{1-d\_{j}^{u} } \cdot\left[1-\sigma\left(\mathbf{v}(w)^{\top} \theta\_{j-1}^{u}\right)\right]^{d\_{j}^{u} }
$$


### 2.2 Negative Sampling
增大正样本概率，降低一部分负样本概率：
$$
\log \sigma\left(v\_{w\_{O} }^{\prime}{ }^{\top} v\_{w\_{I} }\right)+\sum\_{i=1}^{k} \mathbb{E}\_{w\_{i} \sim P\_{n}(w)}\left[\log \sigma\left(-v\_{w\_{i} }^{\prime}{ }^{\top} v\_{w\_{I} }\right)\right]
$$
选取概率，noise distribution：
$$
P_n(w)=U(w)^{3 / 4} / Z
$$

### 2.3 Subsampling of Frequent Words
每个词汇以一定概率丢弃：
$$
P\left(w\_{i}\right)=1-\sqrt{\frac{t}{f\left(w\_{i}\right)} }
$$

