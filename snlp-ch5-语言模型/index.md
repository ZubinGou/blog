# 《统计自然语言处理》第5章 - 语言模型


- 语言模型（language model, LM）
- 目前主要采用：n元语法模型（n-gram model），构建简单、直接，但同时也因为数据缺乏而必须采取平滑（smoothing）算法
## 5.1 n元语法（n-gram）
- 对于句子$s=w_1w_2...w_l$，其概率计算公式：
$$\begin{aligned} p(s) &=p\left(w\_{1}\right) p\left(w\_{2} \mid w\_{1}\right) p\left(w\_{3} \mid w\_{1} w\_{2}\right) \cdots p\left(w\_{l} \mid w\_{1} \cdots w\_{l-1}\right) \\\\ &=\prod\_{i=1}^{l} p\left(w\_{i} \mid w\_{1} \cdots w\_{i-1}\right) \end{aligned}$$
- 为减少自由参数，将历史$w_1w_2...w\_{i-1}$映射为等价类$E\left(w\_{1} w\_{2} \ldots w\_{i-1}\right)$，假定 $p\left(w\_{i} \mid w\_{1}, w\_{2}, \cdots, w\_{i-1}\right)=p\left(w\_{i} \mid E\left(w\_{1}, w\_{2}, \cdots, w\_{i-1}\right)\right)$
	- 历史划分成等价类方法：最近n-1词相同（n-gram）：
		- $E\left(w\_{1} w\_{2} \ldots w\_{i-1} w\_{i}\right)=E\left(v\_{1} v\_{2} \ldots v\_{k-1} v\_{k}\right)$当仅当$\left(w\_{i-n+2} \ldots w\_{i-1} w\_{i}\right)=\left(v\_{k-n+2} \ldots v\_{k-1} v\_{k}\right)$
- n-gram
	- 一般取n=3
	- n=1：词$w_i$独立与历史，一元文法记作uni-gram、monogram
	- n=2：词$w_i$仅与前一个历史词$w\_{i-1}$有关，二元文法模型称一阶马尔科夫链（Markov Chain），记作bigram、bi-gram
	- n=3：词$w_i$仅与前两个历史词有关，三元文法称二阶马尔科夫链，记作trigram、tri-gram
- 二元语法模型：
	- $p(s)=\prod\_{i=1}^{l} p\left(w\_{i} \mid w\_{1} \ldots w\_{i-1}\right) \approx \prod\_{i=1}^{l} p\left(w\_{i} \mid w\_{i-1}\right)$
	- 假设 $w\_0=\<BOS\>$ 句首标记，结尾 $\<EOS\>$ 句尾标记
	- $\begin{aligned} p(\text { Mark wrote a book }) &=p(\text { Mark } \mid\langle B O S\rangle) \times p(\text { wrote } \mid \text { Mark }) \\\\ \times p(a \mid \text { wrote }) & \times p(\text { book } \mid a) \times p(\langle E O S\rangle \mid \text { book }) \end{aligned}$
	- 最大似然估计（maximum likelihood estimation, MLE），统计频率然后归一化得到：$p\left(w\_{i} \mid w\_{i-1}\right)=\frac{c\left(w\_{i-1} w\_{i}\right)}{\sum\_{w\_{i} } c\left(w\_{i-1} w\_{i}\right)}$
- n元语法模型
	- $p(s)=\prod\_{i=1}^{l-1} p\left(w\_{i} \mid w\_{i-n+1}^{i-1}\right)$
	- 约定$w\_{-n+2}$到$w\_0$为 $\<BOS\>$ ， $w\_{l+1}=\<EOS\>$
	- 最大似然估计：$p\left(w\_{i} \mid w\_{i-n+1}^{i-1}\right)=\frac{c\left(w\_{i-n+1}^{i}\right)}{\sum\_{w\_{i} } c\left(w\_{i-n+1}^{i}\right)}$


## 5.2 语言模型性能的评价
- 常用度量：	
	- 模型计算出测试数据的概率
		- 对句子$\left(t\_{1}, t\_{2}, \ldots, t\_{l\_{T} }\right)$构造的测试集T：
			- $p(T)=\prod\_{i=1}^{l\_{T} } p\left(t\_{i}\right)$
	- cross-entropy
		- $H\_{p}(T)=-\frac{1}{W\_{T} } \log \_{2} p(T)$
		- 表示利用压缩算法对数据集中$W_T$个词进行编码，每个编码平均比特位数
	- perplexity 困惑度
		- $P P\_{T}(T)=2^{H P(T)}$
		- 模型分配给测试集T中每一个词汇的概率的几何平均值的倒数
- 在英语文本中，n元语法模型计算的困惑度范围大约为50～1000之间（对应的交叉熵范围为6～10个比特位），具体值与文本的类型有关


## 5.3 数据平滑
### 5.3.1 问题的提出
- 数据平滑（data smoothing）：避免零概率问题
- 基本思想：劫富济贫，提高低概率、降低高概率
- 加一法：
	- $p\left(w\_{i} \mid w\_{i-1}\right)=\frac{1+c\left(w\_{i-1} w\_{i}\right)}{\sum\_{w\_{i} }\left[1+c\left(w\_{i-1} w\_{i}\right)\right]}=\frac{1+c\left(w\_{i-1} w\_{i}\right)}{|V|+\sum\_{w\_{i} } c\left(w\_{i-1} w\_{i}\right)}$
	- $|V|$为词汇表单词个数

### 5.3.2 加法平滑方法
- 假设每一个n元语法发生的次数比实际次数多$\delta$次，$0 \leq \delta \leq 1$
- $p\_{\text {add } }\left(w\_{i} \mid w\_{i-n+1}^{i-1}\right)=\frac{\delta+c\left(w\_{i-n-1}^{i}\right)}{\delta|V|+\sum\_{w\_{i} } c\left(w\_{i-n+1}^{i}\right)}$

### 5.3.2 古德-图灵（Good-Turing）估计法
- 基本思路：假定出现$r$次的n元语法出现$r^*$次：
	- $r^{*}=(r+1) \frac{n\_{r+1} }{n\_{r} }$
- $n_r$是训练语料中恰好出现r次的n元语法的数目
- 归一化：
	- $p\_{r}=\frac{r^{\*} }{N}$，$N=\sum\_{r=0}^{\infty} n\_{r} r^{\*}$
- N等于分布最初计数：$N=\sum\_{r=0}^{\infty} n\_{r} r^{*}=\sum\_{r=0}^{\infty}(r+1) n\_{r+1}=\sum\_{r=1}^{\infty} n\_{r} r$
- 所有事件概率和：$\sum\_{r>0} n\_{r} p\_{r}=1-\frac{n\_{1} }{N}<1$
- 有$n\_{1} / {N}$的概率分给r=0的未见事件
