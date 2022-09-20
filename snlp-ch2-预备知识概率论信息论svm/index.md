# 《统计自然语言处理》第2章 - 预备知识：概率论、信息论、SVM


## ch2 预备知识
### 2.1 概率论
- 最大似然估计：$q\_{N}\left(s\_{k}\right)=\frac{n\_{N}\left(s\_{k}\right)}{N}$， $\lim \_{N \rightarrow \infty} q\_{N}\left(s\_{k}\right)=P\left(s\_{k}\right)$
- 条件概率：$P(A \mid B)=\frac{P(A \cap B)}{P(B)}$
- 乘法规则：$P(A \cap B)=P(B) P(A \mid B)=P(A) P(B \mid A)$
	- $P\left(A\_{1} \cap \cdots \cap A\_{n}\right)=P\left(A\_{1}\right) P\left(A\_{2} \mid A\_{1}\right) P\left(A\_{3} \mid A\_{1} \cap A\_{2}\right) \cdots P\left(A\_{n} \mid \bigcap\_{i=1}^{n-1} A\_{i}\right)$
- 全概率公式：$P(A)=\sum\_{i} P\left(A \mid B\_{i}\right) P\left(B\_{i}\right)$
- 贝叶斯法则：$P\left(B\_{j} \mid A\right)=\frac{P\left(A \mid B\_{j}\right) P\left(B\_{j}\right)}{P(A)}=\frac{P\left(A \mid B\_{j}\right) P\left(B\_{j}\right)}{\sum\_{i=1}^{n} P\left(A \mid B\_{i}\right) P\left(B\_{i}\right)}$
- 随机变量X的分布函数：$P(X \leqslant x)=F(x), \quad-\infty<x<\infty$
- 二项式分布$\mathrm{X} \sim \mathrm{B}(\mathrm{n}, \mathrm{p})$：$p\_{i}=\left(\begin{array}{c}n \\\\ i\end{array}\right) p^{i}(1-p)^{n-i}, \quad i=0,1, \cdots, n$
- $(X_1, X_2)$ 的联合分布：$p\_{ij}=P\left(X\_{1}=a\_{i}, X\_{2}=b\_{j}\right), \quad i=1,2, \ldots ; j=1,2, \ldots$
- 条件概率分布：$P\left(X\_{1}=a\_{i} \mid X\_{2}=b\_{j}\right)=\frac{p\_{i j}}{\sum\_{k} p\_{k j}}, \quad i=1,2, \cdots$
- 贝叶斯决策理论：
	- $P\left(w\_{i} \mid x\right)=\frac{p\left(x \mid w\_{i}\right) P\left(w\_{i}\right)}{\sum\_{j=1}^{c} p\left(x \mid w\_{j}\right) P\left(w\_{j}\right)}$
	- $p\left(x \mid w\_{i}\right) P\left(w\_{i}\right)=\max \_{j=1,2, \cdots, c} p\left(x \mid w\_{j}\right) P\left(w\_{j}\right),$ 则 $x \in w\_{i}$
- 随机变量$X$的期望（rhs级数收敛时）：$E(X)=\sum\_{k=1}^{\infty} x\_{k} p\_{k}$
- 随机变量$X$的方差：$\begin{aligned} \operatorname{var}(X) &=E\left((X-E(X))^{2}\right) \\\\ &=E\left(X^{2}\right)-E^{2}(X) \end{aligned}$

### 2.2 信息论
#### 熵
- 离散型随机变量$X$的熵：$H(X)=-\sum\_{x \in \mathbf{R}} p(x) \log \_{2} p(x)$
	- 又称自信息（Self-information）
	- 描述不确定性
- 最大熵：在已知部分知识的前提下，关于未知分布最合理的推断应该是符合已知知识最不确定或最大随机的推断。
	- 用熵最大的模型推断某种语言现象的可能性（？）：$\hat{p}=\underset{p \in C}{\operatorname{argmax}} H(p)$

#### 联合熵与条件熵
- 联合熵：$H(X, Y)=-\sum\_{x \in X} \sum\_{y \in Y} p(x, y) \log p(x, y)$
	- 一对随机变量平均所需信息量
- 条件熵：$\begin{aligned} H(Y \mid X) &=\sum\_{x \in X} p(x) H(Y \mid X=x) \\\\ &=\sum\_{x \in X} p(x)\left[-\sum\_{y \in Y} p(y \mid x) \log p(y \mid x)\right] \\\\ &=-\sum\_{x \in X} \sum\_{y \in Y} p(x, y) \log p(y \mid x) \end{aligned}$
- 熵的连锁规则：
	- $\begin{aligned} H(X, Y) &=-\sum\_{x \in X} \sum\_{y \in Y} p(x, y) \log [p(x) p(y \mid x)] \\\\ &=-\sum\_{x \in X} \sum\_{y \in Y} p(x, y)[\log p(x)+\log p(y \mid x)] \\\\ &=-\sum\_{x \in X} \sum\_{y \in Y} p(x, y) \log p(x)-\sum\_{x \in X} \sum\_{y \in Y} p(x, y) \log p(y \mid x) \\\\ &=-\sum\_{x \in X} p(x) \log p(x)-\sum\_{x \in X} \sum\_{y \in Y} p(x, y) \log p(y \mid x) \\\\ &=H(X)+H(Y \mid X) \end{aligned}$
	- 一般情况：$H\left(X\_{1}, X\_{2}, \ldots, X\_{n}\right)=H\left(X\_{1}\right)+H\left(X\_{2} \mid X\_{1}\right)+H\left(X\_{3} \mid X\_{2}, X\_{2}\right)+\ldots+H\left(X\_{n} \mid X\_{n-1}, X\_{n-2}, \ldots, X\_{1}\right)$
- 字符串的熵率：$H\_{\mathrm{rate}}=\frac{1}{n} H\left(X\_{1 n}\right)=-\frac{1}{n} \sum\_{x\_{1 y}} p\left(x\_{1 n}\right) \log p\left(x\_{1 n}\right)$
- 某语言为随机过程$L=\left(X\_{i}\right)$，其熵率：$H\_{\text {rate }}(L)=\lim \_{n \rightarrow \infty} \frac{1}{n} H\left(X\_{1}, X\_{2}, \cdots, X\_{n}\right)$

#### 互信息（mutual information, MI）
- $H(X)-H(X \mid Y)=H(Y)-H(Y \mid X)$
- $\begin{aligned} I(X ; Y) &=H(X)-H(X \mid Y) \\\\ &=H(X)+H(Y)-H(X, Y) \\\\ &=\sum\_{x} p(x) \log \frac{1}{p(x)}+\sum\_{y} p(y) \log \frac{1}{p(y)}+\sum\_{x, y} p(x, y) \log p(x, y) \\\\ &=\sum\_{x, y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)} \end{aligned}$
- 知道Y后X的不确定性的减少量
- ![0d32f4de338fec9a59e9d260aa50d04c.png](/_resources/b23d39e5236741c4b8e42713a392c495.png)
- 体现了两个变量的依赖程度
	- 完全依赖：$I(X ; X)=H(X)-H(X \mid X)=H(X)$，故熵也称自信息
	- 完全独立：$I(X ; Y)=0,$ 即 $p(x, y)=p(x) p(y)$ 
	- $I(X ; Y)\gg 0$：高度相关
	- $I(X ; Y)\ll 0$：Y加大X的不确定性
- 条件互信息：$I(X ; Y \mid Z)=I((X ; Y) \mid Z)=H(X \mid Z)-H(X \mid Y, Z)$
- 互信息连锁规则：$\begin{aligned} I\left(X\_{1 n} ; Y\right) &=I\left(X\_{1}, Y\right)+\cdots+I\left(X\_{n} ; Y \mid X\_{1}, \cdots, X\_{n-1}\right) \\\\ &=\sum\_{i=1}^{n} I\left(X\_{i} ; Y \mid X\_{1}, \cdots, X\_{i-1}\right) \end{aligned}$

#### 相对熵（relative entropy）
- 又称Kullback-Leibler差异（Kullback-Leibler divergence），或简称KL距离/KL散度
- $D(p \| q)=\sum\_{x \in X} p(x) \log \frac{p(x)}{q(x)}=E\_{p}\left(\log \frac{p(x)}{q(x)}\right)$
- 互信息是联合分布与独立分布的相对熵
	- $I(X ; Y)=\sum\_{x, y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)}=D(p(x, y) \| p(x) p(y))$
- 条件相对熵：$D(p(y \mid x) \| q(y \mid x))=\sum\_{x} p(x) \sum\_{y} p(y \mid x) \log \frac{p(y \mid x)}{q(y \mid x)}$
- 相对熵连锁规则：$D(p(x, y) \| q(x, y))=D(p(x) \| q(x))+D(p(y \mid x) \| q(y \mid x))$
 
#### 交叉熵（cross entropy）
- 随机变量X和模型q的交叉熵：
	- $\begin{aligned} H(X, q) &=H(X)+D(p \| q) \\\\ &=-\sum\_{x} p(x) \log q(x) \\\\ &=E\_{p}\left(\log \frac{1}{q(x)}\right) \end{aligned}$
- 语言$L＝(X_i)～p(x)$与其模型q的交叉熵：
	- $H(L, q)=-\lim \_{n \rightarrow \infty} \frac{1}{n} \sum\_{x\_{1}^{n}} p\left(x\_{1}^{n}\right) \log q\left(x\_{1}^{n}\right)$
	- 假定语言L是稳态（stationary）遍历的（ergodic）随机过程，则：
		- $H(L, q)=-\lim \_{n \rightarrow \infty} \frac{1}{n} \log q\left(x\_{1}^{n}\right)$
	- n足够大时，近似为$-\frac{1}{N} \log \left(q\left(x\_{1}^{N}\right)\right)$，交叉熵越小表示模型越接近真实语言模型

#### 困惑度（perplexity）
- $\mathrm{PP}\_{q}=2^{H(L, q)} \approx 2^{-\frac{1}{n} \log q\left(i\_{1}^{n}\right)}=\left[q\left(l\_{1}^{n}\right)\right]^{-\frac{1}{n}}$
- 等价地，语言模型设计的任务就是寻找（对于测试数据）困惑度最小的模型

#### 噪声信道模型（noisy channel model）
- ![95c43e23d7de101182620a86fc2d58f0.png](/_resources/539c0fc798e94c4eab8da80dab7aec84.png)
- 二进制对称信道（binary symmetric channel, BSC）
	- ![c935c1359a8fc70c26f906f0a2993b12.png](/_resources/31e1aed18cec4e97a983011266c32831.png)
- 信道容量（capacity）：$C=\max \_{p(X)} I(X ; Y)$
	- 用降低传输速率来换取高保真通信的可能性
	- 即平均互信息量的最大值
- NLP不需要编码，句子可视为已经编码的符号序列：![6a6c3a286f40601755ea61120285b1a6.png](/_resources/ce281f6e4099463eac2f6c8f3727ea5b.png)
- 给定输出求最可能输入：
	- 贝叶斯公式：$\begin{aligned} \hat{I} &=\underset{I}{\operatorname{argmax}} p(I \mid O)=\underset{I}{\operatorname{argmax}} \frac{p(I) p(O \mid I)}{p(O)} \\\\ &=\underset{I}{\operatorname{argmax}} p(I) p(O \mid I) \end{aligned}$
	- $p(I)$为语言模型（language model），是指在输入语言中“词”序列的概率分布
	- $p(O \mid I)$为信道概率（channel probability）


### 2.3 支持向量机（support vector machine, SVM）
#### 线性分类
- $\begin{aligned} f(x) &=\langle w \cdot x\rangle+b \\\\ &=\sum\_{i=1}^{n} w\_{i} x\_{i}+b \end{aligned}$
- 最优超平面：
	- 以最大间隔分开数据
	- ![f1abf1db54e9e34d45bb40bc65ee0ed7.png](/_resources/6c9ed09b16624e20a1a203811f4198c8.png)
- 多分类问题：
	- 每类一个超平面
	- 决策函数：$c(x)=\underset{1 \leqslant i \leqslant m}{\operatorname{argmax}}\left(\left\langle w\_{i} \cdot x\right\rangle+b\_{i}\right)$

#### 线性不可分
- 非线性问题：映射样本x到高维特征空间，再使用线性分类器
- 假设集：$f(x)=\sum\_{i=1}^{N} w\_{i} \varphi\_{i}(x)+b$
- 决策规则：$f(x)=\sum\_{i=1}^{l} \alpha\_{i} y\_{i}\left\langle\varphi\left(x\_{i}\right) \cdot \varphi(x)\right\rangle+b$
	- 线性分类器重要性质：可以表示成对偶形式
	- 决策规则（分类函数）可以用测试点和训练点的内积来表示
- 核（kernel）函数方法
	- 用原空间中的函数实现高维特征空间的内积

#### 构造核函数
- $K(x, z)=\langle\varphi(x) \cdot \varphi(z)\rangle$
- 决策规则：$f(x)=\sum\_{i=1}^{l} \alpha\_{i} y\_{i} K\left(x\_{i}, x\right)+b$
- 核函数必要条件：
	- 对称：$K(x, z)=\langle\varphi(x) \cdot \varphi(z)\rangle=\langle\varphi(z) \cdot \varphi(x)\rangle=K(z, x)$
	- 内积性质：$\begin{aligned} K(x, z)^{2} &=\langle\varphi(x) \cdot \varphi(z)\rangle^{2} \leqslant\|\varphi(x)\|^{2}\|\varphi(z)\|^{2} \\\\ &=\langle\varphi(x) \cdot \varphi(x)\rangle\langle\varphi(z) \cdot \varphi(z)\rangle=K(x, x) K(z, z) \end{aligned}$
- 核函数充分条件：Mercer定理条件，X的任意有限子集，相应的矩阵是半正定的。
- 核函数充分必要条件
	- 矩阵$K=\left(K\left(x\_{i}, x\_{j}\right)\right)\_{i, j=1}^{n}$半正定（即特征值非负）
- 常用核函数：多项式核函数、径向基函数、多层感知机、动态核函数等
