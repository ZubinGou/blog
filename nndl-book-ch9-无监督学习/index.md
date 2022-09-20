# 《神经网络与深度学习》第9章 - 无监督学习

ch9 无监督学习

## 9.1 无监督特征学习
无监督学习问题分类：
1. 无监督特征学习（Unsupervised Feature Learning）
	- 降维、可视化、监督学习前的预处理
2. 概率密度估计（Probabilistic Density Estimation）
3. 聚类（Clustering）
	- K-Means、谱聚类

监督学习、无监督学习三要素：
1. 模型
2. 学习准则
	- 最大似然估计（密度估计常用）、最小重构错误（无监督特征学习常用）
3. 优化算法

### 9.1.1 主成分分析

主成分分析（Principal Component Analysis，PCA）：数据降维，使转换后的空间中数据方差最大。

样本投影方差：

$$
\begin{aligned}
\sigma(\boldsymbol{X} ; \boldsymbol{w}) &=\frac{1}{N} \sum\_{n=1}^{N}\left(\boldsymbol{w}^{\top} \boldsymbol{x}^{(n)}-\boldsymbol{w}^{\top} \overline{\boldsymbol{x}}\right)^{2} \\\\
&=\frac{1}{N}\left(\boldsymbol{w}^{\top} \boldsymbol{X}-\boldsymbol{w}^{\top} \overline{\boldsymbol{X}}\right)\left(\boldsymbol{w}^{\top} \boldsymbol{X}-\boldsymbol{w}^{\top} \overline{\boldsymbol{X}}\right)^{\top} \\\\
&=\boldsymbol{w}^{\top} \boldsymbol{\Sigma} \boldsymbol{w}
\end{aligned}
$$

其中：

$$
\boldsymbol{\Sigma}=\frac{1}{N}(\boldsymbol{X}-\overline{\boldsymbol{X}})(\boldsymbol{X}-\overline{\boldsymbol{X}})^{\top}
$$

即原样本的协方差矩阵。

拉格朗日方法转化为无约束优化：

$$
\max \_{\boldsymbol{w}} \boldsymbol{w}^{\top} \boldsymbol{\Sigma} \boldsymbol{w}+\lambda\left(1-\boldsymbol{w}^{\top} \boldsymbol{w}\right)
$$

求导令导数为0：

$$
\boldsymbol{\Sigma} \boldsymbol{w}=\lambda \boldsymbol{w}
$$

𝒘 是协方差矩阵 𝚺 的特征向量，𝜆 为特征值．同时
$$
\sigma(\boldsymbol{X} ; \boldsymbol{w})=\boldsymbol{w}^{\top} \boldsymbol{\Sigma} \boldsymbol{w}=\boldsymbol{w}^{\top} \lambda \boldsymbol{w}=\lambda
$$

因此，PCA 可以转换为矩阵特征值分解，投影向量 𝒘 为矩阵 𝚺 的最大特征值对应的特征向量。取前 $D'$ 个特征向量：

$$
\boldsymbol{\Sigma} \boldsymbol{W}=\boldsymbol{W} \operatorname{diag}(\lambda)
$$

PCA 减少了数据相关性，但不能保证投影后数据类别可分性更好。提高可分类性的方法一般为监督方法，如线性判别分析（Linear Discriminant Analysis，LDA）

PCA 一个明显的缺点是失去了特征的可解释性。

### 9.1.2 稀疏编码
稀疏编码（Sparse Coding）

启发：哺乳动物视觉细胞感受野，每个神经元仅对其感受野的特定刺激做出响应，外界刺激子在视觉神经系统的表示具有稀疏性，符合生物低功耗特性。

线性编码：将输入的样本表示为一组基向量的线性组合，在 P 维空间中表示 D 维空间的样本 x：

$$
\begin{aligned}
\boldsymbol{x} &=\sum\_{m=1}^{M} z\_{m} \boldsymbol{a}\_{m} \\\\
&=\boldsymbol{A z},
\end{aligned}
$$

基向量 A 也称为字典

编码的关键：找到一组完备基向量，如通过PCA。PCA得到的编码通常是稠密向量，没有稀疏性。

> 完备：基向量数等于其支撑维度（组成满秩方阵）
> 过完备：基向量数大于其支撑的维度。

为了得到稀疏编码，可以找一组“过完备”的基向量，加上稀疏性限制，得到“唯一”稀疏编码。

对一组输入 x 的稀疏编码目标函数：
$$
\mathcal{L}(\boldsymbol{A}, \boldsymbol{Z})=\sum\_{n=1}^{N}\left(\left\|\boldsymbol{x}^{(n)}-A \boldsymbol{z}^{(n)}\right\|^{2}+\eta \rho\left(\boldsymbol{z}^{(n)}\right)\right),
$$

𝜌(⋅) 是一个稀疏性衡量函数，𝜂 是一个超参数，用来控制稀疏性的强度

稀疏性定义：向量非零元素的比例。大多数元素接近零的向量也成为稀疏向量。

**衡量稀疏性**

$\ell\_{0}$ 范数：

$$
\rho(\boldsymbol{z})=\sum\_{m=1}^{M} \mathbf{I}\left(\left|z\_{m}\right|>0\right)
$$

不满足连续可导，很难优化，所以稀疏性衡量函数常使用 $\ell\_{1}$ 范数：

$$
\rho(\boldsymbol{z})=\sum\_{m=1}^{M}\left|z\_{m}\right|
$$

或对数函数：

$$
\rho(\boldsymbol{z})=\sum\_{m=1}^{M} \log \left(1+z\_{m}^{2}\right)
$$

或指数函数：

$$
\rho(z)=\sum\_{m=1}^{M}-\exp \left(-z\_{m}^{2}\right)
$$

稀疏表示的本质：用尽可能少的资源表示尽可能多的知识，人脑皮质层学习输入表征采用了这一方法，对熟练的东西会调用更少的脑区域。

**训练方法**
训练目标：基向量A、每个输入的表示

优化方法：交替优化
1. 固定基向量，优化编码：

$$
\min \_{z^{(n)}}\left\|\boldsymbol{x}^{(n)}-\boldsymbol{A} \boldsymbol{z}^{(n)}\right\|^{2}+\eta \rho\left(\boldsymbol{z}^{(n)}\right), \forall n \in[1, N]
$$

2. 固定编码，优化基向量：

$$
\min \_{\boldsymbol{A}} \sum\_{n=1}^{N}\left(\left\|\boldsymbol{x}^{(n)}-\boldsymbol{A} \boldsymbol{z}^{(n)}\right\|^{2}\right)+\lambda \frac{1}{2}\|\boldsymbol{A}\|^{2}
$$

**稀疏编码优点（相比稠密向量的分布式表示）**
1. 计算量小
2. 可解释性强：编码对应少数特征
3. 特征选择：自动选择和输入相关的少数特征，降低噪声，减少过拟合。


### 9.1.3 自编码器
自编码器（Auto-Encoder，AE）：通过无监督方法学习一组数据的有效编码

思路：将 x 通过编码器转换为中间变量 y，再将 y 通过解码器转换为输出 $\bar{x}$，目标是使得输出和输入无限接近。

作用：使用其中的编码器进行特征降维，作为 ML 模型的输入。

优化目标：最小重构错误（Reconstrcution Error）：

$$
\begin{aligned}
\mathcal{L} &=\sum\_{n=1}^{N}\left\|\boldsymbol{x}^{(n)}-g\left(f\left(\boldsymbol{x}^{(n)}\right)\right)\right\|^{2} \\\\
&=\sum\_{n=1}^{N}\left\|\boldsymbol{x}^{(n)}-f \circ g\left(\boldsymbol{x}^{(n)}\right)\right\|^{2} .
\end{aligned}
$$

- 特征空间维度 M 一般小于原始空间维度，AE 相当于是降维/特征抽取。
- 当 $M \geq D$ 时，存在解使得 $f \circ g$ 为单位函数，使得损失为0，解就没有太多意义。
- 当加上限制，如编码稀疏性、取值范围、f和g的形式等，可以得到有意义的解

> 如让编码只能去 K 个不同的值，则变为了 K 聚类问题。

![387ff345a4e6635d6ca899745edba0cd.png](../resources/f7159ea0351e4c169a448a92cc1e8652.png)

编码器：

$$
\boldsymbol{z}=f\left(\boldsymbol{W}^{(1)} \boldsymbol{x}+\boldsymbol{b}^{(1)}\right)
$$

解码器：

$$
\boldsymbol{x}^{\prime}=f\left(\boldsymbol{W}^{(2)} \boldsymbol{z}+\boldsymbol{b}^{(2)}\right)
$$

捆绑权重（Tied Weight）：令 $\boldsymbol{W}^{(2)}=\boldsymbol{W}^{(1)^{\top}}$ ，参数更少，更容易学习，同时有一定正则化作用。

重构错误：

$$
\mathcal{L}=\sum\_{n=1}^{N} \| \boldsymbol{x}^{(n)}-\boldsymbol{x}^{\prime(n)}\|^{2}+\lambda\| \boldsymbol{W} \|\_{F}^{2}
$$


### 9.1.4 稀疏自编码器
稀疏自编码器（Sparse Auto-Encoder）：让特征维度 M 大于输入维度 D，并使特征尽量稀疏的自编码器。

目标函数：

$$
\mathcal{L}=\sum\_{n=1}^{N} \| \boldsymbol{x}^{(n)}-\boldsymbol{x}^{\prime(n)}\|^{2}+\eta \rho(\boldsymbol{Z})+\lambda\| \boldsymbol{W} \|^{2}
$$

𝜌(𝒁) 为稀疏性度量函数，可以用稀疏编码的稀疏衡量函数，也可以定义为一组训练样本中每个神经元激活的概率，用平均活性值近似：

$$
\hat{\rho}\_{j}=\frac{1}{N} \sum\_{n=1}^{N} z\_{j}^{(n)}
$$

我们希望稀疏度接近实现给定的值 $\rho^*$，如0.05，用 KL 距离衡量：
$$
\mathrm{KL}\left(\rho^{*} \| \hat{\rho}\_{j}\right)=\rho^{*} \log \frac{\rho^{*}}{\hat{\rho}\_{j}}+\left(1-\rho^{*}\right) \log \frac{1-\rho^{*}}{1-\hat{\rho}\_{j}}
$$

稀疏性度量函数定义为：
$$
\rho(\boldsymbol{Z})=\sum\_{j=1}^{p} \mathrm{KL}\left(\rho^{*} \| \hat{\rho}\_{j}\right)
$$

### 9.1.5 堆叠自编码器
堆叠自编码器（Stacked Auto-Encoder，SAE）：使用逐层堆叠的方式训练深层的自编码器，可以采用逐层训练（Layer-Wise Training）来学习参数。

### 9.1.6 降噪自编码器
降噪自编码器（Denoising Auto-Encoder）：通过引入噪声来增加编码鲁棒性的自编码器。

![cb284fd8da4888a86a1779974f3c4999.png](../resources/373033d6f5ff487f856b739063c54882.png)


## 9.2 概率密度估计

概率密度估计（Probabilistic Density Estimation）：简称密度估计，即基于样本估计随机变量的概率密度函数。

### 9.2.1 参数密度估计
参数密度估计（Parametric Density Estimation）：根据先验知识假设随机变量服从某种分布，然后用训练样本估计分布的参数。

对样本 D 的对数似然函数：
$$
\log p(\mathcal{D} ; \theta)=\sum\_{n=1}^{N} \log p\left(\boldsymbol{x}^{(n)} ; \theta\right)
$$

可以使用最大似然估计（MLE）来寻找参数，参数估计问题转变为最优化问题：
$$
\theta^{M L}=\underset{\theta}{\arg \max } \sum\_{n=1}^{N} \log p\left(\boldsymbol{x}^{(n)} ; \theta\right) .
$$

**正态分布**

![c0a11ab6135c5b571e3009894635f6f5.png](../resources/ffa1e97aa8204808b9cceb74cca652f2.png)

**多项分布**

![62536cdcd2d549dfd9b192d7698543ae.png](../resources/81f5231f3b2643d6a7cfecef6fb5efc0.png)

求导数为0得：

$$
\mu\_{k}^{M L}=\frac{m\_{k}}{N}, \quad 1 \leq k \leq K
$$

参数密度估计的问题：
1. 模型选择：实际分布往往复杂
2. 不可观测：一些关键变量无法观测，很难准确估计数据真实分布
3. 维度灾难：高维数据参数估计困难，需要大量样本避免过拟合。

### 9.2.2 非参数密度估计
非参数密度估计（Nonparametric Density Estimation）：不假设数据服从某种分布，通过样本空间划分为不同的区域并估计每个区域概率来近似概率密度函数。

高维空间中随机向量 x，假设其服从未知分布 p(x)，则 x 落入小区域 R 的概率为：

$$
P=\int\_{\mathcal{R}} p(\boldsymbol{x}) d \boldsymbol{x} .
$$

N 个样本中落入 R 的数量 K 服从二项分布：

$$
P\_{K}=(\begin{array}{l}
N \\\\
K
\end{array}) P^{K}(1-P)^{1-K}
$$

N 很大时，可以近似认为：

$$
P \approx \frac{K}{N}
$$

假设 R 足够小，内部概率均匀：

$$
P \approx p(\boldsymbol{x}) V
$$

综上：

$$
p(\boldsymbol{x}) \approx \frac{K}{N V}
$$

非参数密度估计常用方法：
1. 固定区域 V，统计落入不同区域的数量
	- 直方图方法
	- 核方法
2. 改变区域大小，使得落入每个区域的样本数量为 K：K邻近法

**直方图方法（Histogram Method）**
直观可视化低维数据分布，很难扩展到高维变量（维度灾难）


**核密度估计（Kernel Density Estimation）**
也叫 Parzen 窗方法

定义超立方体核函数：
$$
\phi\left(\frac{\boldsymbol{z}-\boldsymbol{x}}{H}\right)= \begin{cases}1 & \text { if }\left|z\_{i}-x\_{i}\right|<\frac{H}{2}, 1 \leq i \leq D \\\\ 0 & \text { else }\end{cases}
$$

求和得到落入 R 区域的样本数量：

$$
K=\sum\_{n=1}^{N} \phi\left(\frac{\boldsymbol{x}^{(n)}-\boldsymbol{x}}{H}\right)
$$

x 点概率密度估计：

$$
p(\boldsymbol{x})=\frac{K}{N H^{D}}=\frac{1}{N H^{D}} \sum\_{n=1}^{N} \phi\left(\frac{\boldsymbol{x}^{(n)}-\boldsymbol{x}}{H}\right)
$$

也可以采用更加平滑的高斯核函数：

$$
\phi\left(\frac{z-x}{H}\right)=\frac{1}{(2 \pi)^{1 / 2} H} \exp \left(-\frac{\|z-x\|^{2}}{2 H^{2}}\right)
$$

则 x 点概率密度估计：

$$
p(\boldsymbol{x})=\frac{1}{N} \sum\_{n=1}^{N} \frac{1}{(2 \pi)^{1 / 2} H} \exp \left(-\frac{\|\boldsymbol{z}-\boldsymbol{x}\|^{2}}{2 H^{2}}\right)
$$

**K 近邻方法（K-Nearest Neighbor Method）**
估计 x 点密度：
1. 找到以 x 为中心的球体，使得落入球体的样本数量为 K
2. 利用下式计算密度：

$$
p(\boldsymbol{x}) \approx \frac{K}{N V}
$$

## 9.3 总结和深入阅读
概率密度估计与后文关联：
- ch11：通过概率图模型介绍更一般的参数密度估计方法，包括含隐变量的参数估计方法
- ch12：两种比较复杂的生成模型：玻尔兹曼机、深度信念网络
- ch13：两种深度生成模型：变分自编码器、对抗生成网络
- ch15：序列生成模型

> 生成模型：根据参数估计出的模型来生成数据。

无监督学习没有监督学习成功的原因：缺少有效的客观评价方法，无监督方法好坏需要代入下游任务中验证。

## 习题选做
#### 习题 9-1 分析主成分分析为什么具有数据降噪能力？
PCA的核心思想是：将数据集映射到用一组特征向量（基）来表示，数据集在某个基上的投影即是特征值。噪声与主要特征一般不相关，所以较小的特征值往往对应着噪声的方差，去掉较小的特征可以减小噪声。

#### 习题 9-3 对于一个二分类问题，试举例分析什么样的数据分布会使得主成分分析得到的特征反而会使得分类性能下降．
不满足方差越大，信息量越多的假设时，如下图：

![864f3722f01240db2d3552b44c453ad3.png](../resources/0ca43124885f4d1a9f8c84aefa6ee1a7.png)

PCA 会按照 y 轴降维，使得数据两类数据混在一起而不可分（这里可以使用有监督的 LDA 降维）。

同样，当噪声过大时、数据维度本身较小时，也不适合用PCA。


#### 习题 9-5 举例说明，K 近邻方法估计的密度函数不是严格的概率密度函数，其在整个空间上的积分不等于 1．

> exercise 2.61: Show that the K-nearest-neighbor density model defines an improper distribution whose integral over all space is divergent.
> -- Bishop's pattern recognition and machine learning

证明思路：概率密度函数在 $(-\infty, \infty)$ 上求积分不收敛到1，而是 $\infty$


假设一维条件下 $K=1$ 的 KNN 密度估计，有一个点 $x=0$，则 $x$ 处密度估计为：

$$
p(x)=\frac{K}{N V}=\frac{1}{|x|}
$$

其中 𝑉 为区域 ℛ 的体积。

当 $N=1$ 时满足：
$$
\int\_{-\infty}^{\infty} p(x) \mathrm{d} x=\infty
$$

当 $N \gt 1$ 时，假设有一系列点：

$$
X\_{1} \leq X\_{2} \leq \ldots \leq X\_{N}
$$

对 $x\leq X_1$ 的部分：

$$
p(x)=\frac{K}{N\left(X\_{k}-x\right)}, \quad x \leq X\_{1}
$$

我们只计算这部分的积分：

$$
\int\_{-\infty}^{X\_{1}} \frac{K}{N\left(X\_{k}-x\right)} \mathrm{d} x=\left[\frac{K}{N} \ln \left|X\_{k}-x\right|\right]\_{-\infty}^{X\_{1}}=\infty
$$

由于密度为正，所以在 $(-\infty, \infty)$ 上的积分也发散，从而说明了 KNN 密度估计并不严格。
