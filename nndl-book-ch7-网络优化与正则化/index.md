# 《神经网络与深度学习》第7章 - 网络优化与正则化



# ch7 网络优化与正则化
> 任何数学技巧都不能弥补信息的缺失．
> —— 科尼利厄斯·兰佐斯（Cornelius Lanczos） 匈牙利数学家、物理学家

神经网络应用两大问题：
1. 优化问题
	- 非凸损失函数难以找到全局最优解
	- 参数多、数据大，使得二阶优化方法（牛顿法等）代价过高、一阶优化方法训练效率低
	- 梯度消失和梯度爆炸使基于梯度的训练方法失效
2. 泛化问题
	- 深度方法拟合能力强，容易过拟合，采用正则化方法改进模型泛化能力

## 7.1 网络优化
网络优化：寻找一个神经网络模型来使得经验（或结构）风险最小化的过程。

DNN 是高度非线性模型，其风险函数为非凸函数，因此风险最小化为非凸优化问题。

神经网络结构具有多样性，很难找到通用的优化方法。

**高维变量的非凸优化**

- 低维空间的非凸优化：主要问题是存在局部最优点，难点是初始化参数和逃离局部最优点。
- 高维空间的非凸优化：难点是逃离鞍点（Saddle Point），即既是某些维的最高点，又是另一些维的最低点。
	- 高维空间中大部分驻点（Stationary Point）都是鞍点
	- 局部最小值（Local Minima）：每个维度都是最低点，概率非常低
	- 随机梯度下降可以有效逃离鞍点

平坦最小值（Flat Minima）和尖锐最小值（Sharp Minma）：深度网络参数多且冗余，局部最小解通常是平坦最小值，鲁棒性、抗扰动能力较好。

![8b03b6bdabdd1b5b012138971fef2d31.png](/blog/_resources/2603c35aca6a406d9f7647220cb5b9f4.png)

局部最小解等价性：大神经网络中，大部分局部最小解等价，且接近全局最小解的训练损失。没有必要找全局最小值，这反而可能过拟合。

**神经网络优化的改善方法**
1. 更有效优化算法：动态学习率、梯度估计修正等
2. 更好的参数初始化、数据预处理
3. 修改网络结构得到更好的优化地形（Optimization Landscape）：ReLU激活、残差、逐层归一化等
4. 更好的超参数优化方法

## 7.2 优化算法
DNN 主要通过梯度下降法寻找最小化结构风险的参数，分为：
1. 批量梯度下降
2. 随机梯度下降
3. 小批量梯度下降

优化算法分类：
1. 调整学习率，使优化更稳定
2. 梯度估计修正，优化训练速度

### 7.2.1 小批量梯度下降
小批量梯度下降法（Mini-Batch Gradient Descent）：每次迭代 K 个训练样本

第 t 次迭代偏导数：

$$
\mathfrak{g}_{t}(\theta)=\frac{1}{K} \sum\_{(\boldsymbol{x}, \boldsymbol{y}) \in \mathcal{S}\_{t}} \frac{\partial \mathcal{L}(\boldsymbol{y}, f(\boldsymbol{x} ; \theta))}{\partial \theta}
$$

参数更新
$$
\theta\_{t} \leftarrow \theta\_{t-1}-\alpha g\_{t}
$$

影响小批量梯度下降的因素：
1. 批量大小 K
2. 学习率 $\alpha$
3. 梯度估计

### 7.2.2 批量大小选择
Batch Size 不影响梯度期望，但影响方差，越大越稳定

Batch Size 较小时候可以采用**线性缩放规则（Linear Scaling Rule）**：Batch Size 和学习率同比率增大 [Goyal et al., 2017]

大批量稳定，小批量收敛快

大批量越可能收敛到尖锐最小值，小批量越可能收敛到平坦最小值（泛化更好）[Keskar et al., 2016] （应与小批量随机性更大有关）

### 7.2.3 学习率调整
常用 lr 调整方法：
1. 学习率衰减
2. 学习率预热
3. 周期性学习率调整
4. 自适应调整学习率方法（AdaGrad、RMSprop、AdaDelta，对每个参数设置不同学习率）

#### 学习率衰减（Learning Rate Decay）/ 学习率退火（Learning Rate Anealing）

- 分段常数衰减（Piecewise Constant Decay）/ 阶梯衰减（Step Decay）
- 逆时衰减（Inverse Time Decay）
$$
\alpha\_{t}=\alpha\_{0} \frac{1}{1+\beta \times t}
$$

- 指数衰减（Exponential Decay）

$$
\alpha\_{t}=\alpha\_{0} \beta^{t}
$$

- 自然指数衰减（Natural Exponential Decay）

$$
\alpha\_{t}=\alpha\_{0} \exp (-\beta \times t)
$$

- 余弦衰减（Cosine Decay）

$$
\alpha\_{t}=\frac{1}{2} \alpha\_{0}\left(1+\cos \left(\frac{t \pi}{T}\right)\right)
$$

![0e85ca8db068d0e269b573f6b0cf7864.png](/blog/_resources/665b10f03842415d808436f36a393e20.png)

#### 学习率预热
常用：逐渐预热（Gradual Warmup）[Goyal et al., 2017]

$$
\alpha\_{t}^{\prime}=\frac{t}{T^{\prime}} \alpha\_{0}, \quad 1 \leq t \leq T^{\prime}
$$

#### 周期性学习率调整
- 循环学习率（Cyclic Learning Rate）
	- 三角循环学习率（Triangular Cyclic Learning Rate）
- 带热重启的随机梯度下降（Stochastic Gradient Descent with Warm Restarts，SGDR）[Loshchilov et al., 2017a]
	- 重启之后再余弦衰减

![639eec26a903370e875d5b29e0770788.png](/blog/_resources/983b26c9b4674a8e971bee5a5966cb77.png)

#### AdaGrad 算法
AdaGrad 算法（Adaptive Gradient Algorithm）[Duchi et al., 2011] ：借鉴 l2 正则化思想，每次迭代自适应调整每个参数的学习率。

每个参数梯度平方累计值：

$$
G\_{t}=\sum\_{\tau=1}^{t} \boldsymbol{g}\_{\tau} \odot \boldsymbol{g}\_{\tau}
$$

参数更新差值：

$$
\Delta \theta\_{t}=-\frac{\alpha}{\sqrt{G\_{t}+\epsilon}} \odot \mathbf{g}\_{t}
$$

Hung-yi Lee:

![d8d82ed693e6086e2055efcfa59b2359.png](/blog/_resources/f691c2979f8141359831286b705234b2.png)
![599d94f6bc898d9d2189834d9649dd79.png](/blog/_resources/fada25cef7e34b2592370d322c1762ac.png)

用梯度平方和近似二次微分

AdaGrad 缺点：一定次数迭代后如果没有到最优点，而学习率已经非常小，难以再继续优化。

#### RMSprop 算法
避免 AdaGrad 学习率过早衰减到零。

梯度平方的指数衰减移动平均：

$$
\begin{aligned}
G\_{t} &=\beta G\_{t-1}+(1-\beta) g\_{t} \odot g\_{t} \\\\
&=(1-\beta) \sum\_{\tau=1}^{t} \beta^{t-\tau} g\_{\tau} \odot g\_{\tau}
\end{aligned}
$$

参数更新差值：

$$
\Delta \theta\_{t}=-\frac{\alpha}{\sqrt{G\_{t}+\epsilon}} \odot \mathbf{g}\_{t}
$$

#### AdaDelta 算法
AdaDelta 在 RMSprop 基础上引入参数更新差值的平方指数衰减移动平均，抑制了学习率的波动

$$
\Delta X\_{t-1}^{2}=\beta\_{1} \Delta X\_{t-2}^{2}+\left(1-\beta\_{1}\right) \Delta \theta\_{t-1} \odot \Delta \theta\_{t-1}
$$

参数更新差值：

$$
\Delta \theta\_{t}=-\frac{\sqrt{\Delta X\_{t-1}^{2}+\epsilon}}{\sqrt{G\_{t}+\epsilon}} \mathrm{g}\_{t}
$$


### 7.2.4 梯度估计修正
梯度估计（Gradient Estimation）的修正

#### 动量法
每次迭代计算负梯度的“加权移动平均”作为参数更新方向，增加稳定性：

$$
\Delta \theta\_{t}=\rho \Delta \theta\_{t-1}-\alpha g\_{t}(\theta\_{t-1})=-\alpha \sum\_{\tau=1}^{t} \rho^{t-\tau} g\_{\tau}
$$

其中 $\rho$ 为动量因子，通常设为0.9，$\alpha$ 为学习率

当前梯度叠加部分上次梯度，可以近似看作二阶梯度。

#### Nesterov 加速梯度
Nesterov 加速梯度（Nesterov Accelerated Gradient，NAG）/ Neserov 动量法（Nesterov Momentum）：对动量法的改进，在根据历史梯度更新后的位置计算梯度更新，更加合理。

$$
\Delta \theta\_{t}=\rho \Delta \theta\_{t-1}-\alpha \mathfrak{g}\_{t}\left(\theta\_{t-1}+\rho \Delta \theta\_{t-1}\right)
$$

![97f64dd99c652b6ba0010ff05ecfe20f.png](/blog/_resources/924d30374b8644e588ea24a547cd16f7.png)

#### Adam 算法
Adam算法（Adaptive Moment Estimation Algorithm）[Kingma et al., 2015]：梯度平方指数加权平均（RMSprop）+ 梯度指数加权平均（Momentum）

$$
\begin{gathered}
M\_{t}=\beta\_{1} M\_{t-1}+\left(1-\beta\_{1}\right) g\_{t} \\\\
G\_{t}=\beta\_{2} G\_{t-1}+\left(1-\beta\_{2}\right) g\_{t} \odot g\_{t}
\end{gathered}
$$

可以分别看作梯度的均值（一阶矩）和未减去均值的方差（二阶矩）。

需要进行偏差修正：
$$
\begin{aligned}
\hat{M}\_{t} &=\frac{M\_{t}}{1-\beta\_{1}^{t}} \\\\
\hat{G}\_{t} &=\frac{G\_{t}}{1-\beta\_{2}^{t}}
\end{aligned}
$$

更新差值：
$$
\Delta \theta\_{t}=-\frac{\alpha}{\sqrt{\hat{G}\_{t}+\epsilon}} \hat{M}\_{t}
$$

Nadam算法：用 Nesterov 加速梯度改进Adam

#### 梯度截断
梯度截断（Gradient Clipping）[Pascanu et al., 2013]

1. 按值截断：对所有参数设置范围
$$
\boldsymbol{g}\_{t}=\max \left(\min \left(\boldsymbol{g}\_{t}, b\right), a\right)
$$

2. 按模截断：二范数超过阈值时整体缩放，适合 RNN。

$$
\boldsymbol{g}\_{t}=\frac{b}{\left\|\boldsymbol{g}\_{t}\right\|}\_2 \boldsymbol{g}\_{t}
$$

> 书中二范数符号不准确，一般范数下标应该在右下角，右上角容易误解为平方。


### 7.2.5 优化算法小结
优化算法公式概括：

$$
\begin{aligned}
\Delta \theta\_{t} &=-\frac{\alpha\_{t}}{\sqrt{G\_{t}+\epsilon}} M\_{t} \\\\
G\_{t} &=\psi\left(\mathbf{g}\_{1}, \cdots, \boldsymbol{g}\_{t}\right) \\\\
M\_{t} &=\phi\left(\mathbf{g}\_{1}, \cdots, \mathbf{g}\_{t}\right)
\end{aligned}
$$

![23f888ad5e6500a4bda71388fb5414ca.png](/blog/_resources/4c7d331e8c7145f08f3b6a5045c5dbe0.png)

![200397e2f0dc522a9ce06ad72220d8b0.png](/blog/_resources/dee48c37fad84a14a3c69cdf0839b725.png)


## 7.3 参数初始化
![f6dc542859738aaa426d8146bcdefb73.png](/blog/_resources/f15690282dcb4ca8be099fc66bafe0f2.png)
*来源：https://www.cnblogs.com/shine-lee/p/11908610.html*


参数初始化非常关键，关系到网络优化效率和泛化能力。

初始化通常三种做法：
1. 预训练初始化（Pre-trained Initialization）
	- 不够灵活，网络架构调整不便
2. 随机初始化（Random Initialization）
	- 解决**对称权重**问题：相同初始值权重更新相同
	- 三种随机初始化方法：
		1. 基于固定方差
		2. 基于方差缩放
		3. 正交初始化 
3. 固定值初始化：对特定参数用特殊值初始化
	- bias 设 0
	- LSTM 遗忘门初始化为 1 或 2，使时序梯度变大
	- 使用 ReLU 的神经元，偏置可以设置为 0.01 使其初期更容易被激活


### 7.3.1 基于固定方差的参数初始化
1. 高斯分布初始化
2. 均匀分布初始化

均匀分布方差：
$$
\operatorname{var}(x)=\frac{(b-a)^{2}}{12}
$$

区间为 $[-r, r]$ 则：
$$
r=\sqrt{3 \sigma^{2}}
$$

关键是设置方差 $\sigma^2$ ：
- 取值小：1. 神经元输出小，多层后信号消失 2. 使 Sigmoid 型函数丢失非线性激活能力
- 取值大：Sigmoid 型函数激活值饱和，导致梯度消失

一般配合逐层归一化使用

### 7.3.2 基于方差缩放的参数初始化
方差缩放（Variance Scaling）：为避免梯度爆炸或消失，应保持每个神经元输入和方差一致，根据连接数量自适应调整初始化分布的方差

#### Xavier 初始化
Xavier 初始化：根据每层神经元数量自动计算初始化参数方差。


l 层神经单元输出：
$$
a^{(l)}=f\left(\sum\_{i=1}^{M\_{l-1}} w\_{i}^{(l)} a\_{i}^{(l-1)}\right)
$$

假设激活函数为线性恒等函数，则均值：
$$
\mathbb{E}\left[a^{(l)}\right]=\mathbb{E}\left[\sum\_{i=1}^{M\_{l-1}} w\_{i}^{(l)} a\_{i}^{(l-1)}\right]=\sum\_{i=1}^{M\_{l-1}} \mathbb{E}\left[w\_{i}^{(l)}\right] \mathbb{E}\left[a\_{i}^{(l-1)}\right]=0
$$

方差：
$$
\begin{aligned}
\operatorname{var}\left(a^{(l)}\right) &=\operatorname{var}\left(\sum\_{i=1}^{M\_{l-1}} w\_{i}^{(l)} a\_{i}^{(l-1)}\right) \\\\
&=\sum\_{i=1}^{M\_{l-1}} \operatorname{var}\left(w\_{i}^{(l)}\right) \operatorname{var}\left(a\_{i}^{(l-1)}\right) \\\\
&=M\_{l-1} \operatorname{var}\left(w\_{i}^{(l)}\right) \operatorname{var}\left(a\_{i}^{(l-1)}\right) .
\end{aligned}
$$

所以应设置：
$$
\operatorname{var}\left(w\_{i}^{(l)}\right)=\frac{1}{M\_{l-1}}
$$

同理，反向传播应设置：
$$
\operatorname{var}\left(w\_{i}^{(l)}\right)=\frac{1}{M\_{l}}
$$

折中设置：
$$
\operatorname{var}\left(w\_{i}^{(l)}\right)=\frac{2}{M\_{l-1}+M\_{l}}
$$

对 $[-r, r]$ 均匀分布初始化则：

$$
r=\sqrt{\frac{6}{M\_{l-1}+M\_{l}}}
$$

Xavier 初始化适用于 Logistic 和 Tanh 激活函数，因为输入往往处在激活函数线性区间。其中 Logistic 函数线性区间斜率约为 0.25，所以初始化方差为 $16 \times \frac{2}{M\_{l-1}+M\_{l}}$

#### He 初始化
对 ReLU 激活函数，通常一半神经元输出为 0， 因此输出方差也近似为恒等函数的一半

考虑前向传播，假设神经元输出：
$$
z\_i^{(l)}=\sum\_{i=1}^{M\_{l-1}} w\_{i}^{(l)} a\_{i}^{(l-1)}
$$

$$
a\_{i}^{l}=\operatorname{ReLU}(z\_i^{l})
$$

两个独立随机变量的方差：
$$
\begin{aligned}
\operatorname{Var}(X Y) &=E\left((X Y)^{2}\right)-(E(X Y))^{2} \\\\
&=E\left(X^{2}\right) E\left(Y^{2}\right)-(E(X) E(Y))^{2} \\\\
&=\left(\operatorname{Var}(X)+(E(X))^{2}\right)\left(\operatorname{Var}(Y)+(E(Y))^{2}\right)-(E(X))^{2}(E(Y))^{2} \\\\
&=\operatorname{Var}(X) \operatorname{Var}(Y)+(E(X))^{2} \operatorname{Var}(Y)+\operatorname{Var}(X)(E(Y))^{2}
\end{aligned}
$$

又：
$$
\begin{aligned}
\operatorname{var}(z) &=\int\_{-\infty}^{+\infty}(z-0)^{2} p(z) d z \\\\
&=2 \int\_{0}^{+\infty} z^{2} p(z) d z \\\\
&=2 E\left(\max (0, z)^{2}\right) \\\\
&=2 E\left(a^2\right)
\end{aligned}
$$

则 ReLU 输出方差：
$$
\begin{aligned}
\operatorname{var}\left(z^{(l)}\right) &=\operatorname{var}\left(\sum\_{i=1}^{M\_{l-1}} w\_{i}^{(l)} a\_{i}^{(l-1)}\right) \\\\
&=\sum\_{i=1}^{M\_{l-1}} \operatorname{var}\left(w\_{i}^{(l)} a\_{i}^{(l-1)}\right) \\\\
&=M\_{l-1} (\operatorname{var}(w\_{i}^{(l)}) \operatorname{var}(a\_{i}^{(l-1)})+E(w\_{i}^{(l)})^{2} \operatorname{var}(a\_{i}^{(l-1)})+\operatorname{var}(w\_{i}^{(l)}) E(a\_{i}^{(l-1)})^{2}) \\\\
&=M\_{l-1} (\operatorname{var}(w\_{i}^{(l)}) \operatorname{var}(a\_{i}^{(l-1)})+\operatorname{var}(w\_{i}^{(l)}) E(a\_{i}^{(l-1)})^{2}) \\\\
&=M\_{l-1} \operatorname{var}\left(w\_{i}^{(l)}\right) E((a\_{i}^{(l-1)})^2) \\\\
&=\frac{1}{2}M\_{l-1} \operatorname{var}\left(w\_{i}^{(l)}\right) \operatorname{var}(z\_{i}^{(l-1)}) \\\\
\end{aligned}
$$

所以：
$$
\operatorname{var}\left(w\_{i}^{(l)}\right)=\frac{2}{M\_{l-1}}
$$

![7fbb7a78dbe31bfbc29495c9da23136a.png](/blog/_resources/fc76e7fdd8eb43fe873c92114fec086a.png)

### 7.3.3 正交初始化
假设一个𝐿 层的等宽线性网络（激活函数为恒等函数）为
$$
\boldsymbol{y}=\boldsymbol{W}^{(L)} \boldsymbol{W}^{(L-1)} \ldots \boldsymbol{W}^{(1)} \boldsymbol{x}
$$

误差项反向传播公式：
$$
\delta^{(l-1)}=\left(\boldsymbol{W}^{(l)}\right)^{\top} \delta^{(l)}
$$

为避免梯度消失或梯度爆炸，希望在误差项反向传播中具有范数保持性（Norm-Perserving）：
$$
\left\|\delta^{(l-1)}\right\|\_{2}=\left\|\delta^{(l)}\right\|\_{2}=\left\|\left(\boldsymbol{W}^{(l)}\right)^{\top} \delta^{(l)}\right\|\_{2}
$$

二范数定义：
$$
\|A\|\_{2}=\sqrt{\lambda\_{\max }\left(A^{*} A\right)}
$$

可证明矩阵与正交矩阵相乘，二范数（谱范数）不变。

所以，可以将 $\boldsymbol{W}^{(l)}$ 初始化为正交矩阵，即正交初始化（Orthogonal Initialization）：
1. 用 $N(0,1)$ 初始化一个矩阵
2. 对该矩阵奇异值分解，得到两个正交矩阵，使用其中一个作为权重矩阵

正交初始化常用于 RNN 循环边的权重矩阵。

对非线性神经网络，需要对正交矩阵乘以一个缩放系数。


## 7.4 数据预处理
尺度不变性（Scale Invariance）：机器学习算法在缩放特征后不影响学习和预测。

eg. 线性分类器尺度不变、KNN尺度敏感

对**尺度敏感**模型，需要对样本预处理：统一特征尺度、消除特征相关性。

理论上神经网络具有尺度不变性，通过参数调整适应尺度，但尺度不同会增加训练难度：
1. 为防止 tanh 等函数进入饱和区而梯度消失，对每个特征尺度需要进行特定初始化
2. 梯度下降方向不指向最优解

![3e95e86b27073a63b8cfc93e64e3048f.png](/blog/_resources/a30049ff2f364a1eb9d3867e587dfa32.png)

**归一化（Normalization**）：泛指同一数据特征尺度的方法。
1. 最大最小值归一化（Min-Max Normalization）

$$
\hat{x}^{(n)}=\frac{x^{(n)}-\min \_{n}\left(x^{(n)}\right)}{\max \_{n}\left(x^{(n)}\right)-\min \_{n}\left(x^{(n)}\right)}
$$

2. 标准化（Standardizatin）/ Z值归一化（Z-Score Normalization）

$$
\hat{x}^{(n)}=\frac{x^{(n)}-\mu}{\sigma}
$$

标准差为0则说明该维特征没有区分性，可直接删掉。

3. 白化（Whitening）：降低输入数据之间的冗余性，并使所有特征具有相同方差。
	- 主要实现方式：主成分分析（Principal Component Analysis，PCA）去掉各成分相关性。

![71831e4516edb51c09617331e212eb08.png](/blog/_resources/d3bed5f37b2e4b728c9b18cc9ec01930.png)


## 7.5 逐层归一化
逐层归一化（Layer-wise Normalization）提高效率的原因：
1. 更好的尺度不变性
	- > 内部协变量偏移（Internal Covariate Shift）：神经层输入分布改变后参数需要重新学习
2. 更平滑的优化地形：
	- 大部分神经元处于不饱和区，避免梯度消失； 
	- 优化地形（Optimization Landscape）更加平滑

常用的逐层归一化方法：
- 批量归一化
- 层归一化
- 权重归一化
- 局部相应归一化

### 7.5.1 批量归一化
批量归一化（Batch Normalization，BN）：在仿射变换之后、激活函数之前，将输入$\boldsymbol{z}^{(l)}$ 每一维都归一到标准正态分布：

$$
\hat{z}^{(l)}=\frac{z^{(l)}-\mathbb{E}\left[z^{(l)}\right]}{\sqrt{\operatorname{var}\left(z^{(l)}\right)+\epsilon}}
$$

这里的期望和方差一般使用小批量样本集的均值和方差进行估计。

归一到 0 附近在使用 Sigmoid 函数时，取值在线性区间，削弱了神经网络的非线性性质，可以通过缩放平移来改变取值区间：

$$
\begin{aligned}
\hat{\boldsymbol{z}}^{(l)} &=\frac{\boldsymbol{z}^{(l)}-\mu\_{\mathcal{B}}}{\sqrt{\sigma\_{\mathcal{B}}^{2}+\epsilon}} \odot \boldsymbol{\gamma}+\boldsymbol{\beta} \\\\
& \triangleq \mathrm{B} \mathrm{N}\_{\gamma, \boldsymbol{\beta}}\left(\boldsymbol{z}^{(l)}\right)
\end{aligned}
$$

其中：
$$
\begin{aligned}
\mu\_{\mathcal{B}} &=\frac{1}{K} \sum\_{k=1}^{K} z^{(k, l)}, \\\\
\sigma\_{\mathcal{B}}^{2} &=\frac{1}{K} \sum\_{k=1}^{K}\left(\boldsymbol{z}^{(k, l)}-\mu\_{\mathcal{B}}\right) \odot\left(\boldsymbol{z}^{(k, l)}-\mu\_{\mathcal{B}}\right)
\end{aligned}
$$

批量归一化操作可以看作一个特殊的神经层，加在每一层非线性激活函数之前：
$$
\boldsymbol{a}^{(l)}=f\left(\mathrm{BN}\_{\gamma, \beta}\left(\boldsymbol{z}^{(l)}\right)\right)=f\left(\mathrm{BN}\_{\gamma, \beta}\left(\boldsymbol{W} \boldsymbol{a}^{(l-1)}\right)\right)
$$

批量归一化不但提高优化效率，还是一种隐形的正则化方法：对样本的预测与批次中其他样本有关，不会过拟合某个特定样本。


### 7.5.2 层归一化
RNN 等的神经元输入分布是动态变化的，无法应用 BN

层归一化（Layer Normalization）：对中间一层的所有神经元进行归一化

$$
\begin{aligned}
\hat{z}^{(l)} &=\frac{z^{(l)}-\mu^{(l)}}{\sqrt{\sigma^{(l)^{2}+\epsilon}} \odot \gamma+\beta} \\\\
& \triangleq \mathrm{LN}\_{\gamma, \beta}\left(z^{(l)}\right)
\end{aligned}
$$

其中：
$$
\begin{aligned}
\mu\_{\mathcal{B}} &=\frac{1}{K} \sum\_{k=1}^{K} z^{(k, l)}, \\\\
\sigma\_{\mathcal{B}}^{2} &=\frac{1}{K} \sum\_{k=1}^{K}\left(\boldsymbol{z}^{(k, l)}-\mu\_{\mathcal{B}}\right) \odot\left(\boldsymbol{z}^{(k, l)}-\mu\_{\mathcal{B}}\right)
\end{aligned}
$$

RNN 的层归一化：
$$
\begin{aligned}
&\boldsymbol{z}\_{t}=\boldsymbol{U} \boldsymbol{h}\_{t-1}+\boldsymbol{W} \boldsymbol{x}\_{t} \\\\
&\boldsymbol{h}\_{t}=f\left(\mathrm{LN}\_{\gamma, \beta}\left(\boldsymbol{z}\_{t}\right)\right)
\end{aligned}
$$

层归一化和批量归一化区别：
对于 𝐾 个样本的一个小批量集合 $\boldsymbol{Z}^{(l)}=\left[\boldsymbol{z}^{(1, l)} ; \cdots ; \boldsymbol{z}^{(K, l)}\right]$

- 层归一化：对 $\boldsymbol{Z}^{(l)}$ 每一列进行归一化
- 批量归一化：对 $\boldsymbol{Z}^{(l)}$ 每一行进行归一化

一般批量归一化更好，当 Batch Size 比较小时，可以选择层归一化。

### 7.5.3 权重归一化
权重归一化（Weight Normalization）：通过再参数化（Reparameterization），将连接权重分解为长度和方向两种参数：
$$
\boldsymbol{a}^{(l)}=f\left(\boldsymbol{W} \boldsymbol{a}^{(l-1)}+\boldsymbol{b}\right)
$$

则再参数化 $\boldsymbol{W}$：

$$
\boldsymbol{W}\_{i,:}=\frac{g\_{i}}{\left\|\boldsymbol{v}\_{i}\right\|} \boldsymbol{v}\_{i}, \quad 1 \leq i \leq M\_{l}
$$

### 7.5.4 局部响应归一化
局部响应归一化（Local Response Normalization，LRN），常用于卷积层，

$$
\begin{aligned}
\hat{\boldsymbol{Y}}^{p} &=\boldsymbol{Y}^{p} /\left(k+\alpha \sum\_{j=\max \left(1, p-\frac{n}{2}\right)}^{\min \left(P, p+\frac{n}{2}\right)}\left(\boldsymbol{Y}^{j}\right)^{2}\right)^{\beta} \\\\
& \triangleq \mathrm{LRN}\_{n, k, \alpha, \beta}\left(\boldsymbol{Y}^{p}\right)
\end{aligned}
$$

类似：生物神经元中的侧抑制（lateral inhibition），活跃神经元对相邻神经元有抑制作用。最大汇聚（Max Pooling）也具有侧抑制作用，区别（抑制维度不同）：
1. 最大汇聚：对同一特征映射中邻近神经元抑制
2. 局部响应归一化：对同一位置邻近特征映射的神经元抑制


## 7.6 超参数优化
常见超参数：
1. 网络结构，包括神经元之间的连接关系、层数、每层的神经元数量、激
活函数的类型等．
2. 优化参数，包括优化方法、学习率、小批量的样本数量等．
3. 正则化系数

超参数优化（Hyperparameter Optimization）困难：
1. 是组合优化问题，没有通用有效办法
2. 评估一组超参数配置（Configuration）时间代价高，一些优化方法难以应用（如演化算法（Evolution Algorithm））

主要优化方法：

### 7.6.1 网格搜索
网格搜索（Grid Search）：遍历所有超参数组合，在val集上选择性能最好的配置。

### 7.6.2 随机搜索
随机搜索（Random Search）：有些参数影响较小，用网格搜成本高，所以对超参数进行随机组合，然后选最好的配置。

网格搜索和随机搜索都没有考虑不同参数组合之间的相关性，所以提出自适应的超参数优化方法：
1. 贝叶斯优化
2. 动态资源分配

### 7.6.3 贝叶斯优化
贝叶斯优化（Bayesian optimization）：根据已实验的超参组合，预测下一个可能最大收益的组合。

常用：时序模型优化（Sequential Model-Based Optimization，SMBO）

![dfc8b96cdffa70d039835fabb06e2e6d.png](/blog/_resources/804c762c47f649789aabeb3aad54293b.png)

### 7.6.4 动态资源分配
较早估计出一组配置的效果会比较差，就可以中止这组配置的评估，关键是将有限资源分配给更有可能带来收益的组合。
- 如：早期停止（Early-Stopping）终止对不收敛或收敛较差的配置的训练。

最优臂问题（Best-Arm Problem）：即在给定有限的机会次数下，如何玩这些赌博机并找到收益最大的臂．

![75c347356e3f49dc7483f4f232e797bb.png](/blog/_resources/70e0fa5eb79d4e6485dcc9989994eb62.png)

### 7.6.5 神经架构调整
可以认为，深度学习使机器学习中的“特征工程”问题转变为“网络架构工程”问题。

神经架构搜索（Neural Architecture Search，NAS）[Zoph et al., 2017]：利用元学习的思想，神经架构搜索利用一个控制器来生成另一个子网络的架构描述，控制器可以用 RL 训练。


## 7.7 网络正则化
正则化（Regularization）是一类通过限制模型复杂度，从而避免过拟合，提高泛化能力的方法，比如引入约束、增加先验、提前停止等

### 7.7.1 l1 和 l2 正则化
$$
\theta^{*}=\underset{\theta}{\arg \min } \frac{1}{N} \sum\_{n=1}^{N} \mathcal{L}\left(y^{(n)}, f\left(\boldsymbol{x}^{(n)} ; \theta\right)\right)+\lambda \ell\_{p}(\theta)
$$

则优化问题：
$$
\begin{aligned}
\theta^{*}=& \underset{\theta}{\arg \min } \frac{1}{N} \sum\_{n=1}^{N} \mathcal{L}\left(y^{(n)}, f\left(\boldsymbol{x}^{(n)} ; \theta\right)\right), \\\\
\text { s.t. } \quad \ell\_{p}(\theta) \leq 1
\end{aligned}
$$

![365ccc2e51808bd62d25f56584f82a95.png](/blog/_resources/1ce45d3f758e405d8551f8737ec9129c.png)

弹性网络正则化（Elastic Net Regularization）：同时加入 l1 和 l2 优化。

### 7.7.2 权重衰减
权重衰减（Weight Decay）：

$$
\theta\_{t} \leftarrow(1-\beta) \theta\_{t-1}-\alpha \mathrm{g}\_{t}
$$

在 SGD 中，权重衰减与 l2 效果相同，在 Adam 等较复杂优化中，则不等价。

### 7.7.3 提前停止
提前停止（Early Stop）：验证集错误不再下降则停止。


### 7.7.4 丢弃法 Dropout
丢弃法（Dropout Method）[Srivastava et al., 2014]：训练时随机丢弃一部分神经元。

$$
\operatorname{mask}(\boldsymbol{x})= \begin{cases}\boldsymbol{m} \odot \boldsymbol{x} & \text { 当训练阶段时 } \\\\ p \boldsymbol{x} & \text { 当测试阶段时 }\end{cases}
$$

- 对于隐藏层神经单元，保留率 p 取 0.5 效果最好，随机生成的网络结构最具多样性
- 对于输入层神经单元，通常保留率 p 更接近 1，使输入变化不会太大

![2aaae000c873002e4d3eaa6d2131052c.png](/blog/_resources/fde6487e62314fcdba5e5f1c41967927.png)

集成学习角度的解释：假设共 n 个神经元，则 dropout 出了 $2^n$ 个子网络，每次迭代相当于训练不同的子网络，最终结果可以看作指数个模型集成。

贝叶斯学习角度的解释：dropout 可以看作一种贝叶斯学习的近似，即对要学习的网络多次采用后平均的结果：
$$
\begin{aligned}
\mathbb{E}\_{q(\theta)}[y] &=\int\_{q} f(\boldsymbol{x} ; \theta) q(\theta) d \theta \\\\
& \approx \frac{1}{M} \sum\_{m=1}^{M} f\left(\boldsymbol{x}, \theta\_{m}\right)
\end{aligned}
$$

**RNN 上的 Dropout**
为避免损害时间维度上的记忆能力，不能对每个时刻的隐状态进行随机丢弃：

1. Naive Dropout：可以对**非时间维度**的连接进行 Dropout：

![44bad15230f86345f977ccb4cb16ee47.png](/blog/_resources/f4b553b16b394b9cbdc11d6b7d4fb959.png)

2. 变分丢弃法（Variational Dropout）：根据贝叶斯学习，每次 dropout 采样的参数在各个时间应该不变，所有时刻应该使用相同的掩码：

![c08e661b881d6151eab9c9035def9ba8.png](/blog/_resources/e339e7325b8e4723b7972703049af9b3.png)

### 7.7.5 数据增强
数据增强（Data Augmentation）：目前主要在图像上使用
1. 旋转（Rotation）
2. 翻转（Flip）
3. 缩放（Zoom In/Out）
4. 平移（Shift）
5. 加噪声（Noise）

### 7.7.6 标签平滑
laebl smoothing：在输出标签中添加随机噪声来避免过拟合。

One-hot 的标签是硬目标（Hard Target）：
$$
\boldsymbol{y}=[0, \cdots, 0,1,0, \cdots, 0]^{\top} .
$$

Motivation：
1. 在 softmax 中，使某类概率趋向于 1 需要很大的归一化得分，可能导致其权重越来越大，并导致过拟合
2. 标签错误时，会导致更加严重的过拟合

平滑后为软目标（Soft Target）：
$$
\tilde{\boldsymbol{y}}=\left[\frac{\epsilon}{K-1}, \cdots, \frac{\epsilon}{K-1}, 1-\epsilon, \frac{\epsilon}{K-1}, \cdots, \frac{\epsilon}{K-1}\right]^{\top}
$$

这种标签平滑没有考虑标签之间的相关性，更好的办法是按照类别相关性赋予其他标签不同概率，如教师网络（Teacher Network）的输出作为软目标训练学生网络（Student Network），即知识蒸馏（Knowledge Distillation）


## 习题选做
#### 习题 7-1 在小批量梯度下降中，试分析为什么学习率要和批量大小成正比
理论上，Batch Size 增大 k 倍，LR 应该增大 $\sqrt{k}$ 使梯度保持不变，但实践发现 k 倍效果更好。

#### 习题 7-5 证明公式(7.45)．
He 初始化证明我在上文已给出。

#### 习题 7-8 分析为什么批量归一化不能直接应用于循环神经网络
BN 不适用于 RNN 这种动态结构，对 Batch 中每个 postion 作标准化，需要估计每个 position 的 $\mu$ 和 $\sigma$：
1. 样本长度不同，测试集中过长时间片的参数难以估计
2. Normalize 的对象来自不同的分布，多个 sequence 的同一个 position 很难服从相同分布

#### 习题 7-10 试分析为什么不能在循环神经网络中的循环连接上直接应用丢弃法？
对隐状态随机丢弃，会损失记忆的信息，可以只对非时间维度的参数进行丢弃或对时间维度丢弃相同的参数。


