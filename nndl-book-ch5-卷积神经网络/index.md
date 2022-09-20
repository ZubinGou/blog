# 《神经网络与深度学习》第5章 - 卷积神经网络


卷积神经网络（Convolutional Neural Network，CNN 或 ConvNet）：一种具有局部连接、权重共享等特性的深层前馈神经网络

FC处理图像问题：
1. 参数过多
2. 局部不变性

启发：感受野（Receptive Filed），视觉神经元只接收视网膜特定区域的信号。

CNN一般组成：卷积层 + 汇聚层 + 全连接层，交叉堆叠

CNN结构特性：局部连接、权重共享、汇聚
- 一定程度的平移、缩放和旋转不变性（？）


## 5.1 卷积
### 5.1.1 卷积的定义
#### 一维卷积

长度K的滤波器与信号序列$x_1,x_2,...$的卷积：
$$
y\_{t}=\sum\_{k=1}^{K} w\_{k} x\_{t-k+1}
$$
- 滤波器（Filter）或卷积核（Convolution Kernel）：$w\_{1}, w\_{2}, \cdots$
- 简化：$y=w * x$
- $\boldsymbol{w}=[1 / K, \cdots, 1 / K]$时，卷积相当于简单移动平均
- $\boldsymbol{w}=[1,-2,1]$时，近似信号序列的二阶微分
$$
x^{\prime \prime}(t)=x(t+1)+x(t-1)-2 x(t)
$$
![42392b24f2b3bfe5966f35ec1078d264.png](../../_resources/cccb0297140349709aa69b71464c4e04.png)
- 上图卷积分别检测信号序列的低频和高频信息

#### 二维卷积
给定图像 $\boldsymbol{X} \in \mathbb{R}^{M \times N}$ 和一个滤波器 $\boldsymbol{W} \in \mathbb{R}^{U \times V}$，卷积为：
$$
y\_{i j}=\sum\_{u=1}^{U} \sum\_{v=1}^{V} w\_{u v} x\_{i-u+1, j-v+1}
$$
- 简化：$\boldsymbol{Y}=\boldsymbol{W} * \boldsymbol{X}$
- 均值滤波（Mean Filter）即是一种二维卷积

![9d165e65f8c62b38985d0315ffa315e4.png](../../_resources/cedc8d2749484c6d8e2e683780776bdf.png)

![d42af1e0a6c946685e264d2cdbe0a736.png](../../_resources/d7b3859aa0d14db0930e5eca4bdf398a.png)
- 第一个为高斯滤波器，后俩用来提取边缘特征

### 5.1.2 互相关
互相关（Cross-Correlation），衡量两个序列相关性的函数，也称为不翻转卷积（相比卷积，卷积核不同翻转）：
$$
y\_{i j}=\sum\_{u=1}^{U} \sum\_{v=1}^{V} w\_{u v} x\_{i+u-1, j+v-1}
$$
即：
$$
\begin{aligned}
\boldsymbol{Y} &=\boldsymbol{W} \otimes \boldsymbol{X} \\\\
&=\operatorname{rot} 180(\boldsymbol{W}) * \boldsymbol{X}
\end{aligned}
$$

卷积核可学习时，卷积和互相关能力等价，为了实现方便，一般用互相关代替卷积。

### 5.1.3 卷积的变种
步长（Stride）：卷积核滑动的时间间隔
零填充（Zero Padding）：输入向量两端补零
![49ead6aa388852c7484a5d93b64da0a9.png](../../_resources/69c49ec1b0d1492c9142f5b31cb66b08.png)


设卷积大小为K，常用卷积（步长S均为1）：
1. 窄卷积（Narrow Convolution）：P=1
2. 宽卷积（Wide Convolution）：P=K-1
3. 等宽卷积（Equal-Width Convolution）：P=(K-1)/2

### 5.1.4 卷积的数学性质
1. 交换性
- 不限制长度的两个卷积信号的卷积具有交换性：
$$y=y * x$$
- 固定长度信息和卷积核的宽卷积也具有交换性（？证明）：
$$\operatorname{rot} 180(\boldsymbol{W}) \tilde{\otimes} \boldsymbol{X}=\operatorname{rot} 180(\boldsymbol{X}) \tilde{\otimes} \boldsymbol{W}$$

2. 导数
假设 $\boldsymbol{Y}=\boldsymbol{W} \otimes \boldsymbol{X}$，其中 $\boldsymbol{X} \in \mathbb{R}^{M \times N}, \boldsymbol{W} \in \mathbb{R}^{U \times V}, \boldsymbol{Y} \in \mathbb{R}^{(M-U+1) \times(N-V+1)}$，函数  $f(\boldsymbol{Y}) \in \mathbb{R}$ 为一个标量函数, 则：
$$
\begin{aligned}
\frac{\partial f(\boldsymbol{Y})}{\partial w\_{u v}} &=\sum\_{i=1}^{M-U+1 N-V+1} \frac{\partial y\_{i j}}{\partial w\_{u v}} \frac{\partial f(\boldsymbol{Y})}{\partial y\_{i j}} \\\\
&=\sum\_{i=1}^{M-U+1} \sum\_{j=1}^{N-V+1} x\_{i+u-1, j+v-1} \frac{\partial f(\boldsymbol{Y})}{\partial y\_{i j}} \\\\
&=\sum\_{i=1}^{M-U+1} \sum\_{j=1}^{N-V+1} \frac{\partial f(\boldsymbol{Y})}{\partial y\_{i j}} x\_{u+i-1, v+j-1}
\end{aligned}
$$

即$f(Y)$关于$W$的偏导数为$X$和$\frac{\partial f(\boldsymbol{Y})}{\partial \boldsymbol{Y}}$的卷积（互相关）：
$$
\frac{\partial f(\boldsymbol{Y})}{\partial \boldsymbol{W}}=\frac{\partial f(\boldsymbol{Y})}{\partial \boldsymbol{Y}} \otimes \boldsymbol{X}
$$

同理：
$$
\begin{aligned}
\frac{\partial f(\boldsymbol{Y})}{\partial x\_{s t}} &=\sum\_{i=1}^{M-U+1} \sum\_{j=1}^{N-V+1} \frac{\partial y\_{i j}}{\partial x\_{s t}} \frac{\partial f(\boldsymbol{Y})}{\partial y\_{i j}} \\\\
&=\sum\_{i=1}^{M-U+1 N-V+1} \sum\_{j=1}^{M} w\_{S-i+1, t-j+1} \frac{\partial f(\boldsymbol{Y})}{\partial y\_{i j}}
\end{aligned}
$$

即$f(Y)$关于$X$的偏导数为$W$和$\frac{\partial f(\boldsymbol{Y})}{\partial \boldsymbol{Y}}$的宽卷积（互相关）：
$$
\begin{aligned}
\frac{\partial f(\boldsymbol{Y})}{\partial \boldsymbol{X}} &=\operatorname{rot} 180\left(\frac{\partial f(\boldsymbol{Y})}{\partial \boldsymbol{Y}}\right) \tilde{\otimes} \boldsymbol{W} \\\\
&=\operatorname{rot} 180(\boldsymbol{W}) \tilde{\otimes} \frac{\partial f(\boldsymbol{Y})}{\partial \boldsymbol{Y}}
\end{aligned}
$$


## 5.2 卷积神经网络
### 5.2.1 卷积代替全连接
$$
\boldsymbol{z}^{(l)}=\boldsymbol{w}^{(l)} \otimes \boldsymbol{a}^{(l-1)}+b^{(l)}
$$

卷积层性质：
- 局部连接
- 权重共享：可以理解为一个卷积核捕捉一种局部特征

![b362404d149f1a523d826c16fdc86a29.png](../../_resources/3f67c0bd3c0a4977b0a9b74feebe5092.png)

卷积层参数只有K维权重$w^{(l)}$和一维偏置$b^{(l)}$
神经元数量满足（默认步长1，无零填充）：
$$
\text { } M\_{l}=M\_{l-1}-K+1
$$

### 5.2.2 卷积层
特征映射（Feature Map）：一幅图像（或其他特征映射）在经过卷积提取到的特征，每个特征映射可以作为一类抽取的图像特征
![6c90f149f197867324e0a040dee5ff21.png](../../_resources/8c84ce55a7ac4cf78d5764c21be0799f.png)

$$
X \in \mathbb{R}^{M \times N \times D}
$$
$$
y \in \mathbb{R}^{M^{\prime} \times N^{\prime} \times P}
$$
$$
\mathcal{W} \in \mathbb{R}^{U \times V \times P \times D}
$$
$$
\begin{array}{l}
\boldsymbol{Z}^{p}=\boldsymbol{W}^{p} \otimes \boldsymbol{X}+b^{p}=\sum\_{d=1}^{D} \boldsymbol{W}^{p, d} \otimes \boldsymbol{X}^{d}+b^{p} \\\\
\boldsymbol{Y}^{p}=f\left(\boldsymbol{Z}^{p}\right)
\end{array}
$$

![539df3c8e2119c3438127509404ab4fd.png](../../_resources/2b93459095c741919e90314518612270.png)

参数个数：$P \times D \times(U \times V)+P$

### 5.2.3 汇聚层
汇聚层（Pooling Layer）也叫子采样层（Subsampling Layer）：促进特征选择，降低特征/参数数量，避免过拟合

汇聚（Pooling）是指对每个区域进行下采样（Down Sampling）得到一个值，作为这个区域的概括

常用汇聚函数：
1. 最大汇聚（Maximum Pooling/ Max Pooling）：$y\_{m, n}^{d}=\max \_{i \in R\_{m, n}^{d}} x\_{i}$
2. 平均汇聚（Mean Pooling）：$y\_{m, n}^{d}=\frac{1}{\left|R\_{m, n}^{d}\right|} \sum\_{i \in R\_{m, n}^{d}} x\_{i}$

![56d213933845758d9f138d631f4c58a2.png](../../_resources/0f26c6a440c44d15a4e7b6a4ec57de7d.png)

汇聚层可以看作特殊的卷积层：卷积核大小为$K\times K$，步长为$S\times S$，卷积核为max函数或mean函数

### 5.2.4 卷积网络的整体结构
一个典型的卷积网络是由卷积层、汇聚层、全连接层交叉堆叠而成

![52be94f6d0a339e8782a8c5263e84833.png](../../_resources/073f38173b2b4f2687d04323a8529e4e.png)
（N为1~100或更大，K一般为0~2）

目前，趋向于使用小卷积核、深结构、少汇聚层的全卷积网络


## 5.3 参数学习
CNN参数只有卷积核和偏置，对于：
$$
\boldsymbol{Z}^{(l, p)}=\sum\_{d=1}^{D} \boldsymbol{W}^{(l, p, d)} \otimes \boldsymbol{X}^{(l-1, d)}+b^{(l, p)}
$$

偏导：
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}^{(l, p, d)}} &=\frac{\partial \mathcal{L}}{\partial \boldsymbol{Z}^{(l, p)}} \otimes \boldsymbol{X}^{(l-1, d)} \\\\
&=\delta^{(l, p)} \otimes \boldsymbol{X}^{(l-1, d)}
\end{aligned}
$$
$$
\frac{\partial \mathcal{L}}{\partial b^{(l, p)}}=\sum\_{i, j}\left[\delta^{(l, p)}\right]\_{i, j}
$$

### 5.3.1 卷积神经网络的反向传播算法
#### 汇聚层
上采样与l层特征映射的激活值偏导数逐元素相乘：
$$
\begin{aligned}
\delta^{(l, p)} & \triangleq \frac{\partial \mathcal{L}}{\partial Z^{(l, p)}} \\\\
&=\frac{\partial X^{(l, p)}}{\partial Z^{(l, p)}} \frac{\partial Z^{(l+1, p)}}{\partial \boldsymbol{X}^{(l, p)}} \frac{\partial \mathcal{L}}{\partial \boldsymbol{Z}^{(l+1, p)}} \\\\
&=f\_{l}^{\prime}\left(\boldsymbol{Z}^{(l, p)}\right) \odot \operatorname{up}\left(\delta^{(l+1, p)}\right)
\end{aligned}
$$

#### 卷积层
$$
\begin{aligned}
\delta^{(l, p)} & \triangleq \frac{\partial \mathcal{L}}{\partial Z^{(l, p)}} \\\\
&=\frac{\partial X^{(l, p)}}{\partial Z^{(l, p)}} \frac{\partial Z^{(l+1, p)}}{\partial \boldsymbol{X}^{(l, p)}} \frac{\partial \mathcal{L}}{\partial \boldsymbol{Z}^{(l+1, p)}} \\\\
&=f\_{l}^{\prime}\left(\boldsymbol{Z}^{(l, p)}\right) \odot \operatorname{up}\left(\delta^{(l+1, p)}\right)
\end{aligned}
$$


## 5.4 几种典型的CNN
### 5.4.1 LeNet-5
LeNet-5[LeCun et al., 1998] 

![49c9858119839c413f3c8ccb2d069962.png](../../_resources/1bc54a1bb40d4c0c84a4a80d78d7a5b9.png)

### 5.4.2 AlexNet
AlexNet[Krizhevsky et al., 2012]是第一个现代深度的CNN
- 2012 年ImageNet 图像分类冠军

![bd684df33fe25b753c520ff8d2080023.png](../../_resources/71a1560d5ba5426ba28b6b4672cf0bc9.png)

### 5.4.3 Inception网络
Inception 网络是由有多个 Inception 模块和少量的汇聚层堆叠而成

![81efad8436c1a73921b2cbf9cbb7ece9.png](../../_resources/72b4839baca24cacb3b484e7ee6a6f17.png)
Inception v1：即GoogLeNet[Szegedy et al., 2015]
- 2014 年 ImageNet 图像分类冠军

![523e2cd40a48759d78e21d88cb9b4624.png](../../_resources/a67bed52c7de4f489668a8d219f674b2.png)

Inception v3 网络：用多层的小卷积核来替换大的卷积核

### 残差网络
残差网络（Residual Network，ResNet）：通过给非线性的卷积层增加直连边（Shortcut Connection）（也称为残差连接 Residual Connection））的方式来提高信息的传播效率

将目标函数拆分为两部分：恒等函数（Identity Function）和残差函数（Residue Function）：
$$
h(\boldsymbol{x})=\underbrace{\boldsymbol{x}}\_{\text {恒等函数 }}+\underbrace{(h(\boldsymbol{x})-\boldsymbol{x})}\_{\text {残差函数 }}
$$

![af4e41a34c3babe2823d011c9c969e9e.png](../../_resources/7355f8f513b34545882c87b384d1e8a4.png)


## 5.5 其他卷积方式

### 5.5.1 转置卷积
$$
\begin{aligned}
z &=w \otimes x \\\\
&=\left[\begin{array}{lllll}
w\_{1} & w\_{2} & w\_{3} & 0 & 0 \\\\
0 & w\_{1} & w\_{2} & w\_{3} & 0 \\\\
0 & 0 & w\_{1} & w\_{2} & w\_{3}
\end{array}\right] \boldsymbol{x} \\\\
&=\boldsymbol{C} \boldsymbol{x}
\end{aligned}
$$

反过来，仿射矩阵的转置：
$$
\begin{aligned}
\boldsymbol{x} &=\boldsymbol{C}^{\top} \boldsymbol{z} \\\\
&=\left[\begin{array}{lll}
w\_{1} & 0 & 0 \\\\
w\_{2} & w\_{1} & 0 \\\\
w\_{3} & w\_{2} & w\_{1} \\\\
0 & w\_{3} & w\_{2} \\\\
0 & 0 & w\_{3}
\end{array}\right] \boldsymbol{z} \\\\
&=\operatorname{rot} 180(\boldsymbol{w}) \tilde{\otimes} z
\end{aligned}
$$

我们将这种低维特征映射到高维特征的卷积操作成为转置卷积（Transposed Convolution），也称为反卷积（Deconvolution）

卷积层的前向计算和后向传播也是一种转置关系
![e3a97d5124dde886cf6458eb45ddb5d5.png](../../_resources/9b5fe78043854a36a59bd7045bb656bb.png)

微步卷积：步长 $S\lt 1$ 的转置卷积

### 5.5.2 空洞卷积
空洞卷积（Atrous Convolution）：不增加参数，增加了输出单元感受野，也成为膨胀（Dilated Convolution）

![446b49a96d0043ab3741782300a0c351.png](../../_resources/a32ab208685b436bbfc4f4d10f4b46a4.png)

> 空洞卷积中选择D为半径的圆上的点，感受野是不是更加合理？考虑图像旋转后，方形的点位捕捉特征能力可能受到影响。
> 进而，用圆形的卷积效果是否会更好？运算速度呢？


## 习题
#### 习题 5-1 1）证明公式 (5.6) 可以近似为离散信号序列 𝑥(𝑡) 关于 𝑡 的二阶微分；2）对于二维卷积，设计一种滤波器来近似实现对二维输入信号的二阶微分．

1. 由导数定义：$x'(t)=x(t)-x(t-1)$，进一步求二阶导数即可
2. Laplace operator
$$
\nabla^{2} f=\frac{\partial^{2} f}{\partial x^{2}}+\frac{\partial^{2} f}{\partial y^{2}}
$$
$$
\frac{\partial^{2} f}{\partial x^{2}}=f(x+1, y)+f(x-1, y)-2 f(x, y)
$$
$$
\frac{\partial^{2} f}{\partial y^{2}}=f(x, y+1)+f(x, y-1)-2 f(x, y)
$$
$$
\nabla^{2} f(x, y)=f(x+1, y)+f(x-1, y)+f(x, y+1)+f(x, y-1)-4 f(x, y)
$$

所以：
$$
\mathrm{L} 0=\left[\begin{array}{ccc}
0 & -1 & 0 \\\\
-1 & 4 & -1 \\\\
0 & -1 & 0
\end{array}\right]
$$

#### 习题 5-2 证明宽卷积具有交换性，即公式 (5.13)．
抛开padding，只看有效的重叠部分，因为深度D相同，只考虑平面上观察：本质都是就是遍历了两个方形的所有重合摆放方式，将等式一边的图像均翻转180度，可以想象其遍历顺序完全相同。

#### 习题 5-3 分析卷积神经网络中用 1 × 1 的卷积核的作用．
1x1卷积核，又称为网中网（Network in Network），对于三维输入时是一个正方形长条，可以用来升维/降维、接一个ReLU增加非线性、channal变换/通道信息交互

#### 习题5-4 对于一个输入为100 × 100 × 256的特征映射组，使用3 × 3的卷积核，输出为 100 × 100 × 256 的特征映射组的卷积层，求其时间和空间复杂度．如果引入一个 1 × 1 卷积核，先得到 100 × 100 × 64 的特征映射，再进行 3 × 3 的卷积，得到100 × 100 × 256 的特征映射组，求其时间和空间复杂度．
1. 直接卷积
	- 时间：$100\times 100\times 256\times 256\times 3\times 3$
	- 空间：$100\times 100\times 256$
	- 参数：$(3\times 3 + 1)\times 256 \times 256$
2. 先1x1卷积：
	- 时间：$100\times 100\times 256\times 64\times 1\times 1+100\times 64\times 256\times 3 \times 3$
	- 空间：$100\times 100\times 256+100\times 100\times 64$
	- 参数：$(1\times 1+1)\times 256\times 64+(3\times 3 + 1)\times 64 \times 256$

#### 习题 5-5 对于一个二维卷积，输入为 3 × 3，卷积核大小为 2 × 2，试将卷积操作重写为仿射变换的形式． 参见公式(5.45)．
$$
\left[\begin{array}{lcccccccc}
w\_{11} & w\_{12} & 0 & w\_{21} & w\_{22} & 0 & 0 & 0 & 0 \\\\
0 & w\_{11} & w\_{12} & 0 & w\_{21} & w\_{22} & 0 & 0 & 0 \\\\
0 & 0 & 0 & w\_{11} & w\_{12} & 0 & w\_{21} & w\_{22} & 0 \\\\
0 & 0 & 0 & 0 & w\_{11} & w\_{12} & 0 & w\_{21} & w\_{22}
\end{array}\right]
$$


#### 习题 5-6 计算函数 𝑦 = max(𝑥1, ⋯ , 𝑥𝐷) 和函数 𝑦 = arg max(𝑥1, ⋯ , 𝑥𝐷) 的梯度．
TODO

#### 习题 5-7 忽略激活函数，分析卷积网络中卷积层的前向计算和反向传播（公式(5.39)）是一种转置关系．
前向：
$$
z^{(l+1)}=W^{(l+1)} z^{(l)}
$$

反向：
$$
\delta^{(l)}=\left(W^{(l+1)}\right)^{\top} \delta^{(l+1)}
$$

#### 习题5-8 在空洞卷积中，当卷积核大小为𝐾，膨胀率为𝐷 时，如何设置零填充𝑃 的值以使得卷积为等宽卷积．

$$P=(K-1)/2\times(1+D)$$
