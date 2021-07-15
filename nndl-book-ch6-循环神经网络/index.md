# 《神经网络与深度学习》第6章 - 循环神经网络


# ch6 循环神经网络
学习：随时间反向传播算法

长程依赖：长序列时梯度爆炸和消失 -> 门控机制（Gating Mechanism）

广义记忆网络：递归神经网络、图网络

## 6.1 给网络增强记忆能力
### 6.1.1 延时神经网络
延时神经网络（Time Delay Neural Network，TDNN）：在FNN非输出层都添加一个延时器，记录神经元历史活性值。

l层神经元活性值依赖于l-1层神经元的最近K个时刻的活性值：
$$
\boldsymbol{h}\_{t}^{(l)}=f\left(\boldsymbol{h}\_{t}^{(l-1)}, \boldsymbol{h}\_{t-1}^{(l-1)}, \cdots, \boldsymbol{h}\_{t-K}^{(l-1)}\right)
$$

### 6.1.2 有外部输入的非线性自回归模型
自回归模型（AutoRegressive Model，AR）：用变量历史信息预测自己。
$$
\boldsymbol{y}\_{t}=w\_{0}+\sum\_{k=1}^{K} w\_{k} \boldsymbol{y}\_{t-k}+\epsilon\_{t}
$$

有外部输入的非线性自回归模型（Nonlinear AutoRegressive with Exogenous Inputs Model，NARX）： 每个时候都有输入输出，通过延时器记录最近$K_x$次输入和$K_y$次输出，则t时刻输出$y_t$为
$$
\boldsymbol{y}\_{t}=f\left(\boldsymbol{x}\_{t}, \boldsymbol{x}\_{t-1}, \cdots, \boldsymbol{x}\_{t-K\_{x}}, \boldsymbol{y}\_{t-1}, \boldsymbol{y}\_{t-2}, \cdots, \boldsymbol{y}\_{t-K\_{y}}\right)
$$

### 6.1.3 循环神经网络
活性值/状态（State）/隐状态（Hidden State）更新：
$$
\boldsymbol{h}\_{t}=f\left(\boldsymbol{h}\_{t-1}, \boldsymbol{x}\_{t}\right)
$$

![81e4df99fedfe0f854dccd8d3b624010.png](../../_resources/0bdf149e55604324a5ba88d700bc1455.png)

RNN可以近似任意非线性动力系统
FNN模拟任何连续函数，RNN模拟任何程序

## 6.2 简单循环网络
简单循环网络（Simple Recurrent Network，SRN）
$$
z\_{t}=U \boldsymbol{h}\_{t-1}+W x\_{t}+\boldsymbol{b}
$$
$$
\boldsymbol{h}\_{t}=f\left(\boldsymbol{z}\_{t}\right)
$$

### 6.2.1 循环神经网络的计算能力
对于：
$$
\begin{array}{l}
\boldsymbol{h}\_{t}=f\left(\boldsymbol{U} \boldsymbol{h}\_{t-1}+\boldsymbol{W} \boldsymbol{x}\_{t}+\boldsymbol{b}\right) \\\\
\boldsymbol{y}\_{t}=\boldsymbol{V} \boldsymbol{h}\_{t}
\end{array}
$$

RNN的通用近似定理：全连接RNN在足够多sigmoid隐藏神经元的情况下，可以以任意准确度近似任何一个非线性动力系统：
$$
\begin{array}{l}
\boldsymbol{s}\_{t}=g\left(\boldsymbol{s}\_{t-1}, \boldsymbol{x}\_{t}\right) \\\\
\boldsymbol{y}\_{t}=o\left(\boldsymbol{s}\_{t}\right)
\end{array}
$$

RNN是图灵完备的：所有图灵机可以被一个由Sigmoid型激活函数的神经元构成的全连接循环网络来进行模拟，可以近似解决所有可计算问题


## 6.3 应用到机器学习
### 6.3.1 序列到类别模式
主要用于序列分类
- 用最终状态表征序列：$\hat{y}=g\left(\boldsymbol{h}\_{T}\right)$
- 用状态平均表征序列：$\hat{y}=g\left(\frac{1}{T} \sum\_{t=1}^{T} \boldsymbol{h}\_{t}\right)$

![16de54428a759296151c0a40ceb3a148.png](../../_resources/635fea0b68a447e4bc084e4196738e6a.png)

### 6.3.2 同步的序列到序列模式
主要用于序列标注（Sequence Labeling）
$$
\hat{y}\_{t}=g\left(\boldsymbol{h}\_{t}\right), \quad \forall t \in[1, T]
$$
![504712ba1ce4cd0c08fbe135fd01551c.png](../../_resources/b5d5258f9b154c3abdf742c9a8a95317.png)

### 6.3.3 异步的序列到序列模式
也称编码器-解码器（Encoder-Decoder）模型
$$
\begin{aligned}
\boldsymbol{h}\_{t} &=f\_{1}\left(\boldsymbol{h}\_{t-1}, \boldsymbol{x}\_{t}\right), & & \forall t \in[1, T] \\\\
\boldsymbol{h}\_{T+t} &=f\_{2}\left(\boldsymbol{h}\_{T+t-1}, \hat{\boldsymbol{y}}\_{t-1}\right), & & \forall t \in[1, M] \\\\
\hat{y}\_{t} &=g\left(\boldsymbol{h}\_{T+t}\right), & & \forall t \in[1, M]
\end{aligned}
$$

![d11493a69d1333c611359c6852299976.png](../../_resources/d76fb7f95df3464bbf3db7294b4df942.png)

## 6.4 参数学习
以同步的序列到序列为例：
$$
\mathcal{L}\_{t}=\mathcal{L}\left(y\_{t}, g\left(\boldsymbol{h}\_{t}\right)\right)
$$
$$
\mathcal{L}=\sum\_{t=1}^{T} \mathcal{L}\_{t}
$$
$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{U}}=\sum\_{t=1}^{T} \frac{\partial \mathcal{L}\_{t}}{\partial \boldsymbol{U}}
$$

### 6.4.1 随时间反向传播算法
随时间反向传播（BackPropagation Through Time，BPTT）
$$
\frac{\partial \mathcal{L}\_{t}}{\partial u\_{i j}}=\sum\_{k=1}^{t} \frac{\partial^{+} z\_{k}}{\partial u\_{i j}} \frac{\partial \mathcal{L}\_{t}}{\partial z\_{k}}
$$

t时刻的损失对k时刻的隐藏层净输入的导数：
$$
\begin{aligned}
\delta\_{t, k} &=\frac{\partial \mathcal{L}\_{t}}{\partial z\_{k}} \\\\
&=\frac{\partial \boldsymbol{h}\_{k}}{\partial \boldsymbol{z}\_{k}} \frac{\partial \boldsymbol{z}\_{k+1}}{\partial \boldsymbol{h}\_{k}} \frac{\partial \mathcal{L}\_{t}}{\partial \boldsymbol{z}\_{k+1}} \\\\
&=\operatorname{diag}\left(f^{\prime}\left(\boldsymbol{z}\_{k}\right)\right) \boldsymbol{U}^{\top} \delta\_{t, k+1}
\end{aligned}
$$

参数梯度：
$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{U}}=\sum\_{t=1}^{T} \sum\_{k=1}^{t} \delta\_{t, k} \boldsymbol{h}\_{k-1}^{\top}
$$
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}} &=\sum\_{t=1}^{T} \sum\_{k=1}^{t} \delta\_{t, k} \boldsymbol{x}\_{k}^{\top}, \\\\
\frac{\partial \mathcal{L}}{\partial \boldsymbol{b}} &=\sum\_{t=1}^{T} \sum\_{k=1}^{t} \delta\_{t, k}
\end{aligned}
$$
![c53fb3c1cd1aa036ad02bd8a9dc38abc.png](../../_resources/7b3142af8f174aef8a8237f9882db903.png)

### 6.4.2 实时循环学习算法
实时循环学习（Real-Time Recurrent Learning，RTRL）：前向传播计算梯度
$$
\frac{\partial \mathcal{L}\_{t}}{\partial u\_{i j}}=\frac{\partial \boldsymbol{h}\_{t}}{\partial u\_{i j}} \frac{\partial \mathcal{L}\_{t}}{\partial \boldsymbol{h}\_{t}}
$$

两种算法比较：
- 一般为网络输出维度远地域输入，BPTT计算量更小
- BPTT保持所有时刻的中间维度，空间复杂度较高
- RTRL不需要梯度回传，适合在线学习或无限序列任务


## 6.5 长程依赖问题
由于梯度消失或爆炸问题，很难建模长时间间隔（Long Range）的状态之间的依赖关系

### 6.5.1 改进方案
梯度爆炸：权重衰减（正则项）、梯度截断

梯度消失：优化技巧、改变模型


## 6.6 基于门控的RNN
基于门控的循环神经网络（Gated RNN）

### 6.6.1 长短期记忆网络
长短期记忆网络（Long Short-Term Memory Network，LSTM）

引入新的内部状态（internal state）$\boldsymbol{c}\_{t} \in \mathbb{R}^{D}$ 专门进行线性的循环信息传递，同时非线性地输出信息给隐藏层的外部状态 $\boldsymbol{h}\_{t} \in \mathbb{R}^{D}$

$$
\begin{aligned}
\boldsymbol{c}\_{t} &=\boldsymbol{f}\_{t} \odot \boldsymbol{c}\_{t-1}+\boldsymbol{i}\_{t} \odot \tilde{\boldsymbol{c}}\_{t} \\\\
\boldsymbol{h}\_{t} &=\boldsymbol{o}\_{t} \odot \tanh \left(\boldsymbol{c}\_{t}\right)
\end{aligned}
$$

候选状态：

$$
\tilde{\boldsymbol{c}}\_{t}=\tanh \left(\boldsymbol{W}\_{c} \boldsymbol{x}\_{t}+\boldsymbol{U}\_{c} \boldsymbol{h}\_{t-1}+\boldsymbol{b}\_{c}\right)
$$

门控机制：
$$
\begin{aligned}
\boldsymbol{i}\_{t} &=\sigma\left(\boldsymbol{W}\_{i} \boldsymbol{x}\_{t}+\boldsymbol{U}\_{i} \boldsymbol{h}\_{t-1}+\boldsymbol{b}\_{i}\right) \\\\
\boldsymbol{f}\_{t} &=\sigma\left(\boldsymbol{W}\_{f} \boldsymbol{x}\_{t}+\boldsymbol{U}\_{f} \boldsymbol{h}\_{t-1}+\boldsymbol{b}\_{f}\right) \\\\
\boldsymbol{o}\_{t} &=\sigma\left(\boldsymbol{W}\_{o} \boldsymbol{x}\_{t}+\boldsymbol{U}\_{o} \boldsymbol{h}\_{t-1}+\boldsymbol{b}\_{o}\right)
\end{aligned}
$$
![408288a5d3a5eb6047f9becdea204fdc.png](../../_resources/f361134743fc46d1860e3fe8d5b039e6.png)

简洁描述：
$$
\begin{aligned}
\left[\begin{array}{c}
\tilde{\boldsymbol{c}}\_{t} \\\\
\boldsymbol{o}\_{t} \\\\
\boldsymbol{i}\_{t} \\\\
\boldsymbol{f}\_{t}
\end{array}\right] &=\left[\begin{array}{c}
\tanh \\\\
\sigma \\\\
\sigma \\\\
\sigma
\end{array}\right]\left(\boldsymbol{W}\left[\begin{array}{c}
\boldsymbol{x}\_{t} \\\\
\boldsymbol{h}\_{t-1}
\end{array}\right]+\boldsymbol{b}\right), \\\\
\boldsymbol{c}\_{t} &=\boldsymbol{f}\_{t} \odot \boldsymbol{c}\_{t-1}+\boldsymbol{i}\_{t} \odot \tilde{\boldsymbol{c}}\_{t} \\\\
\boldsymbol{h}\_{t} &=\boldsymbol{o}\_{t} \odot \tanh \left(\boldsymbol{c}\_{t}\right)
\end{aligned}
$$

记忆：
- 短期记忆：S-RNN中隐状态，每个时刻都会被重写
- 长期记忆：网络参数
- 长短期记忆LSTM：记忆单元 c 保存的信息生命周期长于短期记忆h，短于长期记忆

参数设置：
一般深度学习初始参数比较小，但是遗忘的参数初始值一般设得比较大，偏置向量 $b_f$ 设为1或2，防止大量遗忘导致难以捕捉长距离依赖信息（梯度弥散）

### 6.6.2 LSTM网络的各种变体
改进门控机制：
1. 无遗忘门的 LSTM 网络（[Hochreiter et al., 1997] 最早提出的 LSTM 网络）：
$$
\boldsymbol{c}\_{t}=\boldsymbol{c}\_{t-1}+\boldsymbol{i}\_{t} \odot \tilde{\boldsymbol{c}}\_{t}
$$

2. peephole连接：也依赖与上一时刻记忆单元：
$$
\begin{aligned}
\boldsymbol{i}\_{t} &=\sigma\left(\boldsymbol{W}\_{i} \boldsymbol{x}\_{t}+\boldsymbol{U}\_{i} \boldsymbol{h}\_{t-1}+\boldsymbol{V}\_{i} \boldsymbol{c}\_{t-1}+\boldsymbol{b}\_{i}\right) \\\\
\boldsymbol{f}\_{t} &=\sigma\left(\boldsymbol{W}\_{f} \boldsymbol{x}\_{t}+\boldsymbol{U}\_{f} \boldsymbol{h}\_{t-1}+\boldsymbol{V}\_{f} \boldsymbol{c}\_{t-1}+\boldsymbol{b}\_{f}\right) \\\\
\boldsymbol{o}\_{t} &=\sigma\left(\boldsymbol{W}\_{o} \boldsymbol{x}\_{t}+\boldsymbol{U}\_{o} \boldsymbol{h}\_{t-1}+\boldsymbol{V}\_{o} \boldsymbol{c}\_{t}+\boldsymbol{b}\_{o}\right)
\end{aligned}
$$

3. 耦合输入门和遗忘门（$\boldsymbol{f}\_{t}=1-\boldsymbol{i}\_{t}$）：
$$
\boldsymbol{c}\_{t}=\left(1-\boldsymbol{i}\_{t}\right) \odot \boldsymbol{c}\_{t-1}+\boldsymbol{i}\_{t} \odot \tilde{\boldsymbol{c}}\_{t}
$$

### 门控循环单元网络
门控循环单元（Gated Recurrent Unit，GRU）网络 
- 不引入额外记忆单元
- 更新门（Update Gate）
- 重置门（Reset Gate）
$$
z\_{t}=\sigma\left(W\_{z} x\_{t}+U\_{z} h\_{t-1}+b\_{z}\right)
$$
$$
\boldsymbol{r}\_{t}=\sigma\left(\boldsymbol{W}\_{r} \boldsymbol{x}\_{t}+\boldsymbol{U}\_{r} \boldsymbol{h}\_{t-1}+\boldsymbol{b}\_{r}\right)
$$
$$
\tilde{\boldsymbol{h}}\_{t}=\tanh \left(\boldsymbol{W}\_{h} \boldsymbol{x}\_{t}+\boldsymbol{U}\_{h}\left(\boldsymbol{r}\_{t} \odot \boldsymbol{h}\_{t-1}\right)+\boldsymbol{b}\_{h}\right)
$$
$$
\boldsymbol{h}\_{t}=\boldsymbol{z}\_{t} \odot \boldsymbol{h}\_{t-1}+\left(1-\boldsymbol{z}\_{t}\right) \odot \tilde{\boldsymbol{h}}\_{t}
$$

![c4496410ffccdb84a1f142483d6295a2.png](../../_resources/cba3271bf4a145739bb6734065d163a3.png) 
- 当 $z_t$ = 0, r = 1 时，GRU 网络退化为简单循环网络


## 6.7 深层RNN
加深x到y的路径

### 6.7.1 堆叠循环神经网络
堆叠循环神经网络（Stacked Recurrent Neural Network，SRNN）
- 其中，堆叠的简单循环网络（Stacked SRN）也成为循环多层感知机（Recurrent MultiLayer Perceptron，RMLP）

$$
\boldsymbol{h}\_{t}^{(l)}=f\left(\boldsymbol{U}^{(l)} \boldsymbol{h}\_{t-1}^{(l)}+\boldsymbol{W}^{(l)} \boldsymbol{h}\_{t}^{(l-1)}+\boldsymbol{b}^{(l)}\right)
$$
![941131d7dbea18699d55db024d53ff45.png](../../_resources/d9c6db16a0a24964a5497bb6d88b44bb.png)

### 6.7.2 双向循环神经网络
双向循环神经网络（Bidirectional Recurrent Neural Network，Bi-RNN）

$$
\begin{aligned}
\boldsymbol{h}\_{t}^{(1)} &=f\left(\boldsymbol{U}^{(1)} \boldsymbol{h}\_{t-1}^{(1)}+\boldsymbol{W}^{(1)} \boldsymbol{x}\_{t}+\boldsymbol{b}^{(1)}\right) \\\\
\boldsymbol{h}\_{t}^{(2)} &=f\left(\boldsymbol{U}^{(2)} \boldsymbol{h}\_{t+1}^{(2)}+\boldsymbol{W}^{(2)} \boldsymbol{x}\_{t}+\boldsymbol{b}^{(2)}\right) \\\\
\boldsymbol{h}\_{t} &=\boldsymbol{h}\_{t}^{(1)} \oplus \boldsymbol{h}\_{t}^{(2)}
\end{aligned}
$$

![065229f6defac23475a14f553250364b.png](../../_resources/8100f4e648b14ddeb4a6692e740ce82c.png)


## 6.8 扩展到图结构

### 6.8.1 递归神经网络
递归神经网络（Recursive Neural Network，RecNN）：RNN在有向无环图上的扩展
![2ee792e0be9dee2160ba2e01b5c8441b.png](../../_resources/18e3c5dc6352458ebf5b91d8a8c11435.png)
- RecNN退化为线性序列结构时，等价于简单循环网络

RecNN主要用于建模自然语言句子的语义

树结构的长短期记忆模型（Tree-Structured LSTM）

### 6.8.2 图神经网络
图神经网络（Graph Neural Network，GNN）

每个结点 v 用一组神经元表示其状态$\boldsymbol{h}^{(v)}$，初始状态为节点 v 的输入特征$\boldsymbol{x}^{(v)}$。每个节点接收相邻节点的消息，并更新自己的状态

$$
\begin{aligned}
\boldsymbol{m}\_{t}^{(v)} &=\sum\_{u \in \mathcal{N}(v)} f\left(\boldsymbol{h}\_{t-1}^{(v)}, \boldsymbol{h}\_{t-1}^{(u)}, \boldsymbol{e}^{(u, v)}\right), \\\\
\boldsymbol{h}\_{t}^{(v)} &=g\left(\boldsymbol{h}\_{t-1}^{(v)}, \boldsymbol{m}\_{t}^{(v)}\right)
\end{aligned}
$$

上式为同步更新，对于有向图采用异步更新会更有效率，比如RNN和RecNN。

读出函数（Readout Function）得到整个网络的表示：
$$
\boldsymbol{o}\_{t}=g\left(\left\{\boldsymbol{h}\_{T}^{(v)} \mid v \in \mathcal{V}\right\}\right)
$$


## 习题
#### 习题 6-1 分析延时神经网络、卷积神经网络和循环神经网络的异同点．
同：共享权重
异：
1. 延时神经网络依赖最近K个状态（活性值），RNN依赖之前所有状态
2. RNN在时间维度共享权重，CNN在空间维度共享权重

#### 习题 6-2 推导公式 (6.40) 和公式 (6.41) 中的梯度．
分别替换 $\boldsymbol{h}\_{k-1}^{\top}$ 为 $\boldsymbol{x}\_{k}^{\top}$ 和 $1$ 即可

#### 习题6-3 当使用公式(6.50) 作为循环神经网络的状态更新公式时，分析其可能存在梯度爆炸的原因并给出解决方法．
公式（6.50）：

$$
\boldsymbol{h}_{t}=\boldsymbol{h}_{t-1}+g\left(\boldsymbol{x}_{t}, \boldsymbol{h}_{t-1} ; \theta\right),
$$

计算误差项时梯度可能过大，不断反向累积导致梯度爆炸：

$$
\begin{aligned}
\delta_{t, k} &=\frac{\partial \mathcal{L}_{t}}{\partial \boldsymbol{z}_{k}} \\
&=\frac{\partial \boldsymbol{h}_{k}}{\partial \boldsymbol{z}_{k}} \frac{\partial \boldsymbol{z}_{k+1}}{\partial \boldsymbol{h}_{k}} \frac{\partial \mathcal{L}_{t}}{\partial \boldsymbol{z}_{k+1}} \\
&=\operatorname{diag}\left(f^{\prime}\left(\boldsymbol{z}_{k}\right)\right) \boldsymbol{U}^{\top} \delta_{t, k+1}
\end{aligned}
$$

解决方法：引入门控机制等。

#### 习题 6-4 推导 LSTM 网络中参数的梯度，并分析其避免梯度消失的效果．

#### 习题 6-5 推导 GRU 网络中参数的梯度，并分析其避免梯度消失的效果．

#### 习题 6-6 除了堆叠循环神经网络外，还有什么结构可以增加循环神经网络深度？
增加神经网络深度主要方法：增加同一时刻网络输入到输出之间的路径。

如：堆叠神经网络、双向循环网络等。

#### 习题 6-7 证明当递归神经网络的结构退化为线性序列结构时，递归神经网络就等价于简单循环神经网络．
RecNN 退化为线性序列结构时：
$$
\boldsymbol{h}\_{t}=\sigma\left(\boldsymbol{W}\left[\begin{array}{l}
\boldsymbol{h}\_{t-1} \\\\
\boldsymbol{x}\_{t}
\end{array}\right]+\boldsymbol{b}\right)
$$
显而易见，即 SRN.
