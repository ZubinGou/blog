# 《神经网络与深度学习》第8章 - 注意力机制和外部记忆


用CNN等编码一个向量表示文本所有特征，存在信息瓶颈

网络容量（Network Capacity）：存储信息受限与神经元数量和网络复杂度
- 对于过载信息：引入注意力和记忆机制

## 8.1 认知神经科学中的注意力
注意力（Attention）：从大量信息中选择小部分有用信息重点处理，忽略其他信息

注意力分类：
1. 自上而下的有意识的注意力，称为聚焦式注意力（Focus Attention）/ 选择性注意力（Selective Attention）：有预定目的、依赖任务的，主动有意识地聚焦于某一对象的注
意力
2. 自下而上的无意识的注意力，称为基于显著性的注意力（Saliency-Based Attention）：如果一个对象的刺激信息不同于其周围信息，一种无意识的“赢者通吃”（Winner-Take-All）或者门控（Gating）机制就可以把注意力转向这个对象


## 8.2 注意力机制
可以将最大汇聚（Max Pooling）、门控（Gating）机制近似地看作自下而上的基于显著性的注意力机制

注意力机制计算步骤：
1. 在所有输入信息上计算注意力分布
2. 根据注意力分布计算输入信息的加权分布

**注意力分布**
$$
\begin{aligned}
\alpha\_{n} &=p(z=n \mid \boldsymbol{X}, \boldsymbol{q}) \\\\
&=\operatorname{softmax}\left(s\left(\boldsymbol{x}\_{n}, \boldsymbol{q}\right)\right) \\\\
&=\frac{\exp \left(s\left(\boldsymbol{x}\_{n}, \boldsymbol{q}\right)\right)}{\sum\_{j=1}^{N} \exp \left(s\left(\boldsymbol{x}\_{j}, \boldsymbol{q}\right)\right)}
\end{aligned}
$$
- 查询向量（Query Vector）$q$
- 注意力变量$z \in[1, N]$，表示被选择信息的索引位置
- 注意力分布（Attention Distribution）$\alpha\_{n}$
- 注意力打分函数$s(\boldsymbol{x}, \boldsymbol{q})$

$$
\begin{aligned}
\text { 加性模型 } & & 
s(\boldsymbol{x}, \boldsymbol{q})&=\boldsymbol{v}^{\top} \tanh (\boldsymbol{W} \boldsymbol{x}+\boldsymbol{U} \boldsymbol{q})\\
\text { 点积模型 } & & s(\boldsymbol{x}, \boldsymbol{q}) &=\boldsymbol{x}^{\top} \boldsymbol{q} \\\\
\text { 缩放点积模型 } & & s(\boldsymbol{x}, \boldsymbol{q}) &=\frac{\boldsymbol{x}^{\top} \boldsymbol{q}}{\sqrt{D}}, \\\\
\text { 双线性模型 } & & s(\boldsymbol{x}, \boldsymbol{q})&=\boldsymbol{x}^{\top} \boldsymbol{W} \boldsymbol{q},
\end{aligned}
$$

**加权平均**
软性注意力机制（Soft Attention Mechanism
$$
\begin{aligned}
\operatorname{att}(\boldsymbol{X}, \boldsymbol{q}) &=\sum\_{n=1}^{N} \alpha\_{n} \boldsymbol{x}\_{n}, \\\\
&=\mathbb{E}\_{\boldsymbol{z} \sim p(z \mid \boldsymbol{X}, \boldsymbol{q})}\left[\boldsymbol{x}\_{z}\right]
\end{aligned}
$$

![692288459ef878cf2172e5a8cd155df7.png](/blog/_resources/b379b8fe2711480f8af5978078e21fec.png)

### 8.2.1 注意力机制的变体
#### 硬性注意力
只关注某一个输入向量

两种实现方式：
1. 最大采样：$\operatorname{att}(\boldsymbol{X}, \boldsymbol{q})=\boldsymbol{x}\_{\hat{n}}$，$\hat{n}=\underset{n=1}{\arg \max } \alpha\_{n}$
2. 随机采样（根据注意力分布）

硬性注意力缺点：损失函数和注意力分布函数关系不可导，无法用BP训练，通常采用强化学习训练。

#### 键值对注意力
键计算注意力分布，值计算聚合信息
$$
\begin{aligned}
\operatorname{att}((\boldsymbol{K}, \boldsymbol{V}), \boldsymbol{q}) &=\sum\_{n=1}^{N} \alpha\_{n} \boldsymbol{v}\_{n} \\\\
&=\sum\_{n=1}^{N} \frac{\exp \left(s\left(\boldsymbol{k}\_{n}, \boldsymbol{q}\right)\right)}{\sum\_{j} \exp \left(s\left(\boldsymbol{k}\_{j}, \boldsymbol{q}\right)\right)} \boldsymbol{v}\_{n},
\end{aligned}
$$

#### 多头注意力
多头注意力（Multi-Head Attention）是利用多个查询 𝑸 = [𝒒1, ⋯ , 𝒒𝑀]，来并行地从输入信息中选取多组信息：

$$
\operatorname{att}((\boldsymbol{K}, \boldsymbol{V}), \boldsymbol{Q})=\operatorname{att}\left((\boldsymbol{K}, \boldsymbol{V}), \boldsymbol{q}\_{1}\right) \oplus \cdots \oplus \operatorname{att}\left((\boldsymbol{K}, \boldsymbol{V}), \boldsymbol{q}\_{M}\right)
$$

#### 结构化注意力
输入信息本身具有层次（Hierarchical）结构，比如文本可以分为词、句子、段落、篇章等不同粒度的层次。

可以使用层次化的注意力来进行更好的信息选择 [Yang et al.,2016]

可以假设注意力为上下文相关的二项分布，用一种图模型来构建更复杂的结构化注意力分布 [Kim et al., 2017]

#### 指针网络
指针网络（Pointer Network）[Vinyals et al., 2015]是一种Seq2seq模型，输入$\boldsymbol{X}=\boldsymbol{x}\_{1}, \cdots, \boldsymbol{x}\_{N}$，输出$\boldsymbol{c}\_{1: M}=c\_{1}, c\_{2}, \cdots, c\_{M}, c\_{m} \in[1, N], \forall m$

输出为输入序列的下标

$$
\begin{aligned}
p\left(c\_{1: M} \mid \boldsymbol{x}\_{1: N}\right) &=\prod\_{m=1}^{M} p\left(c\_{m} \mid c\_{1:(m-1)}, \boldsymbol{x}\_{1: N}\right) \\\\
& \approx \prod\_{m=1}^{M} p\left(c\_{m} \mid \boldsymbol{x}\_{c\_{1}}, \cdots, \boldsymbol{x}\_{c\_{m-1}}, \boldsymbol{x}\_{1: N}\right),
\end{aligned}
$$

$$
p\left(c\_{m} \mid c\_{1:(m-1)}, \boldsymbol{x}\_{1: N}\right)=\operatorname{softmax}\left(s\_{m, n}\right)
$$
$$
s\_{m, n}=\boldsymbol{v}^{\top} \tanh \left(\boldsymbol{W} \boldsymbol{x}\_{n}+\boldsymbol{U} \boldsymbol{h}\_{m}\right), \forall n \in[1, N]
$$

![9b5068f5d270171849d71b4e1eee55d7.png](/blog/_resources/43a613fd332042f3a2d3be63c377a81e.png)


## 8.3 自注意力模型
自注意力模型（Self-Attention Model）：即内部注意力（Intra-Attention）

自注意力可以作为神经网络中的一层来使用，有效地建模长距离依赖问题 [Vaswani et al., 2017]

经常采用查询-键-值（Query-Key-Value，QKV）模式

![87adaecfc12645c8f7cbc44cb917b18e.png](/blog/_resources/ff0c8448082f4eaba2e4762f11527df3.png)

$$
\begin{array}{l}
\boldsymbol{Q}=\boldsymbol{W}\_{q} \boldsymbol{X} \in \mathbb{R}^{D\_{k} \times N} \\\\
\boldsymbol{K}=\boldsymbol{W}\_{k} \boldsymbol{X} \in \mathbb{R}^{D\_{k} \times N} \\\\
\boldsymbol{V}=\boldsymbol{W}\_{v} \boldsymbol{X} \in \mathbb{R}^{D\_{v} \times N}
\end{array}
$$

$$
\begin{aligned}
\boldsymbol{h}\_{n} &=\operatorname{att}\left((\boldsymbol{K}, \boldsymbol{V}), \boldsymbol{q}\_{n}\right) \\\\
&=\sum\_{j=1}^{N} \alpha\_{n j} \boldsymbol{v}\_{j} \\\\
&=\sum\_{j=1}^{N} \operatorname{softmax}\left(s\left(\boldsymbol{k}\_{j}, \boldsymbol{q}\_{n}\right)\right) \boldsymbol{v}\_{j},
\end{aligned}
$$

如使用缩放点积打分，输出可以简写为：
$$
\boldsymbol{H}=\boldsymbol{V} \operatorname{softmax}\left(\frac{\boldsymbol{K}^{\top} \boldsymbol{Q}}{\sqrt{D\_{k}}}\right)
$$

![c0f955a106e62040cc66ffb13a42e1ba.png](/blog/_resources/66a6091dc1aa44d9ba9e7f9d965f6134.png)
- 实线是可学习的权重
- 虚线是动态生成的权重，可以处理变长信息


## 8.4 人脑中的记忆
信息作为一种整体效应（Collective Effect）存储在大脑组织中，即记忆在大脑皮层是分布式存储的，而不是存储于某个局部区域。

人脑记忆具有周期性和联想性

**记忆周期**
- 长期记忆（Long-Term Memory）：也称为结构记忆或知识（Knowledge），体现为神经元之间的连接形态，其更新速度比较慢
	- 类比权重参数
- 短期记忆（Short-Term Memory）：体现为神经元的活动，更新较快，维持时间为几秒至几分钟
	- 类比隐状态
- 工作记忆（Working Memory）：人脑的缓存，短期记忆一般指输入信息在人脑中的表示和短期存储，工作记忆是和任务相关的“容器”。容量较小，一般可以容纳4组项目。

演化（Evolution）过程：．短期记忆、长期记忆的动态更新过程

**联想记忆**

大脑主要通过**联想**进行检索

联想记忆（Associative Memory）：学习和记住不同对象之间关系的能力

- 基于内容寻址的存储（Content-Addressable Memory，CAM）：联想记忆，通过内容匹配方法进行寻址的信息存储方式
- 随机访问存储（Random Access Memory, RAM)：现代计算机根据地址存储

类比：
- LSTM记忆单元 <-> 计算机的寄存器
- 外部记忆 <-> 计算机的内存

神经网络引入外部记忆途径：
1. 结构化记忆，类似于计算机存储信息
2. 基于神经动力学的联想记忆，有更好的生物学解释性

![3ae89e265dba855042f5db23b1f9c61a.png](/blog/_resources/8a5706990c284ab193cce496ca1f7412.png)


## 8.5 记忆增强神经网络
记忆增强神经网络（Memory Augmented Neural Network，MANN）：简称为记忆网络（Memory Network，MN），装备外部记忆的神经网络。

![6aa33a6b3d8043498fd27adc794bd1b7.png](/blog/_resources/1901836e0eea40c78bcd5df784fa77c5.png)

外部记忆将参数与记忆容量分离，在少量增加参数的条件下可以大幅增加网络容量。因此可以将注意力机制看作一个接口，将信息的存储与计算分离。

### 8.5.1 端到端记忆网络
端到端记忆网络（End-To-End Memory Network，MemN2N）[Sukhbaatar et al., 2015] ：可微网络结构，可以多次从外部记忆中读取信息（只读）。

主网络根据输入 𝒙 生成 𝒒，并使用键值对注意力机制来从外部记忆中读取相关信息 𝒓，
$$
\boldsymbol{r}=\sum\_{n=1}^{N} \operatorname{softmax}\left(\boldsymbol{a}\_{n}^{\top} \boldsymbol{q}\right) \boldsymbol{c}\_{n}
$$
并产生输出：
$$
y=f(q+r)
$$

多跳操作：主网络与外部记忆进行多轮交互，根据上次读取信息继续查询读取。

![202d7bb29bf23c9278407f91061d0427.png](/blog/_resources/995e0678775f4c469114e2a7aeb93079.png)

### 8.5.2 神经图灵机
神经图灵机（Neural Turing Machine，NTM）：由控制器和外部记忆构成。
- 外部记忆：矩阵$M \in \mathbb{R}^{D \times N}$
- 控制器：前馈或循环神经网络

寻址：基于位置、基于内容

基于内容：
![bab7e1792a4fbe0f0545387301357c22.png](/blog/_resources/794eb31a2924496b929fabd58f5010be.png)
- 读向量（read vector）$r_t$
- 删除向量（erase vector）$e_t$
- 增加向量（add vector）$a_t$

$$
\boldsymbol{m}\_{t+1, n}=\boldsymbol{m}\_{t, n}\left(1-\alpha\_{t, n} \boldsymbol{e}\_{t}\right)+\alpha\_{t, n} \boldsymbol{a}\_{t}, \quad \forall n \in[1, N]
$$


## 8.6 基于神经动力学的联想记忆
将基于神经动力学（Neurodynamics）的联想记忆模型引入到神经网络以增加网络容量。联想记忆模型可以利用神经动力学原理实现按内容寻址的信息存储和检索。

联想记忆模型（Associative Memory Model）主要是通过神经网络的动态演化来进行联想，有两种应用场景：
1. 自联想模型（Auto-Associative Model）/自编码器（Auto-Encoder，AE）：输入和输出模式在同一空间。
2. 异联想模型（Hetero-Associative Model）：输入输出模式不在同一空间

### 8.6.1 Hopfield 网络
Hopfield 网络（Hopfield Network）：一种RNN模型，由一组相互连接的神经元构成。所有神经元连接不分层。

![93b6af0afc093e898975074cd9ad78cc.png](/blog/_resources/bb19850f03ac4f4f9bfef51248403cdc.png)

下面讨论离散 Hopfield 网络，神经元状态为 {+1, −1} 两种，还有连续 Hopfield 网络，即神经元状态为连续值。

第i个神经元更新规则：
$$
S\_{i}=\left\{\begin{array}{ll}
+1 & \text { if } \sum\_{j=1}^{M} w\_{i j} S\_{j}+b\_{i} \geq 0 \\\\
-1 & \text { otherwise }
\end{array}\right.
$$

连接权重 $w\_{ij}$ 有以下性质：
$$
\begin{array}{ll}
w\_{i i} & =0 \quad \forall i \in[1, M] \\\\
w\_{i j} & =w\_{j i} \quad \forall i, j \in[1, M]
\end{array}
$$

更新方式：
- 异步：每次随机或者按顺序更新一个神经元
- 同步：一次更新所有神经元，需要同步时钟

**能量函数**
Hopfield 网络的能量函数（Energy Function）：
$$
\begin{aligned}
E &=-\frac{1}{2} \sum\_{i, j} w\_{i j} s\_{i} s\_{j}-\sum\_{i} b\_{i} s\_{i} \\\\
&=-\frac{1}{2} \boldsymbol{s}^{\top} \boldsymbol{W} \boldsymbol{s}-\boldsymbol{b}^{\top} \boldsymbol{s}
\end{aligned}
$$

吸引点（Attractor）：稳态，能量的局部最低点
![4671ecbbe2b44948a717ec01ac8dc9cf.png](/blog/_resources/8d58d18e599044bea061c95acd00ecd5.png)

**联想记忆**
Hopfield 网络会收敛到所处管辖区域内的吸引点，将吸引点看作网络存储中的模式（Pattern），Hopfield的检索是基于内容寻址的检索。

**信息存储**
信息存储是指将一组向量$x\_{1}, \cdots, x\_{N}$存储在网络中的过程，存储过程主要是调整神经元之间的连接权重，因此可以看作一种学习过程。

学习规则可以是简单的平均点积：
$$
w\_{i j}=\frac{1}{N} \sum\_{n=1}^{N} x\_{i}^{(n)} x\_{j}^{(n)}
$$

赫布规则（Hebbian Rule，或 Hebb’s Rule）：常同时激活的神经元连接加强，反之连接消失。

**存储容量**
对于数量为 $M$ 的互相连接的二值神经元网络，其总状态数为 $2^M$

Hopfield 网络的最大容量为 0.14𝑀，玻尔兹曼机的容量为 0.6𝑀

改进学习算法、网络结构或者引入更复杂的运算，可以有效改进联想记忆网络的容量。

### 8.6.2 使用联想记忆增加网络容量
将联想记忆作为更大网络的组件，用来增加短期记忆的容量。参数可以使用Hebbian来学习，或者作为整个网络参数的一部分来学习。


## 习题
#### 习题 8-1 分析 LSTM 模型中隐藏层神经元数量与参数数量之间的关系．
![471dd2084810348f186999e4b837893d.png](/blog/_resources/97408a607c0c4569b5adf1f53585d232.png)
假设输入x维度为n，隐层神经元数为m，参数数量：
- 三个门+cell更新：$(m+n+1)\times m\times 4$

#### 习题 8-2 分析缩放点积模型可以缓解 Softmax 函数梯度消失的原因．
$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d\_{k}}}\right) V
$$

向量点积往往很大，Softmax函数在输入值都很大大的区域会将元素差距拉得非常大，$\hat{y}\_{k}$接近1，梯度也就接近0了

Scale：
$$
E\left(q\_{i} k\_{i}\right)=E q\_{i} E k\_{i}=0
$$

$$
\begin{aligned}
D\left(q\_{i} k\_{i}\right) &=E\left(q\_{i}^{2} k\_{i}^{2}\right)-\left(E\left(q\_{i} k\_{i}\right)\right)^{2} \\\\
&=E q\_{i}^{2} E k\_{i}^{2} \\\\
&=\left(D\left(q\_{i}\right)+\left(E q\_{i}\right)^{2}\right)\left(D\left(k\_{i}\right)+\left(E k\_{i}\right)^{2}\right) \\\\
&=\sigma^{4}=1
\end{aligned}
$$

$$
E\left(Q K^{T}\right)=\sum\_{i=1}^{d\_{k}} E\left(q\_{i} k\_{i}\right)=0
$$

$$
D\left(Q K^{T}\right)=\sum\_{i=1}^{d\_{k}} D\left(q\_{i} k\_{i}\right)=d\_{k} \sigma^{4}=d\_{k}
$$

点积期望为0，通过除以标准差缩放，相当于进行了标准化Standardization，控制softmax输入的方差为1，有效解决了梯度消失问题。


#### 习题 8-3 当将自注意力模型作为一个神经层使用时，分析它和卷积层以及循环层在建模长距离依赖关系的效率和计算复杂度方面的差异．
![b5c016b9f01702d1378a6abcbef4d44b.png](/blog/_resources/6e4ab592f48f4e4dba2d1a17f14191ce.png)
*图片来源：Why self-attention? [Tang, Gongbo, et al., 2018]*

近似时间复杂度：
- 自注意力：$O(1)$
- CNN：$O(\log_k(n))$
- RNN：$O(n)$

![7d24d6478abcae0c2041e65d0fe030a1.png](/blog/_resources/7611df07cdf14af9b62fde51577eedff.png)
*来源：cs224n, 2019, lec14*


#### 习题 8-4 试设计用集合、树、栈或队列来组织外部记忆，并分析它们的差异．
TODO

#### 习题 8-5 分析端到端记忆网络和神经图灵机对外部记忆操作的异同点．
区别：
- 端到端记忆网络：使用键值对注意力从外部记忆读取
- 神经图灵机：同时对所有记忆进行不同程度的读写

#### 习题 8-6 证明 Hopfield 网络的能量函数随时间单调递减
TODO
