# 《神经网络与深度学习》第4章 - 前馈神经网络


# ch4 前馈神经网络
## 4.1 神经元
$$
\begin{aligned}
z &=\sum\_{d=1}^{D} w\_{d} x\_{d}+b \\\\
&=\boldsymbol{w}^{\top} \boldsymbol{x}+b
\end{aligned}
$$

净输入z在经过非线性函数 𝑓(⋅) 后，得到神经元的活性值（Activation）$a=f(z)$

非线性函数 𝑓(⋅) 称为激活函数（Activation Function）
![7d28900fd1d1cc8d2b023a0160ff98f3.png](../../_resources/765784f8325947438aca3ed2be0d20f1.png)

激活函数：
1. 连续可导（允许少数点上不可导）的非线性函数
	- 可导则可以直接利用数值优化方法学习参数
2. 导函数尽可能简单，提高计算效率
3. 导函数值域区间合适，太大太小影响训练效率和稳定性

### 4.1.1 Sigmoid 型函数
指一类S型函数，两端饱和（正负无穷导函数趋近0），常用的有Logistic、Tanh 

Logistic 函数：
$$
\sigma(x)=\frac{1}{1+\exp (-x)}
$$
$$
\sigma^{\prime}(x)=\sigma(x)(1-\sigma(x))
$$
1. 其输出直接可以看作概率分布，使得神经网络可以更好地和统计学习模型进行结合．
2. 其可以看作一个软性门（Soft Gate），用来控制其他神经元输出信息的数量

Tanh 函数：
$$
\tanh (x)=\frac{\exp (x)-\exp (-x)}{\exp (x)+\exp (-x)}
$$
$$
\tanh (x)=2 \sigma(2 x)-1
$$
![8619ce619f2e3511a119f55472700993.png](../../_resources/4152da1b3afd4f868a1b854c911477b5.png)
- Tanh函数输出是零中心化的（Zero-Centered），非零中心化的输出会使得其后一层的神经元的输入发生偏置偏移（Bias Shift），并进一步使得梯度下降的收敛速度变慢．

#### 4.1.1.1 Hard-Logistic 函数和 Hard-Tanh 函数
分段函数近似

$$
\begin{aligned}
\text { hard-logistic }(x)=&\left\{\begin{array}{ll}
1 & g\_{l}(x) \geq 1 \\\\
g\_{l} & 0<g\_{l}(x)<1 \\\\
0 & g\_{l}(x) \leq 0
\end{array}\right.\\
&=\max \left(\min \left(g\_{l}(x), 1\right), 0\right) \\\\
&=\max (\min (0.25 x+0.5,1), 0) .
\end{aligned}
$$

$$
\begin{aligned}
\operatorname{hard}-\tanh (x) &=\max \left(\min \left(g\_{t}(x), 1\right),-1\right) \\\\
&=\max (\min (x, 1),-1)
\end{aligned}
$$

![7764b7a97d7da811d2d779bbf79c8574.png](../../_resources/b9827de3a4014af5a543d60d04f78914.png)


### 4.1.2 ReLU 函数
ReLU（Rectified Linear Unit，修正线性单元），也叫Rectifier函数

$$
\begin{aligned}
\operatorname{ReLU}(x) &=\left\{\begin{array}{ll}
x & x \geq 0 \\\\
0 & x<0
\end{array}\right.\\
&=\max (0, x)
\end{aligned}
$$

优点：
1. 生物学合理性（Biological Plausibility）
2. ReLU 具有很好的稀疏性，大约 50% 的神经元会处于激活状态
3. 相比于 Sigmoid 型函数的两端饱和，ReLU 函数为左饱和函数，一定程度缓解了梯度消失问题，加速收敛

缺点：
1. 非零中心化，给后一层引入偏置偏移
2. 死亡 ReLU 问题（Dying ReLU Problem）：一次不当更新引起永世不得激活

#### 4.1.2.1 带泄露的 ReLU
$$
\text { LeakyReLU( } x)=\max (x, \gamma x)
$$
避免永不激活

#### 4.1.2.2 带参数的 ReLU
$$
\begin{aligned}
\operatorname{PReLU}\_{i}(x) &=\left\{\begin{array}{ll}
x & \text { if } x>0 \\\\
\gamma\_{i} x & \text { if } x \leq 0
\end{array}\right.\\
&=\max (0, x)+\gamma\_{i} \min (0, x)
\end{aligned}
$$
引入可学习参数

#### 4.1.2.3 ELU 函数
ELU（Exponential Linear Unit，指数线性单元）
$$
\begin{aligned}
\operatorname{ELU}(x) &=\left\{\begin{array}{ll}
x & \text { if } x>0 \\\\
\gamma(\exp (x)-1) & \text { if } x \leq 0
\end{array}\right.\\
&=\max (0, x)+\min (0, \gamma(\exp (x)-1)),
\end{aligned}
$$
近似的零中心化的非线性函数

#### 4.1.2.4 Softplus 函数
$$
\text { Softplus }(x)=\log (1+\exp (x))
$$
- ReLU平滑版本
- Softplus 函数其导数刚好是 Logistic 函数
- 单侧抑制、宽兴奋边界的特性，却没有稀疏激活性

![e689bd1567b62772d50d4e76fc34b674.png](../../_resources/94a0b966fbd94803aed687ab707f176b.png)

### 4.1.3 Swish 函数
一种自门控（Self-Gated）激活函数
$$
\operatorname{swish}(x)=x \sigma(\beta x)
$$
![61c3a15d0ff86603bc42a17984b7b470.png](../../_resources/a21689c07ebc4a94a02d9f27bc33783b.png)
- 线性函数和 ReLU 函数之间的非线性插值函数，其程度由参数 𝛽 控制

### 4.1.4 GELU 函数
GELU（Gaussian Error Linear Unit，高斯误差线性单元），也是门控激活函数
$$
\operatorname{GELU}(x)=x P(X \leq x)
$$

### 4.1.5 Maxout 单元
- 一种分段线性函数
- 输入是上一层的全部原始输出$\boldsymbol{x}=\left[x\_{1} ; x\_{2} ; \cdots ; x\_{D}\right]$
- 采用 Maxout 单元的神经网络也叫作Maxout网络
$$
z\_{k}=\boldsymbol{w}\_{k}^{\top} \boldsymbol{x}+b\_{k}
$$
$$
\operatorname{maxout}(\boldsymbol{x})=\max \_{k \in[1, K]}\left(z\_{k}\right)
$$

## 4.2 网络结构
### 4.2.1 前馈网络
- 包括
	- 全连接前馈网络
	- 卷积神经网络
- 可以看作一个**函数**

### 4.2.2 记忆网络
- 也称反馈网络
- 具有记忆功能，可以接受历史信息
- 信息传播可以单向或双向，可以用有向循环图或者无向图表示
- 包括
	- 循环神经网络
	- Hopfield网络
	- 玻尔兹曼机
	- 受限玻尔兹曼机
- 记忆增强神经网络（Memory Augmented Neural Network，MANN）：为了增强记忆容量，引入外部记忆单元和读写机制，eg. 神经图灵机、记忆网络
- 可以看作一个**程序**

### 4.2.3 图网络
- 前馈网络和记忆网络很难处理图结构的数据，如知识图谱、社交网络、分子（Molecular ）网络
- 是前馈网络和记忆网络的泛化，实现方式很多
	- 图卷积网络（Graph Convolutional Network，GCN）
	- 图注意力网络（Graph Attention Network，GAT）
	- 消息传递神经网络（Message Passing Neural Network，MPNN）

![19fa0ec38c25d4584b0b10ca65795869.png](../../_resources/c2ebd71d4bc344e59e5d28abf496db9d.png)

## 4.3 前馈神经网络
前馈神经网络（Feedforward Neural Network，FNN），也称多层感知器（Multi-Layer Perceptron，MLP）
- MLP叫法不合理，FNN由多层Logistic回归模型（连续）组成，而非多层感知器（非连续）
![52b06c39169b6a0c3206d0aaa62d4e7f.png](../../_resources/d679ffef7e89410d8e59bdb282bfb5dc.png)
![b35906494af58ca7ce3a8d49312ab9d1.png](../../_resources/a97981117c22495f9bc41e33aba50a68.png)
$$
\begin{array}{l}
z^{(l)}=W^{(l)} a^{(l-1)}+b^{(l)} \\\\
a^{(l)}=f\_{l}\left(z^{(l)}\right)
\end{array}
$$
- 仿射变换（Affine Transformation，线性变化 + 平移） + 非线性变换
- 整个网络可以看作一个复合函数𝜙(𝒙; 𝑾, 𝒃)

### 4.3.1 通用近似定理
- FNN可以近似任何连续非线性函数
- 通用近似定理（Universal Approximation Theorem）
	![9f5f96a110ffa5411875972e7294133e.png](../../_resources/6e13ea82ba584adf8e3ecadfb38a179a.png)XML

### 4.3.2 应用到机器学习
多层前馈神经网络也可以看成是一种特征转换方法，其输出 𝜙(𝒙) 作为分类器的输入进行分类

### 4.3.3 参数学习
梯度下降法需要计算损失函数对参数的偏导数，链式法则逐一求偏导比较低效，常用方向传播算法。


## 4.4 反向传播算法
第 𝑙 层神经元的误差项：
$$
\delta^{(l)} \triangleq \frac{\partial \mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})}{\partial \boldsymbol{z}^{(l)}} \in \mathbb{R}^{M\_{l}}
$$
误差项𝛿(𝑙) 也间接反映了不同神经元对网络能力的贡献程度，从而比较好地解决了贡献度分配问题（Credit Assignment Problem，CAP）

$$
\begin{aligned}
\delta^{(l)} & \triangleq \frac{\partial \mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})}{\partial \boldsymbol{z}^{(l)}} \\\\
&=\frac{\partial \boldsymbol{a}^{(l)}}{\partial \boldsymbol{z}^{(l)}} \cdot \cdot \cdot \frac{\partial \boldsymbol{z}^{(l+1)}}{\partial \boldsymbol{a}^{(l)}} \cdot {\frac{\partial \mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})}{\partial \boldsymbol{z}^{(l+1)}}} \\\\
&={\operatorname{diag}\left(f\_{l}^{\prime}\left(\boldsymbol{z}^{(l)}\right)\right)}\left(\boldsymbol{W}^{(l+1)}\right)^{\mathrm{T}} \cdot {\delta}^{(l+1)} \\\\
&=f\_{l}^{\prime}\left(\boldsymbol{z}^{(l)}\right) \odot\left(\left(\boldsymbol{W}^{(l+1)}\right)^{\top} \delta^{(l+1)}\right) \quad \in \mathbb{R}^{M}
\end{aligned}
$$
误差的反向传播（BackPropagation，BP）：l层误差通过l+1层误差计算得到

BP算法内涵：l层一个神经元误差项是所有与该神经元相连的l+1层神经元的误差项的权重和，再乘上该神经元激活函数的梯度
$$
\begin{aligned}
\frac{\partial \mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})}{\partial w\_{i j}^{(l)}} &=\llbracket\_{i}\left(a\_{j}^{(l-1)}\right) \delta^{(l)} \\\\
&=\left[0, \cdots, a\_{j}^{(l-1)}, \cdots, 0\right]\left[\delta\_{1}^{(l)}, \cdots, \delta\_{i}^{(l)}, \cdots, \delta\_{M\_{l}}^{(l)}\right]^{\top} \\\\
&=\delta\_{i}^{(l)} a\_{j}^{(l-1)}
\end{aligned}
$$

进一步：
$$
\left[\frac{\partial \mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})}{\partial \boldsymbol{W}^{(l)}}\right]\_{i j}=\left[\delta^{(l)}\left(\boldsymbol{a}^{(l-1)}\right)^{\top}\right]\_{i j}
$$

权重梯度：
$$
\frac{\partial \mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})}{\partial \boldsymbol{W}^{(l)}}=\delta^{(l)}\left(\boldsymbol{a}^{(l-1)}\right)^{\top} \quad \in \mathbb{R}^{M\_{l} \times M\_{l-1}}
$$

偏置梯度：
$$
\frac{\partial \mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})}{\partial \boldsymbol{b}^{(l)}}=\delta^{(l)} \in \mathbb{R}^{M\_{l}}
$$

BP训练FNN过程：
1. 前馈计算每一层的净输入 𝒛(𝑙) 和激活值 𝒂(𝑙)，直到最后一层；
2. 反向传播计算每一层的误差项 𝛿(𝑙)；
3. 计算每一层参数的偏导数，并更新参数．

![b1979b630804acffa86c63beca242ca2.png](../../_resources/5e1471e2964a42ff83cf0c89fef16a4f.png)

## 4.5 自动梯度计算
自动计算梯度的方法三类：
1. 数值微分
2. 符号微分
3. 自动微分

### 4.5.1 数值微分（Numerical Differentiation）
$$
f^{\prime}(x)=\lim \_{\Delta x \rightarrow 0} \frac{f(x+\Delta x)-f(x)}{\Delta x}
$$

对x加上扰动 Δ𝑥，通过上述定义直接求解。
- 找到一个合适的扰动 Δ𝑥 十分困难
	- Δ𝑥 过小，会引起数值计算问题，比如舍入误差
	- Δ𝑥 过大，会增加截断误差，使得导数计算不准确
	- 实用性差
- 实际常用：
$$
f^{\prime}(x)=\lim \_{\Delta x \rightarrow 0} \frac{f(x+\Delta x)-f(x-\Delta x)}{2 \Delta x}
$$
- 数值微分计算复杂度较高 $O\left(N^{2}\right)$

### 4.5.2 符号微分（Symbolic Differentiation）
一种基于符号计算的自动求导方法
> 符号计算也叫代数计算（对应数值计算），是指用计算机来处理带有变量的数学表达式

优点：
1. 可以在编译时计算梯度的属性表示，并利用符号计算方法优化
2. 与平台无关，可以在CPU、GPU上运行

缺点：
1. 编译时间长，尤其是循环
2. 符号微分需要专门语言表示数学表达式，并预先声明变量（符号）
3. 调试困难

### 4.5.3 自动微分（Automatic Differentiation，AD）
一种可以对一个（程序）函数进行计算导数的方法
- 符号微分的处理对象是数学表达式，而自动微分的处理对象是一个函数或一段程序
- 思想：链式法则计算复合函数梯度
- 计算图（Computational Graph）：数学运算的图形化表示
![a5b776a74918b4258a5d227e5c7b21c4.png](../../_resources/470a58cee0174ca7810a89249fc8e698.png)
$$
\begin{aligned}
\frac{\partial f(x ; w, b)}{\partial w} &=\frac{\partial f(x ; w, b)}{\partial h\_{6}} \frac{\partial h\_{6}}{\partial h\_{5}} \frac{\partial h\_{5}}{\partial h\_{4}} \frac{\partial h\_{4}}{\partial h\_{3}} \frac{\partial h\_{3}}{\partial h\_{2}} \frac{\partial h\_{2}}{\partial h\_{1}} \frac{\partial h\_{1}}{\partial w} \\\\
\frac{\partial f(x ; w, b)}{\partial b} &=\frac{\partial f(x ; w, b)}{\partial h\_{6}} \frac{\partial h\_{6}}{\partial h\_{5}} \frac{\partial h\_{5}}{\partial h\_{4}} \frac{\partial h\_{4}}{\partial h\_{3}} \frac{\partial h\_{3}}{\partial h\_{2}} \frac{\partial h\_{2}}{\partial b}
\end{aligned}
$$
- 分为前向模式和反向模式，反向模式和反向传播的计算梯度的方式相同
	- 自下而上，反向模式遍历每个输出，每次自动微分都求出所有相关节点的一个自变量分量 $x\_{i}$ 的导数 $\frac{d \cdot}{d x\_{i}}$
	- 自上而下，前向模式遍历每个输入，每次自动求导都是求出函数 $y\_{i}(x)$ 关于所有相关节点的导数 $\frac{d y\_{i}}{d \cdot}$

- 计算图构建方式
	- 静态计算图（Static Computational Graph）：编译时构建计算图，构建时可以优化，并行能力强
	- 动态计算图（Dynamic Computational Graph）：运行时构建集散图，灵活性高
- 符号微分和自动微分
![19b82ad7241055bea42835b51807b852.png](../../_resources/22500828ab814d4ea8179b1360647f0f.png)


## 4.6 优化问题
### 4.6.1 非凸优化问题
![c26339589097ba278354bd00149f4111.png](../../_resources/c833e52307d849a38935b0e4ed3441a9.png)

### 4.6.2 梯度消失问题
误差反向传播的迭代公式为
$$
\delta^{(l)}=f\_{l}^{\prime}\left(z^{(l)}\right) \odot\left(W^{(l+1)}\right)^{\top} \delta^{(l+1)}
$$
每一层都乘以激活函数导数，使用Sigmoid型函数时，其导数：
![084e0fa7ea65282d41a08fbc7bcedc34.png](../../_resources/6464c9e84f424eaa8c50c5751c68ae31.png)

梯度消失问题（Vanishing Gradient Problem）/梯度弥散问题：Sigmoid型函数导数值域小，两端饱和区导数更是接近于0。这样，误差在传递中不断衰减，网络很深时，梯度过小甚至消失，使得网络很难训练。

解决：使用导数比较大的激活函数，比如 ReLU 等


## 4.7 总结
![988c424ef4fc3d197dcc26862b425928.png](../../_resources/4d8409a2b29d4af48ec23ea9eaac2131.png)

## 习题
#### 习题 4-1 对于一个神经元 𝜎(𝒘T𝒙 + 𝑏)，并使用梯度下降优化参数 𝒘 时，如果输入𝒙 恒大于 0，其收敛速度会比零均值化的输入更慢
Sigmoid型函数在零点处导数最大，收敛最快。


#### 习题 4-2 试设计一个前馈神经网络来解决 XOR 问题，要求该前馈神经网络具有两个隐藏神经元和一个输出神经元，并使用 ReLU 作为激活函数．
```py
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(units=2, activation='relu', input_dim=2))
model.add(Dense(units=1, activation='sigmoid'))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=sgd)

print(model.summary())
print(model.get_weights())
model.fit(X, y, epochs=1000, batch_size=1)
print(model.get_weights())
print(model.predict(X, batch_size=1))
```
随缘，经常遇到死亡ReLU，换成softplus就好了

#### 习题 4-3 试举例说明“死亡 ReLU 问题”，并提出解决方法．
BP中，学习率比较大，对较大的梯度，ReLU神经元更新后的权重和偏置为负，下一轮正向传播时Z为负，ReLU输出a为0，后续反向传播时参数永远不再更新。

解决：Leaky ReLU、PReLU、ELU、Softplus

#### 习题 4-5 如果限制一个神经网络的总神经元数量（不考虑输入层）为 𝑁 + 1，输入层大小为 𝑀，输出层大小为 1，隐藏层的层数为 𝐿，每个隐藏层的神经元数量为𝑁/𝐿 ，试分析参数数量和隐藏层层数 𝐿 的关系
b数量：$N+1$
w数量：$M \times \frac{N}{L}+(L-1)(\frac{N}{L})^2+\frac{N}{L}$

#### 习题 4-7 为什么在神经网络模型的结构化风险函数中不对偏置 𝒃 进行正则化？
正则化降低模型空间/复杂性，防止过拟合。w过大对输入数据敏感，b偏置与特征无关，对所有数据都相同。

#### 习题 4-8 为什么在用反向传播算法进行参数学习时要采用随机参数初始化的方式而不是直接令 𝑾 = 0, 𝒃 = 0？
对称权重现象：会使每层中的参数相同，不同结点无法学习不同特征

#### 习题 4-9 梯度消失问题是否可以通过增加学习率来缓解？
可以缓解，不能解决，反倒可能使得深层梯度爆炸
