# 《神经网络与深度学习》第3章 - 线性模型



# ch3 线性模型
线性模型：通过样本特征的线性组合来进行预测。其线性组合函数为：
$$
\begin{aligned}
f(\boldsymbol{x} ; \boldsymbol{w}) &=w_{1} x_{1}+w_{2} x_{2}+\cdots+w_{D} x_{D}+b \\\\
&=\boldsymbol{w}^{\top} \boldsymbol{x}+b
\end{aligned}
$$

1. 线性回归：直接使用 $y=f(\boldsymbol{x} ; \boldsymbol{w})$ 来预测输出目标
2. 分类问题：离散便签，需要引入非线性决策函数（Decision Function） $g(\cdot)$ 预测输出目标： $y=g(f(\boldsymbol{x} ; \boldsymbol{w}))$
	- $f(\boldsymbol{x} ; \boldsymbol{w})$ 也称判别函数（Discriminant Function）

四种线性分类模型（主要区别：不同损失函数）
1. Logistic 回归
2. Softmax 回归
3. 感知器
4. 支持向量机


## 3.1 线性判别函数和决策边界
线性分类模型（Linear Classification Model）或线性分类器（Linear Classifier）：一个或多个线性判别函数 + 非线性决策函数

### 3.1.1 二分类
分割超平面（Hyperplane） / 决策边界（Decision Boundary） / 决策平面（Decision Surface）： $f(\boldsymbol{x} ; \boldsymbol{w})=0$ 的点组成的平面

特征空间中每个样本点到决策平面的有向距离（Signed Distance）：
$$
\gamma=\frac{f(\boldsymbol{x} ; \boldsymbol{w})}{\|\boldsymbol{w}\|}
$$

![3108e711ea94a00da916e6f1d13bcbd8.png](../../_resources/3820a4e403bc4102a1401762b3ea8701.png)

线性模型学习目标是尽量满足：
$$
y^{(n)} f\left(\boldsymbol{x}^{(n)} ; \boldsymbol{w}^{*}\right)>0, \quad \forall n \in[1, N]
$$

两类线性可分：训练集的所有样本都满足上式。

学习参数 $\boldsymbol{w}$，需要定义合适的损失函数和优化方法。

直接采用0-1损失函数：
$$
y^{(n)} f\left(\boldsymbol{x}^{(n)} ; \boldsymbol{w}^{*}\right)>0, \quad \forall n \in[1, N]
$$
- 存在问题：$\boldsymbol{w}$ 导数为0，无法优化。

### 3.1.2 多分类

判别函数：
1. 一对其余： $C$ 个二分类函数
2. 一对一：$C(C-1)/2$ 个二分类函数
3. argmax：改进的“一对其余”，$C$ 个判别函数

$$
y=\underset{c=1}{\arg \max } f_{c}\left(\boldsymbol{x} ; \boldsymbol{w}_{c}\right) .
$$

“一对其余”和“一对一”存在难以确定区域：
![dd80cae95bb2424e661898ac687c4679.png](../../_resources/788b87d745544b6ba1015d136ca42cdf.png)

多类线性可分：对训练集，每一类均存在判别函数使得该类下所有样本的当前类判别函数最大。

## 3.2 Logistic 回归
Logistic 回归（Logistic Regression，LR）：二分类

引入非线性函数 g 预测类别后验概率：
$$
p(y=1 \mid \boldsymbol{x})=g(f(\boldsymbol{x} ; \boldsymbol{w}))
$$

$g(\cdot)$ 称为激活函数（Activation Funtion）：将线性函数值域挤压到 $(0, 1)$ 之间，表示概率。
- $g(\cdot)$ 的逆函数 $g^{-1}(\cdot)$ 称为联系函数（Link Function）

Logistic 回归使用 Logistic 函数作为激活函数。标签y=1的后验概率：
$$
\begin{aligned}
p(y=1 \mid \boldsymbol{x}) &=\sigma\left(\boldsymbol{w}^{\top} \boldsymbol{x}\right) \\\\
& \triangleq \frac{1}{1+\exp \left(-\boldsymbol{w}^{\top} \boldsymbol{x}\right)},
\end{aligned}
$$

变换得到： 

$$
\begin{aligned}
\boldsymbol{w}^{\top} \boldsymbol{x} &=\log \frac{p(y=1 \mid \boldsymbol{x})}{1-p(y=1 \mid \boldsymbol{x})} \\\\
&=\log \frac{p(y=1 \mid \boldsymbol{x})}{p(y=0 \mid \boldsymbol{x})},
\end{aligned}
$$

其中 $\frac{p(y=1 \mid \boldsymbol{x})}{p(y=0 \mid \boldsymbol{x})}$ 称为几率（Odds），几率的对数称为对数几率（Log Odds，或Logit）

![17dced3f48881b98ed099d685b2b0405.png](../../_resources/49f4530a117b4dc09f8bbe4b4cdfacd1.png)

**参数学习**
损失函数：交叉熵
风险函数：

$$
\begin{aligned}
\mathcal{R}(\boldsymbol{w})=&-\frac{1}{N} \sum_{n=1}^{1 \mathrm{~N} }\left(p_{r}\left(y^{(n)}=1 \mid \boldsymbol{x}^{(n)}\right) \log \hat{y}^{(n)}+p_{r}\left(y^{(n)}=0 \mid \boldsymbol{x}^{(n)}\right) \log \left(1-\hat{y}^{(n)}\right)\right) \\\\
&=-\frac{1}{N} \sum_{n=1}^{N}\left(y^{(n)} \log \hat{y}^{(n)}+\left(1-y^{(n)}\right) \log \left(1-\hat{y}^{(n)}\right)\right) .
\end{aligned}
$$

求导：

$$
\begin{aligned}
\frac{\partial \mathcal{R}(\boldsymbol{w})}{\partial \boldsymbol{w} } &=-\frac{1}{N} \sum_{n=1}^{N}\left(y^{(n)} \frac{\hat{y}^{(n)}\left(1-\hat{y}^{(n)}\right)}{\hat{y}^{(n)} } \boldsymbol{x}^{(n)}-\left(1-y^{(n)}\right) \frac{\hat{y}^{(n)}\left(1-\hat{y}^{(n)}\right)}{1-\hat{y}^{(n)} } \boldsymbol{x}^{(n)}\right) \\\\
&=-\frac{1}{N} \sum_{n=1}^{N}\left(y^{(n)}\left(1-\hat{y}^{(n)}\right) \boldsymbol{x}^{(n)}-\left(1-y^{(n)}\right) \hat{y}^{(n)} \boldsymbol{x}^{(n)}\right) \\\\
&=-\frac{1}{N} \sum_{n=1}^{N} \boldsymbol{x}^{(n)}\left(y^{(n)}-\hat{y}^{(n)}\right)
\end{aligned}
$$

梯度下降法参数更新：
$$
\boldsymbol{w}_{t+1} \leftarrow \boldsymbol{w}_{t}+\alpha \frac{1}{N} \sum_{n=1}^{N} \boldsymbol{x}^{(n)}\left(y^{(n)}-\hat{y}_{\boldsymbol{w}_{t} }^{(n)}\right)
$$

因为风险函数 $\mathcal{R}(\boldsymbol{w})$ 是关于参数的连续可导凸函数，所以还可以用更高阶的优化方法（如牛顿法）来优化。


## 3.3 Softmax 回归
Sotfmax回归（Softmax Regression）：即多项或多类的 Logistic 回归。

预测类别：

$$
\begin{aligned}
p(y=c \mid \boldsymbol{x}) &=\operatorname{softmax} (\boldsymbol{w}_{c}^{\top} \boldsymbol{x}) \\\\
&=\frac{\exp \left(\boldsymbol{w}\_{c}^{\top} \boldsymbol{x}\right)}{\sum\_{c^{\prime}=1}^{C} \exp \left(\boldsymbol{w}\_{c^{\prime} }^{\top} \boldsymbol{x}\right)}
\end{aligned}
$$

向量化表示：
$$
\begin{aligned}
\hat{\boldsymbol{y} } &=\operatorname{softmax}\left(\boldsymbol{W}^{\top} \boldsymbol{x}\right) \\\\
&=\frac{\exp \left(\boldsymbol{W}^{\top} \boldsymbol{x}\right)}{\mathbf{1}_{C}^{\mathrm{T} } \exp \left(\boldsymbol{W}^{\top} \boldsymbol{x}\right)}
\end{aligned}
$$

决策函数：
$$
\begin{aligned}
\hat{y} &=\underset{c=1}{\arg \max } p(y=c \mid \boldsymbol{x}) \\\\
&=\underset{c=1}{\arg \max } \boldsymbol{w}_{c}^{\top} \boldsymbol{x} .
\end{aligned}
$$

风险函数：
$$
\begin{aligned}
\mathcal{R}(\boldsymbol{W}) &=-\frac{1}{N} \sum_{n=1}^{N} \sum_{c=1}^{C} \boldsymbol{y}_{c}^{(n)} \log \hat{\boldsymbol{y} }_{c}^{(n)} \\\\
&=-\frac{1}{N} \sum_{n=1}^{N}\left(\boldsymbol{y}^{(n)}\right)^{\mathrm{T} } \log \hat{\boldsymbol{y} }^{(n)}
\end{aligned}
$$

Softmax 函数求导：

$$
\begin{aligned}
&\frac{\partial \operatorname{softmax}(\boldsymbol{x})}{\partial \boldsymbol{x} }=\frac{\partial\left(\frac{\exp (x)}{1_{K}^{\top} \exp (x)}\right)}{\partial \boldsymbol{x} } \\\\
&=\frac{1}{\mathbf{1}_{K}^{\top} \exp (\boldsymbol{x})} \frac{\partial \exp (\boldsymbol{x})}{\partial \boldsymbol{x} }+\frac{\partial\left(\frac{1}{1_{K}^{\mathrm{T} \exp (x)} }\right)}{\partial \boldsymbol{x} }(\exp (\boldsymbol{x}))^{\top} \\\\
&=\frac{\operatorname{diag}(\exp (\boldsymbol{x}))}{\mathbf{1}_{K}^{\top} \exp (\boldsymbol{x})}-\left(\frac{1}{\left(\mathbf{1}_{K}^{\mathrm{T} } \exp (\boldsymbol{x})\right)^{2} }\right) \frac{\partial\left(\mathbf{1}_{K}^{\top} \exp (\boldsymbol{x})\right)}{\partial \boldsymbol{x} }(\exp (\boldsymbol{x}))^{\top} \\\\
&=\frac{\operatorname{diag}(\exp (\boldsymbol{x}))}{\mathbf{1}_{K}^{\top} \exp (\boldsymbol{x})}-\left(\frac{1}{\left(\mathbf{1}_{K}^{\mathrm{T} } \exp (\boldsymbol{x})\right)^{2} }\right) \operatorname{diag}(\exp (\boldsymbol{x})) \mathbf{1}_{K}(\exp (\boldsymbol{x}))^{\top} \\\\
&=\frac{\operatorname{diag}(\exp (\boldsymbol{x}))}{\mathbf{1}_{K}^{\top} \exp (\boldsymbol{x})}-\left(\frac{1}{\left(\mathbf{1}_{K}^{\top} \exp (\boldsymbol{x})\right)^{2} }\right) \exp (\boldsymbol{x})(\exp (\boldsymbol{x}))^{\top} \\\\
&=\operatorname{diag}\left(\frac{\exp (\boldsymbol{x})}{\mathbf{1}_{K}^{\mathrm{T} } \exp (\boldsymbol{x})}\right)-\frac{\exp (\boldsymbol{x})}{\mathbf{1}_{K}^{\top} \exp (\boldsymbol{x})} \frac{(\exp (\boldsymbol{x}))^{\mathrm{T} }}{\mathbf{1}_{K}^{\mathrm{T} } \exp (\boldsymbol{x})} \\\\
&=\operatorname{diag}(\operatorname{softmax}(\boldsymbol{x}))-\operatorname{softmax}(\boldsymbol{x}) \operatorname{softmax}(\boldsymbol{x})^{\top} .
\end{aligned}
$$

即若 $y=\operatorname{softmax}(z)$，则  $\frac{\partial y}{\partial z}=\operatorname{diag}(y)-y y^{\top}$

所以风险函数求梯度为：
$$
\frac{\partial \mathcal{R}(\boldsymbol{W})}{\partial \boldsymbol{W} }=-\frac{1}{N} \sum_{n=1}^{N} \boldsymbol{x}^{(n)}\left(\boldsymbol{y}^{(n)}-\hat{\boldsymbol{y} }^{(n)}\right)^{\top}
$$

梯度下降法更新：
$$
\boldsymbol{W}\_{t+1} \leftarrow \boldsymbol{W}\_{t}+\alpha\left(\frac{1}{N} \sum_{n=1}^{N} \boldsymbol{x}^{(n)}\left(\boldsymbol{y}^{(n)}-\hat{\boldsymbol{y} }\_{W_{t} }^{(n)}\right)^{\top}\right)
$$


## 3.4 感知器
### 3.4.1 参数学习
感知器（Perceptron）

分类准则：
$$
\hat{y}=\operatorname{sgn}\left(\boldsymbol{w}^{\top} \boldsymbol{x}\right)
$$

学习目标，找到参数使得：
$$
y^{(n)} \boldsymbol{w}^{* \top} \boldsymbol{x}^{(n)}>0, \quad \forall n \in\{1, \cdots, N\}
$$

感知器的学习算法：错误驱动的在线学习算法 [Rosenblatt, 1958]，每错分一个样本，就用该样本更新权重：

$$
\boldsymbol{w} \leftarrow \boldsymbol{w}+y \boldsymbol{x}
$$

损失函数：

$$
\mathcal{L}(\boldsymbol{w} ; \boldsymbol{x}, y)=\max \left(0,-y \boldsymbol{w}^{\top} \boldsymbol{x}\right)
$$

梯度更新：

$$
\frac{\partial \mathcal{L}(\boldsymbol{w} ; \boldsymbol{x}, y)}{\partial \boldsymbol{w} }=\left\{\begin{array}{lll}
0 & \text { if } & y \boldsymbol{w}^{\top} \boldsymbol{x}>0 \\\\
-y \boldsymbol{x} & \text { if } & y \boldsymbol{w}^{\top} \boldsymbol{x}<0
\end{array}\right.
$$

感知器参数学习过程：
（黑色：当前权重向量，红色虚线：更新方向）
![85a655694944a21c12cce1a384f4a2d2.png](../../_resources/db442afde21b44d3ac2276634e6114cb.png)

### 3.4.2 感知器的收敛性

1. 在数据集线性可分时，感知器可以找到一个超平面把两类数据分开，但并不能保证其泛化能力．
2. 感知器对样本顺序比较敏感．每次迭代的顺序不一致时，找到的分割超平面也往往不一致．
3. 如果训练集不是线性可分的，就永远不会收敛．

### 3.4.3 参数平均感知器
投票感知器（Voted Perceptron）：感知器收单个样本影响大，为提高鲁棒性和泛化能力，将所有 K 个权重用置信系数加权平均起来，投票决定结果。


置信系数 $c_{k}$ 设置为当前更新权重后直到下一次更新的迭代次数。则投票感知器为：

$$
\hat{y}=\operatorname{sgn}\left(\sum_{k=1}^{K} c_{k} \operatorname{sgn}\left(\boldsymbol{w}_{k}^{\top} \boldsymbol{x}\right)\right)
$$


平均感知器（Averaged Perceptron）[Collins, 2002]：

$$
\begin{aligned}
\hat{y} &=\operatorname{sgn}\left(\frac{1}{T} \sum_{k=1}^{K} c_{k}\left(\boldsymbol{w}_{k}^{\top} \boldsymbol{x}\right)\right) \\\\
&=\operatorname{sgn}\left(\frac{1}{T}\left(\sum_{k=1}^{K} c_{k} \boldsymbol{w}_{k}\right)^{\top} \boldsymbol{x}\right) \\\\
&=\operatorname{sgn}\left(\left(\frac{1}{T} \sum_{t=1}^{T} \boldsymbol{w}_{t}\right)^{\top} \boldsymbol{x}\right) \\\\
&=\operatorname{sgn}\left(\overline{\boldsymbol{w} }^{\top} \boldsymbol{x}\right)
\end{aligned}
$$

### 3.4.4 扩展到多分类


## 3.5 支持向量机

支持向量机（Support Vector Machine，SVM）：经典二分类算法，找到的超平面具有更好的鲁棒性。

$$
y_{n} \in\{+1,-1\}
$$

超平面：
$$
\boldsymbol{w}^{\top} \boldsymbol{x}+b=0
$$

每个样本到分割超平面的距离：

$$
\gamma^{(n)}=\frac{\left|\boldsymbol{w}^{\top} \boldsymbol{x}^{(n)}+b\right|}{\|\boldsymbol{w}\|}=\frac{y^{(n)}\left(\boldsymbol{w}^{\top} \boldsymbol{x}^{(n)}+b\right)}{\|\boldsymbol{w}\|}
$$

间隔（Margin）：数据集中所有样本到分割超平面的最短距离：

$$
\gamma=\min _{n} \gamma^{(n)}
$$

SVM的目标：

$$
\begin{array}{ll}
\max _{\boldsymbol{w}, b} & \gamma \\\\
\text { s.t. } & \frac{y^{(n)}\left(\boldsymbol{w}^{\top} \boldsymbol{x}^{(n)}+b\right)}{\|\boldsymbol{w}\|} \geq \gamma, \forall n \in\{1, \cdots, N\} .
\end{array}
$$

由于 $\boldsymbol{w}$ 和 $b$ 可以同时缩放不改变间隔，可以限制 $\|\boldsymbol{w}\| \cdot \gamma=1$ ，则上式等价于：

$$
\begin{aligned}
\max _{\boldsymbol{w}, b} & \frac{1}{\|\boldsymbol{w}\|^{2} } \\\\
\text { s.t. } & y^{(n)}\left(\boldsymbol{w}^{\top} \boldsymbol{x}^{(n)}+b\right) \geq 1, \forall n \in\{1, \cdots, N\}
\end{aligned}
$$

支持向量（Suport Vector）： 满足 $y^{(n)}\left(\boldsymbol{w}^{\top} \boldsymbol{x}^{(n)}+b\right)=1$ 的样本点。

![a48c71f1fa2db1a93cb440d9a3f64377.png](../../_resources/5e42dd9a13b14ea681cb9cee1f719291.png)

### 3.5.1 参数学习
将目标函数写成凸优化问题，采用拉格朗日乘数法，并得到拉格朗日对偶函数，采用序列最小优化（Sequential Minimal Optimization, SMO）等高效算法进行优化。

最优参数的SVM决策函数：

$$
\begin{aligned}
f(\boldsymbol{x}) &=\operatorname{sgn}\left(\boldsymbol{w}^{* \top} \boldsymbol{x}+b^{*}\right) \\\\
&=\operatorname{sgn}\left(\sum_{n=1}^{N} \lambda_{n}^{*} y^{(n)}\left(\boldsymbol{x}^{(n)}\right)^{\top} \boldsymbol{x}+b^{*}\right)
\end{aligned}
$$

### 3.5.2 核函数
SVM可以使用核函数（Kernel Function）隐式地将样本从延时特征空间映射到更高维的空间，解决原始特征空间中线性不可分的问题。

则决策函数为：
$$
\begin{aligned}
f(\boldsymbol{x}) &=\operatorname{sgn}\left(\boldsymbol{w}^{\*} \boldsymbol{\phi}(\boldsymbol{x})+b^{\*}\right) \\\\
&=\operatorname{sgn}\left(\sum\_{n=1}^{N} \lambda\_{n}^{\*} y^{(n)} k\left(\boldsymbol{x}^{(n)}, \boldsymbol{x}\right)+b^{\*}\right)
\end{aligned}
$$

$k(\boldsymbol{x}, \boldsymbol{z})=\phi(\boldsymbol{x})^{\top} \phi(\boldsymbol{z})$ 为核函数，通常不需要显示给出 $\phi(\boldsymbol{x})$ 的具体形式，可以通过核技巧（Kernel Trick）来构造，比如构造：

$$
k(\boldsymbol{x}, \boldsymbol{z})=\left(1+\boldsymbol{x}^{\top} \boldsymbol{z}\right)^{2}=\phi(\boldsymbol{x})^{\top} \phi(\boldsymbol{z})
$$

来隐式地计算 $\boldsymbol{x, z}$ 在特征空间 $\phi$ 中的内积，其中：

$$
\phi(\boldsymbol{x})=\left[1, \sqrt{2} x_{1}, \sqrt{2} x_{2}, \sqrt{2} x_{1} x_{2}, x_{1}^{2}, x_{2}^{2}\right]^{\top}
$$

### 3.5.3 软间隔
当线性不可分时，为了容忍部分不满足约束的样本，引入松弛变量（Slack Variable）$\xi$，将优化问题变为

$$
\begin{array}{ll}
\min \_{\boldsymbol{w}, b} & \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum\_{n=1}^{N} \xi\_{n} \\\\
\text { s.t. } & 1-y^{(n)}\left(\boldsymbol{w}^{\top} \boldsymbol{x}^{(n)}+b\right)-\xi\_{n} \leq 0, \quad \forall n \in\{1, \cdots, N\} \\\\
& \xi\_{n} \geq 0, \quad \forall n \in\{1, \cdots, N\}
\end{array}
$$

参数 $C > 0$ 控制间隔和松弛变量之间和平衡，引入松弛变量的间隔称为软间隔（Soft Margin）。

上式也可以表示为 **经验风险 + 正则化项** 的形式：

$$
\min \_{\boldsymbol{w}, b} \quad \sum\_{n=1}^{N} \max \left(0,1-y^{(n)}\left(\boldsymbol{w}^{\top} \boldsymbol{x}^{(n)}+b\right)\right)+\frac{1}{2 C}\|\boldsymbol{w}\|^{2}
$$

前面一项可以看作 Hinge损失函数，后一项看作正则项，$\frac{1}{c}$ 为正则化系数。


## 3.6 损失函数对比
统一定义标签：
$$
y \in\{+1,-1\}
$$

决策函数：
$$
f(\boldsymbol{x} ; \boldsymbol{w})=\boldsymbol{w}^{\top} \boldsymbol{x}+b
$$

![a67dbd71aaa5550b002c5b0d5c34f525.png](../../_resources/57baee45892347eab9241cc1b47b588c.png)

> 平方损失函数其实也可以用于分类问题的 loss 函数，但本质上等同于误差服从高斯分布假设下的极大似然估计，而分类问题大部分时候不服从高斯分布。
> 直观上理解，标签之间的距离没有意义，预测值和标签之间的距离不能反应问题优化程度。

![17ec69c0a09ff745f7396afef15ee28a.png](../../_resources/ec087fd86375437da75ff6320802cdfe.png)


## 习题

#### 习题 3-1 证明在两类线性分类中，权重向量𝒘 与决策平面正交．

判别函数：
$$
f(x)=w^{T} * x+w_{0}
$$

决策平面：
$$
f(x)=w^{T} * x+w_{0}=0
$$

 $w^T$ 平面法向量，任取平面两点构成线段均垂直于法向量。

### 习题 3-2 在线性空间中，证明一个点 𝒙 到平面 𝑓(𝒙; 𝒘) = 𝒘T𝒙 + 𝑏 = 0 的距离为 |𝑓(𝒙; 𝒘)|/‖𝒘‖

点到面距离计算：任取平面一点与该点构成直线 $AB$ ，距离即是 $AB$ 在法向量 $\boldsymbol{w}$ 上的投影。

$$
|\mathrm{AC}|=\left|\overrightarrow{\mathrm{AB} } \cdot \frac{\overrightarrow{\mathrm{n} }}{|\overrightarrow{\mathrm{n} }|}\right|
$$

点积展开计算、消去0项即可。


