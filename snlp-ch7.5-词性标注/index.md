# 《统计自然语言处理》第7.5章 - 词性标注


## 7.5 词性标注
### 7.5.1 概述
- 词性（part-of-speech）是词汇基本的语法属性，也称词类
- 主要难点
	1. 汉语缺乏词形态变化，不能从形态变化判别词类
	2. 常用词兼类现象严重
		- 兼类词：有多种词性的词
	3. 研究者主观原因：词性划分目的和标准不统一


### 7.5.2 基于统计模型的词性标注
- 基于统计模型（n-gram、一阶马尔科夫）的词性标注方法
	- 代表：1983年I.Marshall建立的LOB语料库词性标注系统CLAWS（Constituent-Likelihood Automatic Word-tagging System）
- HMM的词性标注：参数估计
	1. 随机初始参数：过于缺乏限制
	2. 利用词典信息约束模型参数（Jelinek方法）
		- “词汇-词汇标记”对没有在词典中，令该词生成概率为0，否则为可能被标记的词性个数的倒数
		- $b\_{j . l}=\frac{b\_{j . l}^{*} C\left(w^{l}\right)}{\sum\_{w^{m} } b\_{j . m}^{*} C\left(w^{m}\right)}$
		- $b\_{j . i}^{*}=\left\{\begin{array}{ll}0, & \text { 如果 } t^{j} \text { 不是词 } w^{l} \text { 所允许的词性 } \\\\ \frac{1}{T\left(w^{**strong text**l}\right)}, & \text { 其他情况 }\end{array}\right.$
		- 等价于用最大似然估计来估算概率$P(w^k \mid t^i)$以初始化HMM，并假设每个词与其每个可能的词性标记出现的概率相等
	3. 词汇划分等价类，以类为单位进行参数估计，大大减少了参数个数
		- 元词（metawords）$u_L$：所有具有相同可能词性的词汇划分为一组
		- 类似Jelinek方法处理元词：$b\_{j . l}=\frac{b\_{j . L}^{*} C\left(u\_{L}\right)}{\sum\_{u\_{L}^{\prime} } b\_{j . L^{\prime} }^{*} C\left(u\_{L^{\prime} }\right)}$
		- $b\_{j . L}^{*}=\left\{\begin{array}{ll}0, & j \notin L \\\\ \frac{1}{L}, & \text { 否则 }\end{array}\right.$
- HMM训练：前向后向算法
- 模型参数对训练语料的适应性问题


### 7.5.3 基于规则的词性标注
- 按兼类词搭配关系和上下文语境建造词类消歧规则
- 早期：人工构造
- 语料发展：基于机器学习的规则自动提取
- 基于转换的错误驱动的（transformation-based and error-driven）学习方法
	![854384c41492cef39dfc8bf1b8dea46c.png](../../_resources/8785e3413f85461e8236524dc7ba71eb.png)
	- 劣势：学习时间过长
	- 改进：[周明等, 1998]每次迭代只调整受到影响的小部分转换规则，而不需要搜索所有转换规则
- [李晓黎等, 2000]数据采掘方法获取汉语词性标注

### 7.5.4 统计方法与规则方法结合
- 理性主义方法与经验主义方法相结合
- [周强，1995]规则与统计结合
	- 基本思想：
		1. 对初始标注结果，首先用规则排除常见、明显的歧义
		2. 再通过统计排歧，处理剩余多类词并进行未登录词的词性推断
		3. 最后人工校对
- [张民，1998]
	- [周强，1995]的方法规则作用于是非受限的，而且没有考虑统计的可信度，使规则与统计的作用域不明确
	- 引入置信区间，构造基于置信区间的评价函数，实现统计与规则并举
	- HMM，前向后向算法计算状态i的词w出现次数
		$F\left(t\_{i-1}, t\_{i}\right)=\sum\_{t\_{i-2} }\left[F\left(t\_{i-2}, t\_{i-1}\right) \times P\left(t\_{i} \mid t\_{i-1}, t\_{i-2}\right) \times P\left(w\_{i-1} \mid t\_{i-1}\right)\right]$
$B\left(t\_{i-1}, t\_{i}\right)=\sum\_{t\_{i+1} }\left[B\left(t\_{i}, t\_{i-1}\right) \times P\left(t\_{i-1} \mid t\_{i}, t\_{i-1}\right) \times P\left(w\_{i-1} \mid t\_{i-1}\right)\right]$
$\phi(w)\_{i}=\underset{t}{\operatorname{argmax} } \sum\_{t\_{i-1} }\left[F\left(t\_{i-1}, t\_{i}\right) \times B\left(t\_{i-1}, t\_{i}\right) \times P\left(w\_{i} \mid t\_{i}\right)\right]$
	- 假设兼类词w的候选词性为T1，T2，T3，其对应概率的真实值分别为p1，p2，p3，词w的词性为Ti（i＝1,2,3）时的出现次数为$\phi(w)\_{T_i}$
	- $\hat{p}\_{i}=\frac{\phi(w)\_{T\_{i} }}{\sum\_{j=1}^{3} \phi(w)\_{T\_{j} }}$
	- i=1，2，3时，记$\phi(w)\_{T_i}$为n1,n2,n3（令n1>n2>n3）
	- p1与p2相差小时，错误可能性较大
	- 阈值法：$p_1/p_2$是否大于阈值作为是否选择$T_1$也无法区别n1=300,n2=100与n1=3,n2=1的情况（前者显然更加可靠）
	- 可信度方法：根据n1，n2计算出的p1，p2只是p1，p2的近似值，我们必须估计出这种近似的误差，对p1/p2进行修正，然后再对修正后的p1/p2进行判别
- 可信度方法
	- 由于ln（p1/p2）比p1/p2更快地逼近正态分布［Dagan and Itai,1994］，因此，可应用单边区间估计方法计算ln（p1/p2）的置信区间。
	- 假设希望的错误率（desired error probability）（显著性水平）为α（0＜α＜1），则可信度为1-α，服从正态分布的随机变量X的置信区间为$Z\_{1-\alpha} \sqrt{\operatorname{vax} X}$
		- 置信系数$Z\_{1-\alpha}$
		- 标准差$\operatorname{vax} X=\operatorname{vax}\left[\ln \frac{\hat{p}\_{1} }{\hat{p}\_{2} }\right] \approx \frac{1}{n\_{1} }+\frac{1}{n\_{2} }$
		- 最终评价函数
		$\ln \frac{n\_{1} }{n\_{2} } \geqslant \theta+Z\_{1-\alpha} \quad \sqrt{\frac{1}{n\_{1} }+\frac{1}{n\_{2} }}$
- 对统计标注结果的筛选，只对那些被认为可疑的标注结果，才采用规则方法进行歧义消解，而不是对所有的情况都既使用统计方法又使用规则方法


### 7.5.5 词性标注中的生词处理方法
1. 规则：生词处理通常与词形分词和兼类词消解一起进行
2. 统计：通过合理处理词汇的发射频率解决

- 假设一个词汇序列W＝w1w2…wN对应的词性序列为T＝t1t2…tN，那么，词性标注问题就是求解使条件概率P（T|W）最大的T，即
	$\hat{T}=\underset{T}{\arg \max } P(T \mid W)=\underset{T}{\operatorname{argmax} } P(T) \times P(W \mid T)$
- 对于一阶马尔科夫过程；
	$\hat{T}=\underset{t\_{1} \cdot t\_{2}, \cdots, t\_{\mathrm{N} }}{\operatorname{argmax} } P\left(t\_{1}\right) P\left(w\_{1} \mid t\_{1}\right) \prod\_{i=2}^{N} P\left(t\_{i} \mid t\_{i-1}\right) P\left(w\_{i} \mid t\_{i}\right)$
	- $P(t_i \mid t\_{i-1})$为HMM中的状态转移概率，$P(W_i \mid t_i)$为词汇发射概率
- 假设词汇序列W中有生词$x_j$，其词性标注为$t_j$
	$\begin{aligned} \hat{T}=& \underset{t\_{1}, t\_{2}, \cdots, t\_{N} }{\operatorname{argmax} } P\left(t\_{1}\right) P\left(w\_{1} \mid t\_{1}\right) \\\\ & \cdots P\left(t\_{j} \mid t\_{j-1}\right) P\left(x\_{j} \mid t\_{j}\right) \prod\_{i=j-1}^{N} P\left(t\_{i} \mid t\_{i-1}\right) P\left(w\_{i} \mid t\_{i}\right) \end{aligned}$
- [赵铁军等，2001]将生词词汇发射概率赋值为1
	- 简单高效，但缺乏统计先验知识，正确率受到影响
- [张孝非等，2003]将词汇序列W加入训练集
	- HMM假设：$P\left(t\_{j} \mid x\_{j}\right) \approx \sum\_{k=1}^{M} P\left(t\_{k} \mid w\_{j-1}\right) P\left(t\_{j} \mid t\_{k}\right)$
	- Bayes公式计算发射频率：$P\left(x\_{j} \mid t\_{j}\right)=\frac{P\left(x\_{j}\right)}{P\left(t\_{j}\right)} \times P\left(t\_{j} \mid x\_{j}\right)$
	- 带入：$P\left(x\_{j} \mid t\_{j}\right) \approx \frac{P\left(x\_{j}\right)}{P\left(t\_{j}\right)} \times \sum\_{k=1}^{M} P\left(t\_{k} \mid w\_{j-1}\right) P\left(t\_{j} \mid t\_{k}\right)$
	- 最大似然估计：$\begin{aligned} P\left(x\_{j} \mid t\_{j}\right) & \approx \frac{C\left(x\_{j}\right)}{C\left(t\_{j}\right)} \sum\_{k=1}^{M} P\left(t\_{k} \mid w\_{j-1}\right) P\left(t\_{j} \mid t\_{k}\right) \\\\ &=\frac{1}{C\left(t\_{j}\right)} \sum\_{k=1}^{M}\left[\frac{C\left(w\_{j-1} t\_{k}\right)}{C\left(w\_{j-1}\right)} \times \frac{C\left(t\_{k} t\_{j}\right)}{C\left(t\_{k}\right)}\right] \end{aligned}$
