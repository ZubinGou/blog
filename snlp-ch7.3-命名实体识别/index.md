# 《统计自然语言处理》第7.3章 - 命名实体识别


## 7.3 命名实体识别
### 7.3.1 方法概述
- 实体概念在文本中的引用（entity mention，指称项）三种形式
	1. 命名性指称
	2. 名词性指称
	3. 代词性指称
	> “［［中国］乒乓球男队主教练］［刘国梁］出席了会议，［他］指出了当前经济工作的重点。”
	> - 实体概念“刘国梁”的指称项有三个
	> - “中国乒乓球男队主教练”是名词性指称
	> - “刘国梁”是命名性指称
	> - “他”是代词性指称
- 任务发展：
	- 在MUC-6组织NERC任务之前，主要关注的是人名、地名和组织机构名这三类专有名词的识别。
	- 自MUC-6起，地名被进一步细化为城市、州和国家。后来也有人将人名进一步细分为政治家、艺人等小类
	- 在CoNLL组织的评测任务中扩大了专有名词的范围，包含了产品名的识别
	- 在其他一些研究工作中也曾涉及电影名、书名、项目名、研究领域名称、电子邮件地址和电话号码等。尤其值得关注的是，很多学者对生物信息学领域的专用名词（如蛋白质、DNA、RNA等）及其关系识别做了大量研究工作。
- 本节主要关注人名、地名和组织机构名这三类专有名词的识别方法。
- 方法发展：
	- 早期：规则
	- 20世纪90年代后期以来：统计机器学习，主要四类方法：
		![d19ef859b61d2198f2bc25c85d54d66a.png](/_resources/043013c087f742b7a496a247c2b12705.png)


### 7.3.2 基于CRF的命名实体识别
- 可以说是命名实体识别最成功的方法
- 原理：
	- 与基于字的汉语分词方法一样，将命名实体识别过程看作序列标注问题
- 基本思路：
	1. 分词
	2. 人名、简单地名、简单组织机构名识别
	3. 复合地名、复合组织机构名识别
- 常用标注语料库：北京大学计算语言学研究所标注的现代汉语多级加工语料库
- 训练：
	1. 将分词语料的标记符号转化成用于命名实体序列标注的标记
	![9861d6d13f2678d4a14cd4e1cde7f746.png](/_resources/d99a6acaada145c18955f31d776832df.png)
	2. 确定特征模板：
		- 观察窗口：以当前位置的前后n（一般取2~3）个位置范围内的字串及其标记作为观察窗口
		- 由于不同的命名实体一般出现在不同的上下文语境中，因此，对于不同的命名实体识别一般采用不同的特征模板
		![9a40e67ff3789363f9810f00e8be06d7.png](/_resources/92bf254ebb344521b9a5c2a9e1ad258f.png)
	3. 训练CRF模型参数$\lambda$


### 7.3.3 基于多特征的命名实体识别
- 命名实体识别：各种方法都是充分发现和利用实体上下文特征、实体内部特征，特征颗粒度有大（词性和角色级特征）有小（词形特征）	
- [吴友政，2006]基于多特征融合的汉语命名实体识别方法
	- 在分词和词性标注的基础上进一步进行命名实体识别
	- 4个子模型
		- 词形上下文模型：估计在给定词形上下文语境中产生实体的概率
		- 词性上下文模型：估计在给定词性上下文语境中产生实体的概率
		- 词形实体模型：估计在给定实体类型的情况下词形串作为实体的概率
		- 词性实体模型：估计在给定实体类型的情况下词性串作为实体的概率

#### （1）模型描述
- 词形：
	1. 字典中任何一个字或词单独构成一类
	2. 人名（Per）、人名简称（Aper）、地名（Loc）、地名简称（Aloc）、机构名（Org）、时间词（Tim）和数量词（Num）各定义为一类
	- 词形语言模型定义了$|V|+7$个词形，$|V|$表示词典规模
	- 词形序列WC：词性构成的序列
- 词性：
	1. 北大计算语言学研究所开发的汉语文本词性标注标记集
	2. 人名简称词性、地名简称词性
	- 共47个词性标记
	- 词性序列TC
- 命名实体识别
	- 输入：带有词性标注的词序列
		$\mathrm{WT}=w\_{1} / t\_{1} \quad w\_{2} / t\_{2} \quad \cdots \quad w\_{i} / t\_{i} \quad \cdots \quad w\_{n} / t\_{n}$
	- 在分词和标注的基础上：对部分词语拆分、组合（确定实体边界）、和重新分类（确定实体类别）
	- 输出：最优“词形/词性”序列$WC^\*/TC^\*$
		$W C^{\*} / \mathrm{TC}^{\*}=\mathrm{wc}\_{1} / \mathrm{tc}\_{1} \quad \mathrm{wc}\_{2} / \mathrm{tc}\_{2} \quad \cdots \quad \mathrm{wc}\_{i} / \mathrm{tc}\_{i} \quad \cdots \quad \mathrm{wc}\_{m} / \mathrm{tc}\_{m}$
	- 算法：
		1. 词形特征模型
			- 根据词性序列W产生候选命名实体，用Viterbi确定最优词形序列$WC^*$
		2. 词性特征模型
			- 根据词性序列T产生候选命名实体，用Viterbi确定最优词性序列$TC^*$
		3. 混合模型/多特征识别算法
			- 词形和词性混合模型是根据词形序列W和词性序列T产生候选命名实体，一体化确定最优序列WC*/TC*，即本节将要介绍的基于多特征的识别算法
- 多特征识别算法
	- 输入：
		- 词序列：$W=w\_{1} \quad w\_{2} \quad \cdots \quad w\_{i} \quad \cdots \quad w$
		- 词性序列：$t\_{1} \quad t\_{2} \quad \cdots \quad \cdots \quad t\_{i} \quad \cdots \quad t\_{n}$
	- 词形特征模型：$\mathrm{WC}^{\*}=\underset{\mathrm{WC}}{\operatorname{argmax}} P(\mathrm{WC}) \times P(W \mid \mathrm{WC})$
	- 词性特征模型：$\mathrm{T} \mathrm{C}^{\*}=\underset{\mathrm{TC}}{\operatorname{argmax}} P(\mathrm{TC}) \times P(T \mid \mathrm{TC})$
	- 混合：$\begin{aligned} &\left(\mathrm{WC}^{\*}, \mathrm{TC}^{\*}\right) \\\\=&\left.\operatorname{argmax}\_{(\mathrm{WC}, \mathrm{TO}}\right) P(\mathrm{WC}, \mathrm{TC} \mid W, T) \\\\=& \operatorname{argmax}\_{(\mathrm{WC}, \mathrm{TC})} P(\mathrm{WC}, \mathrm{TC}, W, T) / P(W, T) \\\\ \approx & \operatorname{argmax}\_{(\mathrm{WC}, \mathrm{TO}} P(\mathrm{WC}, W) \times[P(\mathrm{TC}, T)]^{\beta} \\\\ \approx & \operatorname{argmax}\_{(\mathrm{WC}, \mathrm{TO}} P(\mathrm{WC}) \times P(W \mid \mathrm{WC}) \times[P(\mathrm{TC}) \times P(T \mid \mathrm{TC})]^{-3} \end{aligned}$
		- β是平衡因子，平衡词形特征和词性特征的权重
		- 词形上下文模型P（WC）
		- 词性上下文模型P（TC）
		- 实体词形模型P（W|WC）
		- 实体词性模型P（T|TC）

#### （2）词形和词性上下文模型
- 三元语法模型近似：
	- $P(\mathrm{WC}) \approx \prod\_{i=1}^{m} P\left(\mathrm{wc}\_{i} \mid \mathrm{wc}\_{i-2} \mathrm{wc}\_{i-1}\right)$
	- $P(\mathrm{TC}) \approx \prod\_{i=1}^{m} P\left(\mathrm{tc}\_{i} \mid \mathrm{tc}\_{i-2} \mathrm{tc}\_{i-1}\right)$


#### （3）实体模型
- 考虑到每一类命名实体都具有不同的内部特征，因此，不能用一个统一的模型刻画人名、地名和机构名等实体模型。例如，人名识别可采用基于字的三元模型，地名和机构名识别可能更适合于采用基于词的三元模型等。
- 为提高外国人名识别性能，划分为日本人名、欧美人名、俄罗斯人名
	![7659398db3ee5190f8775b9c097e8657.png](/_resources/08f13118f9484f60a375ea2a7ea5f68e.png)
- 实体模型：
	- 人名实体模型
	- 地名和机构名实体模型
	- 单字地名实体模型
	- 简称机构名实体模型

#### （4）专家知识
- 在基于统计模型的命名实体识别中，最大的问题是数据稀疏严重，搜索空间太大，从而影响系统的性能和效率。引入**专家系统**知识来限制候选实体产生：
	1. 人名识别的专家知识
	2. 地名识别的专家知识
	3. 机构名识别的专家知识

#### （5）模型训练
- 4个参数
	- 词性上下文模型P（TC）和词形上下文模型P（WC）从《人民日报》标注语料中学习
	- 中国人名、外国人名、地名、机构名的实体词性和词形模型从实体列表语料中训练
- 数据稀疏问题严重：Back-off数据平滑，引入逃逸概率计算权值
	$$\begin{aligned} & \hat{P}\left(w\_{n} \mid w\_{1} \cdots w\_{n-1}\right) \\\\=& \lambda\_{N} P\left(w\_{n} \mid w\_{1} \cdots w\_{n-1}\right)+\lambda\_{N-1} P\left(w\_{n} \mid w\_{2} \cdots w\_{n-1}\right) \\\\ &+\cdots+\lambda \cdot P\left(w\_{n}\right)+\lambda\_{0} p\_{0} \end{aligned}$$
	- 其中$\lambda\_{i}=\left(1-e\_{i}\right) \sum\_{k=i+1}^{n} e\_{k}, 0<i<n, \lambda\_{n}=1-e\_{n}$

#### （6）测试结果
![c6db0176634e7dff6976a9bb0d5441f6.png](/_resources/18407802a89b459b9ede7755222ed125.png)
