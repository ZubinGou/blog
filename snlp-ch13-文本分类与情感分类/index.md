# 《统计自然语言处理》第13章 - 文本分类与情感分类


# ch13 文本分类与情感分类

## 13.1 文本分类概述
- [Sebastiani, 2002]数学模型描述文本分类：
	- 获得函数（分类器）：$\Phi: {D} \times {C} \rightarrow\{ {T}, \quad {F}\}$
	- 文档：$D=\{d_1, d_2, ...,d\_{|D|}\}$
	- 类别：${C}=\left\{ {c}\_{1}, {c}\_{2}, \ldots, {c}\_{|C|}\right\}$
- 关键问题
	- 文本表示
	- 分类器设计
- 文本分类系统
	![4cd185d07b05e0d8520cfee2419a9b68.png](/_resources/9582ff377cf44784b6ea0bd41ae78f8d.png)
	![57a31df7f3712c77169af2b6c03a9711.png](/_resources/34779827acf5402ba07a979a2aee2e33.png)
	1. 文本预处理：分词，取出停用词，过滤低频词，编码归一化等

	2. 文本向量化：如使用向量空间模型VSM或者概率统计模型对文本进行表示，使计算机能够理解计算，用的方法基于集合论模型、基于代数轮模型、基于频率统计模型等
	3. 文本特征提取和选择：特征提取对应着特征项的选择和特征权重的计算。是文本分类的核心内容，常用的特征提取方法：
		1)用映射或者变换的方法对原始特征降维（word2vec）；

		2)从原始的特征中挑选出一些最具代表性的特征；

		3)根据专家的知识挑选出最具影响力的特征；

		4)基于数学的方法选取出最具分类信息的特征。

	4. 分类器选择：回归模型，二元独立概率模型，语言模型建模IR模型

- 文本分类系统分类
	- 基于知识工程（knowledge learning，KE）
		- 专家人工规则
	- 基于机器学习（machine learning，ML）
		- 用训练样本进行特征选择、分类器参数训练
		- 根据选择的特征对分类输入样本进行形式化
		- 输入到分类器进行类别判定

## 13.2 文本表示
- 向量空间模型（vector space modle，VSM）
	- 文档（document）：通常是文章中具有一定规模的片段，如句子、句群、段落、段落组直至整篇文章。
	- 项／特征项（term/feature term）：特征项是VSM中最小的不可分的语言单元，可以是字、词、词组或短语等。一个文档的内容被看成是它含有的特征项所组成的集合，表示为：Document＝D（t1，t2，…，tn），其中tk是特征项，1≤k≤n。
	- 项的权重（term weight）：对文档n个特征项依据一定原则赋予权重$w_k$，D＝D（t1，w1;t2，w2;…;tn，wn），简记为D＝D（w1，w2，…，wn）
	- VSM定义：给定一个文档D（t1，w1;t2，w2;…;tn，wn），D符合以下两条约定：
		1. 各个特征项tk（1≤k≤n）互异（即没有重复）；
		2. 各个特征项tk无先后顺序关系（即不考虑文档的内部结构）
	- 特征项$t_k$看作n维坐标系，权重$w_k$作为坐标值，文本表示维n维向量
		![d038a8bfa260c4b1cd7da04def10bc06.png](/_resources/82468445660f4c8d8cf5603fae23deec.png)
- 向量的相似性度量（similarity）：任意两个文档D1和D2之间的相似系数Sim（D1，D2）指两个文档内容的相关程度（degree of relevance）
	- 向量内积：$\operatorname{Sim}\left(D\_{1}, D\_{2}\right)=\sum\_{k=1}^{n} w\_{1 k} \times w\_{2 k}$
	- 考虑归一化，向量余弦：$\operatorname{Sim}\left(D\_{1}, D\_{2}\right)=\cos \theta=\frac{\sum\_{k=1}^{n} w\_{1 k} \times w\_{2 k}}{\sum\_{k=1}^{n} w\_{1 k}^{2} \sum\_{k=1}^{n} w\_{2 k}^{2}}$
- 除了VSM以外表示方法：
	- 词组表示法：
		- 提高不显著
		- 提高了特征向量语义含量，但降低了特征向量统计质量，使特征向量更加稀疏
	- 概念表示法
		- 用概念（concept）作为特征向量的特征表示
		- 用概念代替单个词可以在一定程度上解决自然语言的歧义性和多样性给特征向量带来的噪声问题，有利于提高文本分类的效果


## 13.3 文本特征选择
- 文本特征可以是：字、词、短语、概念等等
- 常用方法：
	- 文档频率（document frequency, DF）特征提取法
	- 信息增益（information gain, IG）法
	- χ2统计量（CHI）法
	- 互信息（mutual information, MI）方法

### 13.3.1 文档频率DF
- 文档频率（DF）= 包含某特征项的文档数量 / 总文档数量
- 舍弃DF过小（没有代表性）、过大（没有区分度）的特征
- 优点：降低向量计算复杂度，可能提高分类准确率，因为去掉了一部分噪声特征，简单易行
- 缺陷：理论根据不足。根据信息论，某些低频率特征往往包含较多信息

### 13.3.2 信息增益（IG）法
- 信息增益法：根据某特征项$t_i$使得期望信息或者信息熵的有效减少量（信息增益）来判断其重要程度以取舍
- 信息增益 = 不考虑任何特征时文档的熵 - 考虑该特征后文档的熵
$\begin{aligned} \operatorname{Gain}\left(t\_{i}\right)=& \text { Entropy }(S)-\text { Expected Entropy }\left(S\_{t\_{i}}\right) \\\\=&\left\{-\sum\_{j=1}^{M} P\left(C\_{j}\right) \times \log P\left(C\_{j}\right)\right\}-\left\{P\left(t\_{i}\right) \times\left[-\sum\_{j=1}^{M} P\left(C\_{j} \mid t\_{i}\right) \times \log P\left(C\_{j} \mid t\_{i}\right)\right]\right.\\ &\left.+P\left(\bar{t}\_{i}\right) \times\left[-\sum\_{i=1}^{M} P\left(C\_{j} \mid \bar{t}\_{i}\right) \times \log P\left(C\_{j} \mid \bar{t}\_{i}\right)\right]\right\} \end{aligned}$
- 信息增益法是理论上最好的特征选取方法，但实际上许多高信息增益的特征出现频率较低，选取特征数目少时往往存在数据稀疏问题，分类效果差


### 13.3.3 $\chi^2$统计量/开方检验
- $\chi^2$统计量（CHI）衡量的是特征项ti和类别Cj之间的相关联程度，并假设ti和Cj之间符合具有一阶自由度的$\chi^2$分布

- ![9f02bc61446d7b37570ceca72d35a6e2.png](/_resources/5c695beda9de4e259c1f532b0abef568.png)
$\chi^{2}\left(t\_{i}, C\_{j}\right)=\frac{N \times(A \times D-C \times B)^{2}}{(A+C) \times(B+D) \times(A+B) \times(C+D)}$
- 两种实现方法
	1. 最大值法：分别计算$t_i$对于每个类别的CHI值，然后在整个训练语料上：
	$\chi\_{\mathrm{MAX}}^{2}\left(t\_{i}\right)=\max \_{j=1}^{M} x\left\{\chi^{2}\left(t\_{i}, C\_{j}\right)\right\}$
	2. 平均值法：计算各特征对于各类别的平均值
$\chi\_{\mathrm{AVG}}^{2}\left(t\_{i}\right)=\sum\_{j=1}^{M} P\left(C\_{j}\right) \chi^{2}\left(t\_{i}, C\_{j}\right)$
	- 保留统计量高于给定阈值的特征
- 开方检验的缺点：忽略了词频，夸大了低频词的作用（低频词缺陷）。

### 13.3.4 互信息（MI）法
- 基本思想：互信息越大，特征ti和类别Cj共现的程度越大。
	$\begin{aligned} I\left(t\_{i}, C\_{j}\right) &=\log \frac{P\left(t\_{i}, C\_{j}\right)}{P\left(t\_{i}\right) P\left(C\_{j}\right)} \\\\ &=\log \frac{P\left(t\_{i} \mid C\_{j}\right)}{P\left(t\_{i}\right)} \\\\ & \approx \log \frac{A \times N}{(A+C) \times(A+B)} \end{aligned}$
- 若特征ti和类别Cj无关，则P（ti，Cj）＝P（ti）×P（Cj），那么，
I（ti，Cj）＝0
- 两种处理方法
	1. 最大值法：$I\_{\mathrm{MAX}}\left(t\_{i}\right)=\max \_{j=1}^{M} \mathrm{x}\left[P\left(C\_{j}\right) \times I\left(t\_{i}, C\_{j}\right)\right]$
	2. 平均值法：$I\_{\mathrm{AVG}}\left(t\_{i}\right)=\sum\_{j=1}^{M} P\left(C\_{j}\right) I\left(t\_{i}, C\_{j}\right)$

### 其他方法
- DTP（distance to transition point）方法［Moyotl-Hernández and Jiménez-Salazar, 2005］
- 期望交叉熵法
- 文本证据权法
- 优势率方法［Mademnic and Grobelnik, 1999］
- “类别区分词”的特征提取方法［周茜等，2004］
- 基于粗糙集（rough set）的特征提取方法 TFACQ［Hu et al., 2003］
- 强类信息词（strong information class word, SCIW）方法［Li and Zong, 2005a］


## 13.4 特征权重计算方法
![ca339b10ccf104b3cdd78aad29be5aac.png](/_resources/c955db382e48427fa740024766ae6dec.png)
![9619e4941924763c9588f4f0c1765527.png](/_resources/925dc149b0c5448fbca7bfb1e80e6e27.png)


## 13.5 分类器设计
- 常用分类算法
	- 朴素的贝叶斯分类法（naΪve Bayesian classifier）
	- 基于支持向量机（support vector machines,SVM）的分类器
	- k-最近邻法（k-nearest neighbor, kNN）
	- 神经网络法（neural network, NNet）
	- 决策树（decision tree）分类法
	- 模糊分类法（fuzzy classifier）
	- Rocchio分类方法
	- Boosting算法

### 13.5.1 朴素贝叶斯分类器
朴素贝叶斯( naive Bayes)是一种最简单常用的概率生成式模型（Generative Model），生成式模型是指有多少类，我们就学习多少个模型，分别计算新测试样本 $x$跟三个类别的联合概率$P(x, y)$，再根据贝叶斯公式计算选取使得$P(y \mid x)$最大的作为分类。而判别式模型（Discrimitive Model）训练数据得到分类函数和分界面（如SVM），不能反应训练数据本身的特性。

基本思想：利用特征项和类别的联合概率来估计给定文档的类别概率。假设文本是基于词的一元模型。

假设现有的类别$C=(c_1,c_2...c_m)$，则文档最可能属于$\hat{c}=\underset{c \in C}{\operatorname{argmax}} P(c \mid d)$类，使用贝叶斯公式转换为如下形式：
$$
\hat{c}=\underset{c \in C}{\operatorname{argmax}} P(c \mid d)=\underset{c \in C}{\operatorname{argmax}} \frac{P(d \mid c) P(c)}{P(d)}
$$
分母相同可以忽略，得到：
$$
\hat{c}=\underset{c \in C}{\operatorname{argmax}} P(c \mid d)=\underset{c \in C}{\operatorname{argmax}} P(d \mid c) P(c)
$$
这个公式由两部分组成，前面那部分$P(d|c)$ 称为似然函数，后面那部分$P(c)$ 称为先验概率。使用词袋模型来表示文档$d$，文档$d$的每个特征表示为：$d={f_1,f_2,f_3……f_n}$，那么这里的特征$f_i$ 其实就是单词$w_i$ 出现的频率（次数），公式转化为：
$$
\hat{c}=\underset{c \in C}{\operatorname{argmax}} \overbrace{P\left(f\_{1}, f\_{2}, \ldots, f\_{n} \mid c\right)}^{\text {likelihood }} \overbrace{P(c)}^{\text {prior }}
$$
朴素贝叶斯的“朴素”表现在假设各个特征之间相互独立（条件独立性假设），则$P\left(f\_{1}, f\_{2} \ldots \ldots\_{n} \mid c\right)=P\left(f\_{1} \mid c\right){\times} P\left(f\_{2} \mid c\right){\times} \ldots \ldots{\times} P\left(f\_{n} \mid c\right)$，故而公式变为
$$
c\_{N B}=\underset{c \in C}{\operatorname{argmax}} P(c) \prod\_{f \in F} P(f \mid c)
$$
因为每个概率的值很小，多个相乘则可能出现下溢（underflower）， 引入对数函数$log$，在$log\ space$中进行计算：
$$
c\_{N B}=\underset{c \in C}{\operatorname{argmax}} \log P(c)+\sum\_{i \in \text {positions}} \log P\left(w\_{i} \mid c\right)
$$

1. 文档采用DF向量表示法：
	$P\left(\right.$ Doc $\left.\mid C\_{i}\right)=\prod\_{t\_{j} \in V} P\left(\operatorname{Doc}\left(t\_{j}\right) \mid C\_{i}\right)$
$P($ Doc $)=\sum\_{i}\left[P\left(C\_{i}\right) \prod\_{t\_{i} \in V} P\left(\operatorname{Doc}\left(t\_{i}\right) \mid C\_{i}\right)\right]$
$P\left(C\_{i} \mid\right.$ Doc $)=\frac{P\left(C\_{i}\right) \prod\_{t\_{j} \in V} P\left(\operatorname{Doc}\left(t\_{j}\right) \mid C\_{i}\right)}{\sum\_{i}\left[P\left(C\_{i}\right) \prod\_{t\_{j} \in V} P\left(\operatorname{Doc}\left(t\_{j}\right) \mid C\_{i}\right)\right]}$
	- 拉普拉斯估计：$P\left(\operatorname{Doc}\left(t\_{j}\right) \mid C\_{i}\right)=\frac{1+N\left(\operatorname{Doc}\left(t\_{j}\right) \mid C\_{i}\right)}{2+\left|D\_{c\_{i}}\right|}$
	- 分子加1和分母加2背后的基本原理是这样的：在执行实际的试验之前，我们假设已经有两次试验，一次成功和一次失败
2. 文档采用TF向量表示法：
	$P\left(C\_{i} \mid\right.$ Doc $)=\frac{P\left(C\_{i}\right) \prod\_{t\_{i} \in V} P\left(t\_{j} \mid C\_{i}\right)^{\mathrm{TF}\left(t\_{i}, \text { Doc }\right)}}{\sum\_{j}\left[P\left(C\_{j}\right) \prod\_{t\_{i} \in V} P\left(t\_{i} \mid C\_{j}\right)^{\mathrm{TF}\left(t\_{i}, \mathrm{D}\_{0}\right)}\right]}$
	- 拉普拉斯估计：$P\left(t\_{i} \mid C\_{i}\right)=\frac{1+\operatorname{TF}\left(t\_{i}, C\_{i}\right)}{|V|+\sum\_{j} \operatorname{TF}\left(t\_{j}, C\_{i}\right)}$
	- 加一平滑，对每个类别下所有划分的计数加1


### 13.5.2 SVM分类器
- 对于多类模式识别问题通常需要建立多个两类分类器

### 13.5.3 k-最邻近法（kNN）
- 在训练集中找邻近的k个文档，对其中每类的每个文档进行权重（余弦相似度）求和，作为该类和测试文档的相似度，决策规则：
	$y\left(x, C\_{j}\right)=\sum\_{d\_{i} \in k \mathrm{NN}} \operatorname{sim}\left(x, d\_{i}\right) y\left(d\_{i}, C\_{j}\right)-b\_{j}$
	- $y(d_i，C_j)$为1表示di属于分类Cj，0表不属于。
	- $b_j$为二元决策的阈值

### 13.5.4 神经网络（NNet）分类器
- 输入单词或者更复杂特征向量，机器学习输入到分类的非线性映射

### 13.5.5 线性最小平方拟合法（linear least-squares fit, LLSF）
- 从训练集和分类文档中学习得到多元回归模型（multivariate regression model）
- $\boldsymbol{F}\_{\mathrm{LS}}=\arg \min \_{F}\|\boldsymbol{F} \times \boldsymbol{A}-\boldsymbol{B}\|^{2}$
- 矩阵A和矩阵B描述的是训练数据（对应栏分别是输入和输出向量）；FLS为结果矩阵，定义了从任意文档到加权分类向量的映射。对这些分类的权重映射值排序，同时结合阈值算法，就可以来判别输入文档所属的类别。阈值是从训练中学习获取的

### 13.5.6 决策树分类器
- 树的根结点是整个数据集合空间，每个分结点是对一个单一变量的测试，该测试将数据集合空间分割成两个或更多个类别，即决策树可以是二叉树也可以是多叉树。每个叶结点是属于单一类别的记录。
- 训练集生成决策树，测试集修剪决策树
- 一般可通过递归分割的过程构建决策树，其生成过程通常是自上而下的，目的为最佳分割
- 从根结点到叶结点都有一条路径，这条路径就是一条决策“规则”
- 信息增益是决策树训练中常用的衡量给定属性区分训练样本能力的定量标准

### 13.5.7 模糊分类器
- 任何一个文本或文本类都可以通过其特征关键词描述，因此，可以用一个定义在特征关键词类上的模糊集来描述它们。
- 判定分类文本T所属的类别可以通过计算文本T的模糊集FT分别与其他每个文本类的模糊集Fk的关联度SR实现，两个类的关联度越大说明这两个类越贴近

### 13.5.8 Rocchio分类器
- Rocchio分类器是情报检索领域经典的算法
- 基本思想：
	1. 为每个训练文本C建立特征向量
	2. 用训练文本特征向量为每类建立原始向量（类向量）
	3. 对待分类文本，距离最近的类就是所属类别
- 距离：向量点积、余弦相似度等
- 如果C类文本的原型向量为w1，已知一组训练文本，可以预测w1改进的第j个元素值为
	$w\_{1 j}^{\prime}=\alpha w\_{1 j}+\beta \frac{\sum\_{i \in C} x\_{i j}}{n\_{C}}-\gamma \frac{\sum\_{i \in C} x\_{i j}}{n-n\_{C}}$
	- nC是训练样本中正例个数，即属于类别C的文本数；xij是第i个文本特征向量的第j个元素值；α、β、γ为控制参数。α控制了上一次计算所得的w对本次计算所产生的影响，β和γ分别控制正例训练集和反例训练集对结果的影响。

### 13.5.9 基于投票的分类方法
- 多分类器组合
- 核心思想：k个专家判断的有效组合应该优于某个专家个人的判断结果
- 投票算法：
	1. Bagging算法（民主）
		- 票数最多的作为最终类别
	2. Boosting算法（精英）
		- Boosting推进：每次将分类错误的样本加入下一个弱分类器的训练
		- Adaboosting自适应推进：提高错误点的权值，加权投票（精度高的弱分类器权重大）
- Boosting
	- 1984年Valiant提出的”可能近似正确”-PAC(Probably Approximately Correct)学习模型
	- 强与弱
		- 强学习：学习效果好
		- 弱学习：仅比随机好
	- Boost（Schapire 1990）：任意 弱学习算法 -> （任意正确率）强学习算法，加强过程多项式复杂度
	- Adaboost（Freund and Schapire）：
		- 不需要提前知道弱学习算法先验知识

## 13.6 文本分类性能测评
![93d2643f71b5b3245deb26921e5fe355.png](/_resources/cb2c4045c6b94c84b38f4a6f01af5450.png)
- 正确率（Precision）：$P=\frac{T P}{T P+F P}$
- 召回率（Recall）：$R=\frac{T P}{T P+F N}$
- $F\_{\beta}$值（P与R加权调和平均）：$F\_{\beta}=\frac{\beta^{2}+1}{\frac{\beta^{2}}{r}+\frac{1}{p}}=\frac{\left(\beta^{2}+1\right) \times p \times r}{\beta^{2} \times p+r}$
- $F_1$值（P与R调和平均值）：$F\_{1}=\frac{1}{\frac{1}{2} \frac{1}{P}+\frac{1}{2} \frac{1}{R}}=\frac{2 P R}{P+R}$
- 宏平均（Macro-averaging）：先对每一个类统计指标值，然后在对所有类求算术平均值。
- 微平均（Micro-averaging）：对数据集中的每一个实例不分类别进行统计建立全局混淆矩阵，然后计算相应指标。
	- 微平均更多地受分类器对一些常见类（这些类的语料通常比较多）分类效果的影响，而宏平均则可以更多地反映对一些特殊类的分类效果。在对多种算法进行对比时，通常采用微平均算法。
- 平衡点（break-even point）评测法［Aas and Eikvil, 1999］：通过调整分类器的阈值，调整正确率和召回率的值，使其达到一个平衡点的评测方法
- 11点平均正确率方法［Taghva et al., 2004］：为了更加全面地评价一个分类器在不同召回率情况下的分类效果，调整阈值使得分类器的召回率分别为：0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1，然后计算出对应的11个正确率，取其平均值



## 13.7 情感分类
- 情感分析（sentiment analysis）：借助计算机帮助用户快速获取、整理和分析相关评价信息，对带有情感色彩的主观性文本进行分析、处理、归纳和推理［Pang and Lee, 2008］。情感分析包含较多的任务，如情感分类（sentiment classification）、观点抽取（opinion extraction）、观点问答和观点摘要等。
- 情感分类是指根据文本所表达的含义和情感信息将文本划分成褒扬的或贬义的两种或几种类型，是对文本作者倾向性和观点、态度的划分，因此有时也称倾向性分析（opinion analysis）
- 情感分类的特殊性：情感的隐蔽性、多义性和极性不明显性
1. 按机器学习方法分类
	- 有监督学习方法
	- 半监督学习方法
	- 无监督学习方法
2. 按照研究问题分类
	- 领域相关性研究
		- 领域适应性（domain adaptation）研究
	- 数据不平衡问题研究
