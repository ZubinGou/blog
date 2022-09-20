# 《统计自然语言处理》第6章 - 概率图模型


## 6.1 概述
- 概率图模型（probabilistic graphical models）：在概率模型的基础上，使用基于图的方法表示概率分布/概率密度/密度函数，是一种通用化的不确定性知识表示和处理方法。
	- 结点：变量
	- 边：变量概率关系
- 边是否有向：有向概率图模型、无向概率图模型
![116a254a554df802f35b1702b972029a.png](/blog/_resources/13327b7bba164e918909ac929cb4ca87.png)
- 应用：
	- DBN：动态系统的推断和预测
		- HMM：语音识别、汉语自动分词、词性标注、统计机器翻译
		- Kalman filter：信号处理
	- Markov networks / Markov random field(MRF)
		- CRF：序列标注、特征选择、机器翻译
		- Boltzman machine：依存句法分析、语义角色标注
- 概率图模型演变
![c4a62b2ab674e36b6d56725b4118aa9b.png](/blog/_resources/4208836588794580a6fa285504afab7c.png)
- 生成式/产生式模型 vs. 区分式/判别式模型
	- 本质区别：观测序列x与状态序列y的决定关系
- 生成式模型
	- 假定y决定x
	- 对联合分布$p(x, y)$建模，估计生成概率最大的生成序列来获取y
	- 特征：一般有严格独立性假设，特征事先给定
	- 优点：灵活、变量关系清楚、模型可以增量学习获得、可用于数据不完整情况
	- 典型：n-gram、HMM、Naive Bayes、概率上下文无关文法
- 判别式模型
	- 假定x决定y
	- 对后验概率$p(y|x)$建模，从x提取特征，学习参数，使条件概率符合一定形式的最优
	- 特征：任意给定，一般通过函数表示
	- 优点：处理多类或一类与其他类差异比较灵活简单
	- 弱点：模型描述能力有限、变量关系不清、一般为有监督，不能扩展为无监督
	- 典型：最大熵模型、条件随机场、SVM、最大熵马尔科夫模型（maximum-entropy Markov model, MEMM）、感知机（perceptron）


## 6.2 贝叶斯网络
- 贝叶斯网络又称信度网络/信念网络（belief networks）
- 理论基础：贝叶斯公式
- 形式：DAG
	- 结点：随机变量（可观测量、隐含变量、未知参量或假设等）
	- 有向边：条件依存关系
	- ![af63428950c8168ebb864431474ceb43.png](/blog/_resources/f7a3265f853c4838aac3d8c0a10ef295.png)
		- 如图，联合概率函数为：$P(H, S, N)=P(H \mid S, N) \times P(S \mid N) \times P(N)$
- 构造贝叶斯网络
	1. 表示：在某一随机变量的集合$x＝{X_1，L，X_n}$上给出其联合概率分布P。
	2. 推断：回答关于变量的询问，如当观察到某些变量（证据变量）时，推断另一些变量子集的变化。
		- 常用精确推理方法：
			- 变量消除法（variable elimination）：基本任务是计算条件概率$p(X_Q|X_E＝x)$，其中，$X_Q$是询问变量的集合，$X_E$为已知证据的变量集合。其基本思想是通过分步计算不同变量的边缘分布按顺序逐个消除未观察到的非询问变量
			- 团树（clique tree）：使用更全局化的数据结构调度各种操作，以获得更加有益的计算代价
		- 常用近似推理算法：
			- 重要性抽样法（importance sampling）
			- 随机马尔科夫链蒙特卡洛（Markov chain Monte Carlo, MCMC）模拟法
			- 循环信念传播法（loopy belief propagation）
			- 泛化信念传播法（generalized belief propagation）
	3. 学习：参数学习和结构学习
		- 参数学习：决定变量之间相互关联的量化关系，即依存强度估计
			- 即对每个结点X，计算给定父结点条件下X的概率，概率分布可以是任意形式，通常为离散分布或高斯分布
			- 常用参数学习方法：
				- 最大似然估计
				- 最大后验概率法
				- 期望最大化方法（EM）
				- 贝叶斯估计方法（贝叶斯图中常用）
		- 结构学习：寻找变量之间的图关系
			- 很简单情况：专家构造。多数使用系统中人工构造贝叶斯网络几乎不可能。
			- 自动学习贝叶斯网络的图结构
- 贝叶斯网络是一种不定性因果关联模型，能够在已知有限的、不完整、不确定信息的条件下进行学习和推理
	- 应用：广泛应用于故障诊断和维修决策等领域；汉语自动分词和词义消歧	


## 6.3 马尔科夫模型
- 随机过程又叫随机函数，是随时间而随机变化的过程。
- 离散的一阶马尔可夫链（Markov chain）
	- $P(q\_{t}=s\_{i} \mid q\_{t-1}=s\_{j}, q\_{t-1}=s\_{k}, \cdots)=P(q\_{t}=s\_{j} \mid q\_{t-1}=s\_{i})$
- 马尔科夫模型/可视马尔科夫模型（visible Markov model，MM/VMM）
	- 只考虑上式独立于时间t的随机过程：$P(q\_{t}=s\_{j} \mid q\_{t-1}=s\_{i})=a\_{i j}, \quad 1 \leqslant i, j \leqslant N$
	- 满足：$a\_{i j} \geqslant 0$，$\sum\_{i=1}^{N} a\_{i j}=1$
- 有$N$个状态的一阶马尔可夫过程有$N^2$次状态转移，可表示成状态转移矩阵
	- eg. 一段文字中s1：名词，s2：动词，s3：形容词，转移矩阵：
		- ![f49755c3d01a2b270bef8fdfcd3b6c32.png](/blog/_resources/e21f56eec04a4ed3a4864ae3859908c8.png)
		- 假设名词开头，则O=“名动形名”概率为：
		$\begin{aligned} P(O \mid M) &=P(s\_{1}, s\_{2}, s\_{3}, s\_{1} \mid M) \\\\ &=P(s\_{1}) \cdot P(s\_{2} \mid s\_{1}) \cdot P(s\_{3} \mid s\_{2}) \cdot P(s\_{1} \mid s\_{3}) \\\\ &=1 \times a\_{12} \times a\_{23} \times a\_{31} \\\\ &=0.5 \times 0.2 \times 0.4 \\\\ &=0.04 \end{aligned}$
- 马尔科夫模型可视为随机的非确定有限状态机
  	- ![063763cfce271c3250b140c5dad97d8a.png](/blog/_resources/5022f43cb17a495eaf00dbc9b2696c17.png)
	- 序列概率为转移弧概率乘积：

	![0226f9548c08a087a5d7d3fade52d3d3.png](/blog/_resources/6bff78fa7fe240b6a97a14bdcd260b2a.png)
- n-gram与马尔科夫模型
	- 2-gram就是一阶马尔科夫模型
	- 对于$n\ge 3$的n-gram确定数量的历史，可以通过将状态空间描述成多重前面状态的交叉乘积的方式，转化为马尔科夫模型，可以称之为m阶马尔科夫模型，m为历史数。
	- n元语法模型就是n-1阶马尔可夫模型


## 6.4 隐马尔科夫模型
- VMM每个状态代表可观察的事件，限制了模型适应性，提出HMM
- HMM：观察到的事件是隐蔽的状态转换过程的随机函数，模型为双重随机过程
	- ![90edc401c7ca942ce9ec908aafa77c4c.png](/blog/_resources/9b6c48c230ae420c8c1777ddb67d0601.png)
	- 类比：口袋取球，室外人只看到球
- HMM记为五元祖$\mu=(\mathrm{S}, \mathrm{K}, \mathrm{A}, \mathrm{B}, \pi)$
	- S：状态结合
	- K：输出符号集合
	- A：状态转移概率
	- B：符号发射概率
	- $\pi$：初始状态概率分布
- 基本问题：
	1. 估计问题：给定观察序列$O=O_1O_2...O_T$和模型$\mu=(\mathrm{A}, \mathrm{B}, \pi)$，快速计算$P(O\mid \mu)$
	2. 序列问题：给定观察序列$O=O_1O_2...O_T$和模型$\mu=(\mathrm{A}, \mathrm{B}, \pi)$，快速有效选择“最优”状态序列$Q=q_1q_2...q_T$解释观察序列
	3. 训练问题/参数估计问题：给定观察序列$O=O_1O_2...O_T$，如何根据最大似然估计求参数？即如何调节模型$\mu=(\mathrm{A}, \mathrm{B}, \pi)$的参数，使得$P(O\mid \mu)$最大。
	- 解决：前后向算法及参数估计

### 6.4.1 求解观察序列的概率
- 估计问题/解码（decoding）问题：给定观察序列$O=O_1O_2...O_T$和模型$\mu=(\mathrm{A}, \mathrm{B}, \pi)$，快速计算$P(O\mid \mu)$
- 推导：
对于任意的状态序列$Q=q\_{1} q\_{2} \ldots q\_{T}$，有：
$$\begin{aligned} P(O \mid Q, \mu) &=\prod\_{t=1}^{T-1} P(O\_{t} \mid q\_{t}, q\_{t+1}, \mu) \\\\ &=b\_{q\_{1}}(O\_{1}) \times b\_{q\_{2}}(O\_{2}) \times \cdots \times b\_{q\_{T}}(O\_{T}) \end{aligned}$$
并且
$$P(Q \mid \mu)=\pi\_{q\_{1}} a\_{q\_{1} q\_{2}} a\_{q\_{2} q\_{3}} \cdots a\_{q\_{T-1} q\_{T}}$$
由于
$$P(O, Q \mid \mu)=P(O \mid Q, \mu) P(Q \mid \mu)$$
因此
$$\begin{aligned} P(O \mid \mu) &=\sum\_{Q} P(O, Q \mid \mu) \\\\ &=\sum\_{Q} P(O \mid Q, \mu) P(Q \mid \mu) \\\\ &=\sum\_{Q} \pi\_{q\_{1}} b\_{q\_{1}}(O\_{1}) \prod\_{t=1}^{T-1} a\_{q\_{t} q\_{t+1}} b\_{q\_{t+1}}(O\_{t+1}) \end{aligned}$$
- 算法改进：
	- 问题：在N状态、T时间长度时，上述推导需要穷尽$N^T$个所有可能的状态序列，指数爆炸
	- 改进：基于DP的前向算法/前向计算过程（forward procedure），$O(N^2T)$
	- 描述：HMM的DP问题一般用格架（trellis/lattice）的组织形式描述
	![62f4966a8ae3fdbb2eac71ff6ca1fcfc.png](/blog/_resources/01b2e3f7dd37418c8ef4ec724320e793.png)

#### 前向算法
- 前向变量：$\alpha\_{t}(i)=P(O\_{1} O\_{2} \cdots O\_{t}, q\_{t}=s\_{i} \mid \mu)$
- 算法思想：先快速计算前向变量$\alpha_t(i)$，再据此算出$P(O\mid \mu)$
	- 显而易见，$P(O\mid \mu)$为所有T长度的状态下观察序列出现概率和
	- $P(O \mid \mu)=\sum\_{s\_{i}} P(O\_{1} O\_{2} \cdots O\_{\tau}, q\_{T}=s\_{i} \mid \mu)=\sum\_{i=1}^{N} \alpha\_{T}(i)$
- DP思想：t+1的前向变量可以由t时刻所有前向变量归纳计算
	- $\alpha\_{t-1}(j)=(\sum\_{i=1}^{N} \alpha\_{t}(i) a\_{i j}) b\_{j}(O\_{t+1})$
	![43f618a51e5244613909496464f36e89.png](/blog/_resources/c979184de4024acb9b3d387104500ac9.png)
- 前向算法描述（forward procedure）
	1. 初始化：$\alpha\_{1}(\mathrm{i})=\pi b\_{i}(O\_{1}), 1 \leq i \leq N$
	2. 归纳计算：$\alpha\_{i+1}(j)=(\sum\_{i=1}^{N} \alpha\_{t}(i) a\_{i j}) b\_{j}(O\_{t+1}), \quad 1 \leqslant t \leqslant T-1$
	3. 求和终结：$P(O \mid \mu)=\sum\_{i=1}^{N} \alpha\_{T}(i)$
- 复杂度：共T时间，每个时间计算N个前向变量，每个前向变量需要考虑上一时刻的的N个前向变量，所以复杂度为$O(N^2T)$

#### 后向算法
- 与前向算法功能相同，用于快速计算$P(O \mid \mu)$
- 后向变量：$\beta\_{t}(i)=P(O\_{t-1} O\_{t-2} \cdots O\_{T} \mid q\_{t}=s\_{i}, \mu)$
- DP思想：
	- $\beta\_{t}(i)=\sum\_{j=1}^{N} a\_{i j} b\_{j}(O\_{t+1}) \beta\_{t-1}(j)$
	![b7de6f1ccf50dd3dc3964dd9c2f79bef.png](/blog/_resources/b23ab601b50945dda996bd7471712fba.png)
- 后向算法描述（backward precedure）
	1. 初始化：$\beta\_{\mathrm{T}}(\mathrm{i})=1, \quad 1 \leq \mathrm{i} \leq \mathrm{N}$
	2. 归纳计算：$\beta\_{i}(i)=\sum\_{j=1}^{N} a\_{i j} b\_{j}(O\_{t+1}) \beta\_{i+1}(j), \quad T-1 \geqslant t \geqslant 1 ; 1 \leqslant i \leqslant N$
	3. 求和终结：$P(O \mid \mu)=\sum\_{i=1}^{N} \pi\_{i} b\_{i}(O\_{1}) \beta\_{1}(i)$

#### 前后向结合算法
$$\begin{aligned} P(O, q\_{t}=s\_{i} \mid \mu) &=P(O\_{1} \cdots O\_{T}, q\_{t}=s\_{i} \mid \mu) \\\\ &=P(O\_{1} \cdots O\_{t}, q\_{t}=s\_{i}, O\_{t-1} \cdots O\_{T} \mid \mu) \\\\ &=P(O\_{1} \cdots O\_{t}, q\_{t}=s\_{i} \mid \mu) \times P(O\_{t+1} \cdots O\_{T} \mid O\_{1} \cdots O\_{t}, q\_{t}=s\_{i}, \mu) \\\\ &=P(O\_{1} \cdots O\_{t}, q\_{t}=s\_{i} \mid \mu) \times P(O\_{t+1} \cdots O\_{T} \mid q\_{t}=s\_{i}, \mu) \\\\ &=\alpha\_{t}(i) \beta\_{t}(i) \end{aligned}$$

$$P(O \mid \mu)=\sum\_{i=1}^{N} \alpha\_{t}(i) \times \beta\_{t}(i), \quad 1 \leqslant t \leqslant T$$


### 6.4.2 维特比（Viterbi）算法
- 求解序列问题：给定观察序列$O=O_1O_2...O_T$和模型$\mu=(\mathrm{A}, \mathrm{B}, \pi)$，快速有效选择“最优”状态序列$Q=q_1q_2...q_T$解释观察序列
- “最优状态序列”的标准不唯一
	- 使该状态序列的每一个状态都单独具有最大概率：
		- $\gamma\_{t}(i)=P(q\_{t}=s\_{i} \mid O, \mu)$最大
		- 贝叶斯：$\gamma\_{t}(i)=P(q\_{t}=s\_{i} \mid O, \mu)=\frac{P(q\_{t}=s\_{i}, O \mid \mu)}{P(O \mid \mu)}$
		- 前后向算法：$\gamma\_{t}(i)=\frac{\alpha\_{t}(i) \beta\_{t}(i)}{\sum\_{i=1}^{N} \alpha\_{t}(i) \times \beta\_{i}(i)}$
		- 时间t最优状态：$\hat{q}\_{t}=\underset{1 \leqslant \leqslant N}{\operatorname{argmax}}[\gamma\_{i}(i)]$
		- 断序问题：忽略了状态间的关系，可能导致两状态转移概率为0，则最优状态序列不合法
	- 在给定模型$\mu$和观察序列$O$的条件下，使条件概率$P（Q\mid O，\mu）$最大的状态序列
		- $\hat{Q}=\underset{Q}{\operatorname{argmax}} P(Q \mid O, \mu)$
		- 避免了断序问题
		- 维特比算法运用DP搜索求解
- 维特比算法
	- 维特比变量：在时间t时，HMM沿着某一条路径到达状态$s_i$，并输出观察序列$O_1O_2…O_t$的最大概率
		- $\delta\_{t}(i)=\max \_{q\_{1} \cdot q\_{2}, \cdots, q\_{i-1}} P(q\_{1}, q\_{2}, \cdots, q\_{t}=s\_{i}, O\_{1} O\_{2} \cdots O\_{t} \mid \mu)$
	- 递归关系：$\delta\_{t+1}(i)=\max \_{j}[\delta\_{t}(j) \cdot a\_{j i}] \cdot b\_{i}(O\_{t+1})$
	- 算法描述
		1. 初始化：
		$\delta\_{1}(i)=\pi\_{i} b\_{i}(O\_{1}), \quad 1 \leqslant i \leqslant N$
		$\psi\_{1}(i)=0$
		2. 归纳计算：
		$\delta\_{i}(j)=\max \_{1 \leqslant i \leqslant N}[\delta\_{i-1}(i) \cdot a\_{i j}] \cdot b\_{j}(O\_{t}), \quad 2 \leqslant t \leqslant T ; 1 \leqslant j \leqslant N$
		记忆回退路径：
		$\psi\_{t}(j)=\underset{1 \leqslant i \leqslant N}{\operatorname{argmax}}[\delta\_{i-1}(i) \cdot a\_{i j}] \cdot b\_{j}(O\_{t}), \quad 2 \leqslant t \leqslant T ; 1 \leqslant i \leqslant N$
		3. 终结：
		$\hat{Q}\_{T}=\underset{1 \leqslant i \leqslant N}{\operatorname{argmax}}[\delta\_{T}(i)]$
		$\hat{P}(\hat{Q}\_{T})=\max \_{1 \leqslant i \leqslant N}[\delta\_{T}(i)]$
		4. 路径回溯：
		$\hat{q}\_{t}=\psi\_{t+1}(\hat{q}\_{t-1}), \quad t=T-1, T-2, \cdots, 1$
	- 复杂度：易知，与前后向算法一致，为$O(N^2T)$
	- 改进：实际应用常求n-best个最佳路径，在格架每个结点记录m-best（m<n）状态


### 6.4.3 HMM参数估计
- 训练问题/参数估计问题：给定观察序列$O=O_1O_2...O_T$，如何调节模型$\mu=(\mathrm{A}, \mathrm{B}, \pi)$的参数，使得$P(O\mid \mu)$最大：
	- $\underset{\mu}{\arg \max } P(O\_{\text {training }} \mid \mu)$
- 模型参数：构成$\mu$的$\pi_i, a\_{ij}, b_j(k)$
- 可以采用最大似然估计：
	- $\bar{\pi}\_{i}=\delta(q\_{1}, s\_{i})$
		- $\delta(x, y)$为Kronecker函数，x=y时为1，否则为0
	- $\begin{aligned} \bar{a}\_{i j} &=\frac{Q \text { 中从状态 } q\_{i} \text { 转移到 } q\_{j} \text { 的次数 }}{ Q \text { 中所有从状态 } q\_{i} \text { 转移到另一状态(包括 } q\_{j} \text { 自身 }) \text { 的次数 }} \\\\ &=\frac{\sum\_{t=1}^{T-1} \delta(q\_{t}, s\_{i}) \times \delta(q\_{t-1}, s\_{j})}{\sum\_{t=1}^{T-1} \delta(q\_{t}, s\_{i})} \end{aligned}$
	- $\begin{aligned} \bar{b}\_{j}(k) &=\frac{Q \text { 中从状态 } q\_{j} \text { 输出符号 } v\_{k} \text { 的次数 }}{Q \text { 到达 } q\_{j} \text { 的次数 }} \\\\ &=\frac{\sum\_{t=1}^{T} \delta(q\_{t}, s\_{j}) \times \delta(O\_{t}, v\_{k})}{\sum\_{t=1}^{T} \delta(q\_{t}, s\_{j})} \end{aligned}$
	- 由于HMM的状态序列Q无法观察，因此这种最大似然估计方法不可行，可以采用EM算法
- 期望最大化（expectation maximization， EM）算法
	- 可用于含有隐变量的统计模型的参数最大似然估计
	- 基本思想（迭代爬山）：在模型参数限制下随机赋值参数，得到模型$\mu_0$，计算隐变量期望值，用期望替代实际次数（未知）计算新参数值，反复迭代，直到收敛与最大似然估计。
	- 可以达到局部最优
	- Baum-Welch算法或称前向后向算法（forward-backward algorithm）用于具体实现这种EM方法
- Baum-Welch算法/前向后向算法（forward-backward algorithm）
	- 思路：
		- 期望：
			- $\begin{aligned} \hat{\xi}\_{t}(i, j) &=P(q_t=s_i, q\_{t+1}=s_j\mid O, \mu) \\\\&=\frac{P(q\_{t}=s\_{i}, q\_{t-1}=s\_{j}, O \mid \mu)}{P(O \mid \mu)} \\\\ &=\frac{\alpha\_{t}(i) a\_{i j} b\_{j}(O\_{t-1}) \beta\_{t+1}(j)}{P(O \mid \mu)} \\\\ &=\frac{\alpha\_{t}(i) a\_{i j} b\_{j}(O\_{t-1}) \beta\_{t-1}(j)}{\sum\_{i=1}^{N} \sum\_{j=1}^{N} \alpha\_{t}(i) a\_{i j} b\_{j}(O\_{t+1}) \beta\_{t-1}(j)} \end{aligned}$
		![bc623ed5f8d73054090757b08172b47c.png](/blog/_resources/986738539a16439ea2384bb7bab5009f.png)
			- $\gamma\_{t}(i)=\sum\_{j=1}^{N} \hat{\xi}\_{t}(i, j)$
		- 估计：
			- $\bar{\pi}\_{i}=P(q\_{1}=s\_{i} \mid O, \mu)=\gamma\_{1}(i)$
			- $\begin{aligned} \bar{a}\_{i j} &=\frac{Q \text { 中从状态 } q\_{i} \text { 转移到 } q\_{j} \text { 的期望次数 }}{ Q \text { 中所有从状态 } q\_{i} \text { 转移到另一状态(包括 } q\_{j} \text { 自身 }) \text { 的期望次数 }} \\\\ &=\frac{\sum\_{i=1}^{T-1} \xi\_{t}(i, j)}{\sum\_{t=1}^{T-1} \gamma\_{t}(i)} \end{aligned}$
			- $\begin{aligned} \bar{b}\_{j}(k)=& \frac{Q \text { 中从状态 } q\_{j} \text { 输出符号 } v\_{k} \text { 的期望次数 }}{Q \text { 到达 } q\_{j} \text { 的期望次数 }} \\\\ &=\frac{\sum\_{i=1}^{T} \gamma\_{t}(j) \times \delta(O\_{t}, v\_{k})}{\sum\_{t=1}^{T} \gamma\_{t}(j)} \end{aligned}$
	- 算法描述：
		1. 初始化，随机给$\pi_i, a\_{ij}, b_j(k)$赋值，满足约束：
			- $\sum\_{i=1}^{N} \pi\_{i}=1$
			- $\sum\_{j=1}^{N} a\_{i j}=1, 1 \leqslant i \leqslant N$
			- $\sum\_{k=1}^{M} b\_{j}(k)=1, 1 \leqslant j \leqslant N$
		- 得到模型$\mu_0$。令i=0，执行EM估计如下：
		2. EM计算
			- E-步骤：由模型$μ_i$根据期望公式计算期望值$\xi_t(i, j)$和$\gamma_t(i)$；
			- M-步骤：用E-步骤得到的期望值，根据估计公式重新估计参数$\pi_i, a\_{ij}, b_j(k)$的值，得到模型$\mu\_{i＋1}$。
		3. 循环计算：令i＝i＋1。重复执行EM计算，直到$\pi_i, a\_{ij}, b_j(k)$收敛。

- HMM实际应用，注意
	- 多个概率连乘引起浮点数下溢
		- Viterbi算法只涉及乘法和求最大值，可以对概率连乘取对数，避免下溢并加快运算
		- 前向后向算法中，采用$|\log {P}(O \mid \mu\_{i+1})-\log P({O} \mid \mu\_{ {i}})|<\varepsilon$判断收敛。但执行EM计算时有加法运算，这就使得EM计算中无法采用对数运算，在这种情况下，可以设置一个辅助的比例系数，将概率值乘以这个比例系数以放大概率值，避免浮点数下溢。在每次迭代结束重新估计参数值时，再将比例系数取消。


## 6.5 层次化的隐马尔科夫模型hierarchical hidden Markov models, HHMM）
- 提出原因：NLP应用中，因处理序列具有递归特性，尤其长度较大是，HMM复杂度剧增
- HHMM结构：多层随机过程构成。在HHMM中每个状态本身就是一个独立的HHMM，因此一个HHMM的状态产生一个观察序列，而不是一个观察符号。
![61373158789bd743f1b70e89378a7f4c.png](/blog/_resources/b52baf5682d844b49f97875444d9bd5f.png)
- 状态：
	- 终止状态：双圈，用于控制转移过程返回上层状态
	- 生产状态（production state）：只有生产状态才能通过常规HMM机制，即根据输出符号的概率分布产生可观察的输出符号（图中未标出）
	- 内部状态：不直接产生可观察符号的隐藏状态
- 状态转移：
	- 垂直转移（vertical transition）：不同层间转移
	- 水平转移（horizontal transition）：同层转移
- 观察序列的产生：状态转移到某生成，产生一个观察输出后，终止状态控制转移过程返回到激活该层状态转移的上层状态。这一递归转移过程将形成一个生产状态序列，而每个生产状态生成一个观察输出符号，因此生产状态序列将为顶层状态生成一个观察输出序列。
- 形式化描述：
	- 状态$q_i^d(d\in \{1,..., D\})$，i为状态下标，d为层次标号
	- 内部状态转移概率矩阵：$\begin{aligned} A^{q^{d}} &=\{a\_{i j}^{q^{d}}\} \\\\ &=\{P(q\_{j}^{d+1} \mid q\_{i}^{d+1})\} \end{aligned}$，其中$a\_{i j}^{a^{d}}$表示从状态i水平转移到状态j的概率
	- 子状态初始分布矩阵：$\begin{aligned} \Pi^{q^{d}} &=\{\pi^{d}(q\_{i}^{d+1})\} \\\\ &=\{P(q\_{i}^{d+1} \mid q^{d})\} \end{aligned}$
	- 参数输出概率矩阵：$\begin{aligned} B^{q\_{i}^{D}} &=\{b^{q\_{i}^{D}}(k)\} \\\\ &=\{P(\sigma\_{k} \mid q\_{i}^{D})\} \end{aligned}$
	- HHMM参数集合：$\lambda=\{\{A^{q^{d}}\}\{\Pi^{q^{d}}\}\{B^{q^{D}}\}\}$
- 与HMM一样，HHMM也有估计问题、序列问题和训练问题，详见原文[Fine et al., 1998]


## 6.6 马尔科夫网络
![8e7c685f2783e9b1d5ba434dfcba26bd.png](/blog/_resources/f6906badf2044446ab933fb5ad107596.png)
- 马尔科夫网络
	- 无向图模型，可以表示贝叶斯网络无法表示的一些依赖关系，如循环依赖；另一方面，不能表示贝叶斯网络能够表示的某些关系，如推导关系
	- 一组有关马尔科夫性质的随机变量的联合概率分布模型
	- 由无向图G和定义于G上的势函数组成
- 完全子图（complete subgraph）又称团（clique）
	- 团的完全子图称为子团
- 团势能（clique potentials）
	- 无向图不使用条件概率密度对模型进行参数化，使用一种参数化因子：团势能
	- 又称团势能函数/势函数（clique potential function），是定义在团上的非负实函数
	- 每个团对应一个势函数，表示团的一个状态
	- $\mathbf{x}_C$来表示团C中所有的结点，用$\phi(\mathbf{x}_C)$表示团势能。
		- 如图6-11中团：$\mathbf{x}\_{\mathbf{C}\_{1}}=\{x\_{1}, \quad x\_{2}\}, \quad \mathbf{x}\_{\mathbf{C}\_{2}}=\{x\_{1}, \quad x\_{3}, \quad x\_{4}\}$
		- 势能非负，故一般定义 $\phi(\mathbf{x}_C)=\exp(-E(\mathbf{x}_C))$，$E(\mathbf{x}_C)$为$\mathbf{x}_C$的能量函数
- 如果分布P $\phi(x_1，x_2，…，x_n)$的图模型可以表示为一个马尔可夫网络H，当C是H上完全子图的集合时，我们说H上的分布P $\phi(x_1，x_2，…，x_n)$可以用C的团势能函数$\phi(\mathbf{x}_C)$进行因子化：$\phi＝\phi_1(\mathbf{x}\_{C_1}),...,\phi_K(\mathbf{x}\_{C_K})$。P $\phi(x1，x2，…，xn)$可以看作H上的一个吉布斯分布（Gibbs distribution），其概率分布密度为：
$
p(x\_{1}, x\_{2}, \cdots, x\_{n})=\frac{1}{Z} \prod\_{i=1}^{K} \phi\_{i}(\mathbf{x}{C{i}})
$
	- 其中，Z是一个归一化常量，称为划分函数（partition function）。
	- 其中，$x\_{C_i} \subseteq {x_1，x_2，…，x_n}$（1≤i≤K），并且满足$\bigcup\_{i=1}^{K} x\_{C_i}=\{x_1,x_2,…,x_n \}$。
- 显然，在无向图模型中每个$C_i$对应于一个团，而相应的吉布斯分布就是整个图模型的概率分布。
	- 图6-11中的两个团$x\_{C_1}＝{x_1，x_2}$和$x\_{C_2}＝{x_1，x_3，x_4}$就可以定义相应的吉布斯分布，因为满足条件$x\_{C_1} \cup x\_{C_2}＝{x_1，x_2，x_3，x_4}$。
- 因子化的乘积运算可以变成加法运算
	$p(x\_{1}, x\_{2}, \cdots, x\_{n})=\frac{1}{Z} \exp \{-\sum\_{i=1}^{K} E\_{c\_{i}}(x\_{c\_{i}})\}=\frac{1}{Z} \exp \{-E(\mathbf{x})\}$
	- 其中，$\sum\_{i=1}^{K} E\_{C\_{i}}(x\_{C\_{i}})$

## 6.7 最大熵模型
### 6.7.1 最大熵原理
- 熵最大的概率概率分布最真实地反应了事件的分布情况，因为熵最大时随机变量最不确定，最难准确预测其行为。
	- 即：在已知部分信息的前提下，关于未知分布最合理的推断应该是符合已知信息最不确定或最大随机的推断

### 6.7.2 最大熵模型的参数训练
- 最大熵模型参数训练任务：选取有效特征$f_i$及其权重$\lambda_i$
- 各种特征条件和歧义候选可以组合出很多特征函数，必须进行筛选，常用筛选方法：
	1. 选取在训练数据中频次超过一定阈值的候选特征
	2. 互信息
	3. 增量式特征选择方法（比较复杂，不常用）
- 参数$\lambda$获取：通用迭代算法（generalized iterative scaling，GIS）


## 6.8 最大熵马尔科夫模型（maximum-entropy Markov model, MEMM）
- 又称条件马尔科夫模型（conditional Markov model，CMM）
- 结合HMM与最大熵模型特点，广泛用于序列标注问题
- HMM存在问题：
	1. 很多序列标注任务中，尤其当不能枚举观察输出时，需要用大量特征来刻画观察序列
	2. 很多NLP任务中，问题是已知观察序列求解状态序列，HMM采用生成式的联合概率模型（状态序列与观察序列的联合概率$P(S_T, O_T)$）求解这种条件概率问题$P(S_T\mid O_T)$，不适合处理很多特征描述观察序列的情况
- MEMM直接采用条件概率模型$P(S_T\mid O_T)$，使得观察输出可以用特征表示，借助最大熵框架进行特征选取
- HMM与MEMM区别：
	- ![43fde9cbfeb1c8790866c3b967eba975.png](/blog/_resources/83baebd1d5194f1bb102a79f9fe5cab9.png)
	- HMM中$\mu$解码求解的是：$\underset{S\_{T}}{\operatorname{argmax}} P(O\_{T} \mid S\_{T}, \mu)$
	- MEMM中M解码器求解的是：$\underset{S\_{T}}{\operatorname{argmax}} P(S\_{T} \mid O\_{T}, \mu)$
	- HMM当前观察输出只取决于当前状态，MEMM当前观察输出还可能取决于前一时刻的状态
- MEMM思路
	- 概率因子化为马尔可夫转移概率，该转移概率依赖于当前时刻的观察和前一时刻的状态：$P(S\_{1} \cdots S\_{T} \mid O\_{1} \cdots O\_{T})=\prod\_{t=1}^{T} P(S\_{t} \mid S\_{t-1}, O\_{t})$
	- 对于前一时刻每个可能的状态取值$S\_{t－1}＝s'$和当前观察输出$O_t＝o$，当前状态取值$S_t＝s$的概率通过最大熵分类器建模：
	$P(s \mid s^{\prime}, o)=P\_{j}(s \mid o)=\frac{1}{Z(o, s)} \exp (\sum\_{a} \lambda\_{a} f\_{a}(o, s))$
	- $Z(o, s′)$为归一化因子，$f_a(o, s)$为特征函数，$λ_a$为特征函数的
权重，可以利用GIS算法从训练样本中估计出来
	- $f\_{a}(o\_{t}, s\_{t})=f\_{(b, r)}(o\_{t}, s\_{t})=\{\begin{array}{ll}1, & b(o\_{t})=\text { True, } s\_{t}=r \\\\ 0, & \text { 其他 }\end{array}.$
	- HMM中用于参数估计的Baum-Welch算法修改后可用于MEMM的状
态转移概率估计。

- MEMM特点
	- 有向图和无向图的混合模型，主体还是有向图框架。
	- 相比HMM，MEMM最大优点为允许使用任意特征刻画观察序列，这一特性有利于针对特定任务充分利用领域知识设计特征
	- MEMM比起HMM、CRFs训练更高效，HMM和CRF训练需要前后向算法作为内部循环，MEMM估计状态转移概率可以独立进行
	- MEMM缺点：标记偏置问题（label bias）


## 6.9 条件随机场
- 条件随机场（conditional random fields，CRFs）
	- 是用来标注和划分序列结构数据的概率化结构模型
	- 对于给定的输出标识序列Y和观测序列X，CRF通过定义条件概率$P(Y|X)$，而不是联合概率分布$P(X, Y)$来描述模型
	- CRF也可以看作一个无向图模型或者马尔科夫随机场
- CRF定义：无向图每个结点对应随机变量$Y_v$，其取值范围为可能的标记集合$\{y\}$。如果以观察序列X为条件，每一个随机变量$Y_v$都满足以下马尔科夫特性：
	$p(Y\_{v} \mid X, Y\_{w}, w \neq v)=p(Y\_{v} \mid X, Y\_{w}, w \sim v)$
	其中$w\sim v$表示两结点邻近。那么，$(X, Y)$为一个条件随机场
- ![6b4bd8dddc672d85134a00357908cde7.png](/blog/_resources/03c04aa3b77b4539b0c769e77e673552.png)
- CRF也需要解决三类基本问题：特征选取、参数训练和解码
- CRF特点：
	- 相比HMM，主要优点是条件随机性，只需要考虑已经出现的观测状态的特性，没有独立性的严格要求，对于整个序列内部的信息和外部观测信息均可有效利用，避免了MEMM和其他针对线性序列模型的条件马尔可夫模型会出现的标识偏置问题。
	- CRF具有MEMM的一切优点，两者的关键区别在于，MEMM使用每一个状态的指数模型来计算给定前一个状态下当前状态的条件概率，而CRF用单个指数模型来计算给定观察序列与整个标记序列的联合概率。因此，不同状态的不同特征权重可以相互交替代换




