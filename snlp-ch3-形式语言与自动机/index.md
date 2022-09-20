# 《统计自然语言处理》第3章 - 形式语言与自动机


## ch3 形式语言与自动机
注：本章笔记参考（王柏、杨娟）《形式语言与自动机》
### 3.1 基本概念
图、树、字符串

### 3.2 形式语言
#### 语言定义和运算
- 字母表 T：字符的有限集合。
- 字符串：T中字符构成序列。
- 字符串运算：concatenation, 逆 $\omega^T$ 或 $\overline{\omega}$、幂、闭包（$T^*, T^+$）
- 语言：设T为字母表，任何**集合**$L \subseteq T^*$是字母表T上的一个语言。
- 语言运算：并、交、补、差、积、幂

#### 文法
（1）概念
- 文法：定义语言的数学模型
- 表示方法：
  - 有限集合：列举法
  - 无线集合：文法产生系统、机器识别系统
- 元语言：描述语言的语言，文法是一种元语言
- 对象语言：描述的语言

（2）Chomsky文法体系
- 可被替代 ->
- G = (N, T, P, S)
  - N 非终结符的有限集合
  - T 终结符的有限集合 
  - P 生成式有限集合
  - S 起始符

（3）推倒与句型
- 直接推导：由生成式$A\rightarrow \beta$得直接推导：$\alpha A\gamma \Rightarrow \alpha\beta\gamma$
- 推导序列：称$\alpha\_{0}\Rightarrow\alpha\_{1}\Rightarrow\ldots\Rightarrow\alpha\_{n}$长度为n的推导序列
- 推导出：$\alpha \xRightarrow[G]{*} \alpha^{\prime}$, $\alpha \xRightarrow[G]{+} \alpha^{\prime}$
- 句型：推导序列每一步产生的字符串
- 句子：只含有终结符句型
- 语言：句子的集合

（4）Chomsky文法分类
按产生式的形式分类：

| 分类    | 别称              |特点     | 对应语言       | 对应自动机     |
| ------- | ---------------|------- | -------------- | -------------- |
| 0型文法 | 无限制文法PSG    |     无限制  | 递归可枚举语言 | 图灵机TM         |
| 1型文法 | 上下文有关文法CSG|       左长小于右  | 上下文有关语言CSL | 线性有界自动机LBA |
| 2型文法 | 上下文无关文法CFG |    左长等于1    | 上下文无关语言CFL | 下推自动机PDA     |
| 3型文法 | 正则RG、左/右线性RLG/LLG|  左/右线性 | 正则语言RL      | 有限自动机FA     |

关系：$L\left(G\_{0}\right) \supseteqq L\left(G\_{1}\right) \supseteqq L\left(G\_{2}\right) \supseteqq L\left(G\_{3}\right)$

#### CFG识别句子的派生树
- 派生树也称语法树（syntactic tree）、分析树（parsing tree）、推导树
- ![ced65054c416e6cbd535089e2171d7a4.png](/_resources/b4612b03745f49809e94b8e93cac2093.png)
- 二义性文法：文法G对于同一个句子的分析树 >= 2


### 3.3 自动机
#### （1）有限自动机FA
- DFA与NFA

#### （2）正则文法与自动机
- RG <-> FA

#### （3）CFG与下推自动机PDA
- ![f5bdc4ae48dc4b81d634b31c29955b88.png](/_resources/5f05d95a47474b2c87311952f2d443d1.png)
- CNF(Chomsky Normal Form)文法格式：$A \rightarrow BC \mid a$
- 2型文法（CFG）可以转换为等价CNF
- CFG <-> PDA

#### （4）图灵机TM
- 图灵机与双向有限自动机的区别：图灵机可以改变“带(tape)”上的符号
- 0型文法 <-> TM

#### （5）线性限界自动机LBA
- LBA：确定的单带图灵机，其读／写头不能超越原输入带上字符串的初始和终止位置

- 各类自动机的区别：信息存储空间的差异。
	- FA：状态
	- PDA：状态 + 堆栈
	- LBA：状态 + 输入/输出带
	- TM：无限制

### 3.4 自动机在NLP中的应用
- 有限自动机又称为有限状态机（finite state machine, FSM）

#### 单词拼写检查
- [Oflazer,1996]FA用于拼写检查，[Damerau,1964]最小编辑距离

#### 单词形态分析
- 有限状态转换机（finite state transducer, FST）
	- FST在状态转移时输出，而FA/FSM只转移，不输出
- ![06b65a903a22905c54b4310958db0eac.png](/_resources/d77b3664794e4a9c8fcebaecd69fa676.png)
	- 识别heavy单词原型
	- 产生如下两条关于单词heavy的形态分析规则：
		- heavier→heavy＋er
		- heaviest→heavy＋est

#### 词性消歧（part-of-speech tagging）
- 词性标注方法之一：FST [Roche and Schabes, 1995]
	1. 词性标注规则 -> FST
	2. FST -> 扩展为全局操作
	3. 合并FST为一个
	4. 将FST转化为确定的FST
