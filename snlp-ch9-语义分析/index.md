# 《统计自然语言处理》第9.1章 - 语义分析


# ch9 语义分析
- NLP最终目的一定程度上是在语义理解的基础上实现响应的操作
- 语义计算十分困难：模拟人脑思维过程，建立语言、知识与客观世界之间的可计算逻辑关系，并实现具有高区分能力的语义计算模型，至今未解
- 语义分析任务
	- 词层次：词义消歧（word sense disambiguation，WSD）
	- 句子层面：语义角色标注（semantic role labeling，SRL）
	- 篇章层面：指代消歧/共指消歧（coreference resolution）、篇章语义分析


## 9.1 词义消歧概述
- 最小语用单位：词
- 词义消歧任务：确定一个多义词在给定的上下文语境中的具体含义
- 早期：规则
- 20世纪80年代后：统计
	- 有监督的消歧方法（supervised disambiguation）
	- 无监督的消歧方法（unsupervised disambiguation）
- 统计消歧基本观点：一个词的不同语义一般发生在不同上下文
	- 有监督：词语义的上下文分类问题（classification task）
	- 无监督（clustering task）
		1. 聚类算法对同一个多义词的所有上下文进行等价类划分
		2. 识别时，将上下文与各词义等价类比较
- 基于词典信息的消歧方法（dictionary-based disambiguation）
- 测试数据：为避免手工标注的困难，采用制造人工数据的方法获取大规模训练和测试数据。
	- 制造的人工数据称为伪词（pseudoword），[Manning and Schütze，1999]基本思路是将两个词汇合并，如banana-door，替代所有语料中的banana和door（消歧就是判断到底是哪个词）
