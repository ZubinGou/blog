# 【NLP Papers】Contextual Word Representations: A Contextual Introduction


Word Representations 综述
[Noah A. Smith, 2020]

## 1 Preliminaries
两种word定义：
1. word token：word observed in a piece of text
2. word type: distinct word, rather than a specific instance

每个word type可能有多个word token实例。

## 2 Discrete Words
1. simplest representation of text: sequence of characters
2. integerized: give each word type a unique integer value >=0, advantages:
	- every word type was stored in the same amount of memory
	- array-based data structures could be used to index other information by word types

we refer to integer-based representations of word types as **discrete representations**

## 3 Words as Vectors
非discrete起源，从实际应用出发：
- 文本分类
- 机器翻译：用 word token 作为翻译凭据
- 在上下文中给定evidence，选择 word type 输出 ？

discrete representations 无法在词之间共享信息，词之间无法比较相似性。

many strands:
- WordNet(Fellbaum, 1998)： a lexical database that stores words and relationships among them such as synonymy and hyponymy
- part of speech
- use program to draw informatin from corpora

use these strands to derive a notion of a word type as a **vector**, dimensions are features:
- one hot
- use a dimension to mark a known class (e.g. days of the week)
- use a dimension to place variants of word types in a class.
- use surface attributes to "tie together" word type that look similar e.g. capitalization patterns, lenths, and the presence of a digit
- allocate dimensions to try to capture word types' meanings e.g. in "typical weight" *elephant* get 12,000 and *cat* get 9.

feartures from: 
- experts
- derived using automated algorithms


## 4. Words as Distributional Vectors: Context as Meaning
idea: words used in similar ways are likely to have related meanings.

*distributional* view of word meaning: looking at the full distribution of contexts in corpus where $w$ is found.

approchs:
- hierarchical clustering, Brown et al. (1992)  (highly successful)
- word vectors with each dimension corresponded to the frequency the word type occurred in some context (Deerwester et al., 1990), dimensionality reduction is applied.

*vector space semantics* (see Turney and Pantel, 2010
for a survey): v(man) - v(woman) = v(king) - v(queen)

cons of reduced-dimensionality vectors: features are not interpretable

the word's meaning is distributed across the whole vector, that is **distributed representations**

scalability problems

**word2vec**

a common pattern: construct word vectors and publish them for everyone to use.

interesting ideas:
- Finutuning rather than "learning from scratch"
- use expert-build data structures. e.g. retrofitting with WordNet
- use bilingual dictionaries to "align" the vectors
- use character sequence to build vectors

## 5. Contextual Word Vectors
idea: words have different meaning in different context（一词多义）

- from "word type vectors" to "word token vectors"
	- similar meaning words are easy to find for word token in context

- **ELMo**: embeddings from language models (Peters et al., 2018a)
	- use NN to contextualize word type vector to word token vector
	- optimization task: language modeling (next word prediction)
- **ULMFiT**: (Howard and Ruder, 2018)
	- benefit for text classification
- **BERT**: (Devlin et al., 2019)
	- innovations to the learning method and learned from more data
- GPT-2 Radford et al. (2019)
- RoBERTa Liu et al. (2019b)
- T5 Raffel et al. (2019)
- XLM Lample and Conneau (2019)
- XLNet Yang et al. (2019)


## 6. Cautionary Notes
- Word vectors are biased
	- ME: Isn't bias sometimes knowledge?
- Language is a lot more than words
- NLP is not a single problem
	- evaluation is important

## 7. What's Next
- variations on contextual word vectors to new problems
- modifications to the learning methods
- improving preformance in setting where little supervision is available
- computatoinally less expensive
- characterize the generalizations that these meth-
ods are learning in linguistic terms

## 8. Further Reading
linguistics:
- Emily M. Bender. Linguistic Fundamentals for Natural Language Processing: 100 Essentials from Morphology and Syntax. Morgan & Claypool, 2013
- Emily M. Bender and Alex Lascarides. Linguistic Fundamentals for Natural Language Processing II: 100
Essentials from Semantics and Pragmatics. Morgan & Claypool, 2019.
- (Sections 1–4 chapter 14 of) Jacob Eisenstein. Introduction to Natural Language Processing. MIT Press, 2019.

contextual word vectors original papers:
- EMLo: Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. In Proceedings of NAACL, 2018a.
- BERT: Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proc. of NAACL, 2019.

## 其他
文本表示方法：
- bag-of-words：one-hot，tf-idf，textrank
- 主题模型：LSA(SVD)，pLSA，LDA
- 静态词向量：word2vec，fastText，GloVe
- 动态词向量：ELMo，GPT，BERT
