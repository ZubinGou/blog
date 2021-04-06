# 【NLP Papers】ELMo: Deep contextualized word representations


[Peters et al., NAACL 2018a]

- use bidirectional language model to train contextual word vector.
- use these vector as pre-train part of existing models, improve SOTA across six tasks.
- analysis showing that exposing the deep internals of the pre-trained network is crucial

## 1 Introduction
pre-trained word representations should model both:
1. complex characteristic of word use (e.g., syntax and semantics)
2. how these uses vary across linguistic contexts (i.e., to model polysemy)

ELMo: Embeddings from Language Models

capture:
- higher-level LSTM states: context-dependent meaning
- lower-level LSTM states: syntax
	- e.g., can be used for POS tagging

## 2 Related work
1. "word type" embeddings
	- (Turian et al., 2010; Mikolov et al., 2013; Pennington et al., 2014)
2. +subword 
	- (Wieting et al., 2016; Bojanowski et al., 2017)
3. "word sense" embeddings
	- (Neelakantan et al., 2014)
4. context-dependent representations
	- context2vec (Melamud et al., 2016)
	- CoVe (McCann et al., 2017) 
	- (Peters et al., 2017)
5. different layers of biRNNs encode different information
	- (Hashimoto et al., 2017)
	- (Søgaard and Goldberg, 2016)
	- (Belinkov et al. 2017)
	- (Melamud et al.,2016) 
6. pretrain
	- (Dai and Le, 2015)
	- (Ramachandran et al. 2017)


## 3 ELMo: Embeddings from Language Models
ELMo word representations are functions of the entire input sentence, computed on top of two-layer biLMs with character convolutions, as a linear function of the internal network states.

semi-supervised:
1. biLM is pretrained as a large scale
2. easily incorporated into a wide range of existing neural NLP architectures.

### 3.1 Bidirectional language models
$$
\begin{array}{l}
\sum\_{k=1}^{N}\left(\log p\left(t\_{k} \mid t\_{1}, \ldots, t\_{k-1} ; \Theta\_{x}, \vec{\Theta}\_{L S T M}, \Theta\_{s}\right)\right. \\\\
\left.\quad+\log p\left(t\_{k} \mid t\_{k+1}, \ldots, t\_{N} ; \Theta\_{x}, \overleftarrow{\Theta}\_{L S T M}, \Theta\_{s}\right)\right)
\end{array}
$$

- share some weights between directions

### 3.2 ELMo
a L-layer biLM computes a set of $2L+1$ representations:
$$
\begin{aligned}
R\_{k} &=\left\{\mathbf{x}\_{k}^{L M}, \overrightarrow{\mathbf{h}}\_{k, j}^{L M}, \overleftarrow{\mathbf{h}}\_{k, j}^{L M} \mid j=1, \ldots, L\right\} \\\\
&=\left\{\mathbf{h}\_{k, j}^{L M} \mid j=0, \ldots, L\right\}
\end{aligned}
$$

collapses all layers in $R$ int a single vector:
$$
\mathbf{E L M o}\_{k}=E\left(R\_{k} ; \mathbf{\Theta}\_{e}\right)
$$
1. simplest way (as in TagLM (Peters et al., 2017) and CoVe (McCann et al., 2017).):
$$
E\left(R\_{k}\right)=\mathbf{h}\_{k, L}^{L M}
$$
2. task specific weighting:
$$
\mathbf{E L M o}\_{k}^{t a s k}=E\left(R\_{k} ; \Theta^{t a s k}\right)=\gamma^{t a s k} \sum\_{j=0}^{L} s\_{j}^{t a s k} \mathbf{h}\_{k, j}^{L M}
$$

$\mathbf{s}^{\text {task }}$ are softmax-normalized weights and scalar parameter $\gamma^{\text {task }}$ allows the task model to scale the entire ELMo vector.

### 3.3 Using biLMs for supervised NLP tasks
1. run biLM to get layer representations for each word
2. let the end task learn a linear combination of these representations.

freeze weights of biLM and pass the ELMo enhanced representation $\left[\mathbf{x}\_{k} ; \mathbf{E L M o}\_{k}^{\text {task }}\right]$ into the task RNN.

observe further improvements by also including ELMo at the output of the task RNN by introducing another set of output specific linear weights and replacing $\mathbf{h}_k$ with $\left[\mathbf{h}\_{k} ; \mathbf{E L M o}\_{k}^{\text {task }}\right]$

prevent overfit:
- moderate amount of dropout
- adding $\lambda\|\mathbf{w}\|\_{2}^{2}$ to the loss

### 3.4 Pre-trained bidirectional language model architecture
similar to architectures in J´ozefowicz et al. (2016) and Kim et al. (2015)
- modified to support joint training of both directions
- residual connection

halved all embedding and hidden dimensions from the single best model CNN-BIG-LSTM in J´ozefowicz et al (2016). 
> - L = 2 biLSTM layers with 4096 units and 512 dimension projections
> - residual connections

context insensitive type representation:
> - 2048 character n-gram convolutional filters followed by two highway layers  (Srivastava et al., 2015) 
> - linear projection down to a 512 representation


## 4 Evaluation
![e857175ff8b1434e0ca07576a6ef8140.png](../../../_resources/519908af977a440bb92bcfaa123a96ed.png)
- Question answering
- Textual entailment
- Semantic role labeling
- Coreference resolution
- Named entity extraction
- Sentiment analysis

## 5 Analysis
1. deep contextual representations works better than just top layer
2. syntactic information at lower layers while semantic information as higher layers

### 5.1 Alternate layer weighting schemes
- previous word on contextual representations: only used last layer, whether it be from biLM or MT encoder
- ablation study:

![52d200d883248a661a1e28760d123c77.png](../../../_resources/8e4ce8032cc841e2a96e39dd8e14c9b0.png)

### 5.2 Where to include ELMo?
- word emebedings only as input to the lowest layers
- however, including ELMo at the output of biRNN improves for some tasks

### 5.3 What information is captured by the biLM's representations?
![06099455f4f52e8635fb943f2b9ff91d.png](../../../_resources/448a8b55c99641acb533735a3340e67a.png)

**Word sense disambiguation**
![51dcd3d27b1373cb35b4b8ed9051cd2c.png](../../../_resources/7f6d5f47921c4db388824fbe0e1ad183.png)

**POS tagging**
- context representations as input to a linear classifier of POS
![37620ab059acf7212952defddfed2955.png](../../../_resources/f0b8b1f0c65b4a8087012fdd5f81c2cd.png)

**Implications for supervised tasks**
- including all biLM layers is important for downstream tasks

### 5.4 Sample efficiency
1. using ELMo increases the sample efficiency.
2. ELMo-enhanced models use smaller training sets more efficiently than those without.
![c1ce4981b57b22252bc93a70c5cb365e.png](../../../_resources/38134ddcfb9949eda5147b969a382c73.png)

### 5.5 Visualizatino of learned weights
visualize the ELMo learned weights across the tasks
![24771a3b35ca56a32be3ca4ed7f70e0a.png](../../../_resources/260c5d9f72c8478c901fdc8294a37a62.png)


