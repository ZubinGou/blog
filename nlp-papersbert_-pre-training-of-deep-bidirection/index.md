# 【NLP Papers】BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding



[Devlin et al., NAACL 2019]

BERT: Bidirectional Encoder representations from Transformers

## 1 Introduction
two pre-train strategies:
1. feature-based
	- ELMo: task-specific architecture
2. fine-tuning
	- GPT

- limitations: standard language models are unidirectional
- masked language model (MLM, inspired by Cloze task)
- use a "next sentence prediction" task that jointly pretain text-pair representations


## 2 Related Work
### 2.1 Unsupervised Feature-based Approaches
from word2vec to ELMo...

### 2.2 Unsupervised Fine-tuning Approaches
GPT use left-to-right language modeling and auto-encoder objectives

### 2.3 Transfer Learning from Supervised Data


## 3 BERT
two steps:
1. pretraining
2. fine-tuning

WordPiece embeddings

### 3.1 Pre-training BERT
**Task #1: Masked LM**
- mask 15% of all WordPiece tokens in each sequence at random.
- mismatch if [MASK] between pre-training and fine-tuning 
	- 80%: [MASK]
	- 10%: random token
	- 10%: unchanged

**Task #2: Next Sentence Prediction(NSP)**
- purpose: many tasks such as QA and NLI are based on two sentences.
- pre-train for a binarized next sentence prediction
	- 50% IsNext
	- 50% NotNext
- final model achieves 97%-98% accuracy on NSP
- BERT transfers all parameters to initialize end-task model parameters

### 3.2 Fine-tuning BERT
input and output for different tasks

## 4 Experiments
GLUE, SQuAD v1.1, SQuAD v2.0, SWAG

## 5 Ablation Studies
- Effect of Pre-training Tasks
- Effect of Model Size
- Feature-based Approach with BERT

## 6 Conclusion
![88119d5820069cd6ea1914c6049a9682.png](/blog/_resources/e701f88d745347bab49f75565d6430cd.png)

![d076d4b7ae2b58b370bc36b79dee3ac6.png](/blog/_resources/4276e9de919f4c90aca49affa924e24c.png)
