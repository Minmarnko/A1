# NLP A1
 AIT NLP Assignment 1

- [Student Information](#student-information)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Training Data](#training-data)
- [Word Embedding Models Comparison](#word-embedding-models-comparison)
- [Similarity Scores](#similarity-scores)
- [Model Comparison Report](#model-comparison-report)

## Student Information
Name - Min Marn Ko  
ID - st125437

## Installation and Setup
Webapp at localhost:8501
streamlit

## Usage
Enter one or more input words and the website displays most similar top 10 words from  each model's vocabulary.

## Training Data
Corpus source - nltk datasets('abc') : Austrailian Broadcasting Commission  
Token Count |C| - 134349 
Vocabulary Size |V| - 9775  
Embedding dimension - 50  
Learning rate - 0.001  
Epochs - 1000 

Training parameters are consistant across all three models.  

## Word Embedding Models Comparison

| Model             | Window Size | Training Loss | Training Time | Syntactic Accuracy | Semantic Accuracy |
|-------------------|-------------|---------------|---------------|--------------------|-------------------|
| Skipgram          | 2     | 24.10       | 6 min 40 sec       | 0.00%            | 0.77%           |
| Skipgram (NEG)    | 2     |  2.07       | 4 min 03 sec       | 0.00%            | 1.73%           |
| Glove             | 2     |  6.02       | 1 min 41 sec       | 0.00%            | 1.73%           |
| Glove (Gensim)    | -     | -       | -       | 55.45%            | 93.87%           |

## Similarity Scores

| Model               | Skipgram | Skipgram (NEG) | GloVe | GloVe (Gensim) | Y true |
|---------------------|-----------|----------------|-------|----------------|--------|
| **Spearman Correlation**             | 0.0932   | 0.1546        | -0.0090 | 0.5000        |  |


## Model Comparison Report
The loss trends across all three models indicate that they failed to reach convergence, as observed in the training graphs. This lack of convergence can be attributed to the limited corpus size and insufficient hyperparameter tuning. Despite these limitations, the performance of Skipgram, Skipgram (Negative Sampling), and GloVe showed incremental improvements, reflecting the advantages of their respective architectural designs.

In terms of training time, the Skipgram models exhibited similar durations, with Skipgram (Negative Sampling) slightly faster. The GloVe model stood out with a significantly shorter training time of just 1 minute and 41 seconds, thanks to its superior computational efficiency (\( O(|C|^{0.8}) \)) compared to Skipgram’s complexity of \( O(|V|) \).

The models trained from scratch achieved poor results in syntactic and semantic accuracy, with syntactic accuracy remaining at 0% for all three models and semantic accuracy peaking at 1.73%. These outcomes are likely influenced by the limited and reduced corpus, with the GloVe model further constrained by memory restrictions during weight calculations.

Similarity test results further underscore the benefits of pretrained models. The pretrained GloVe model, implemented via Gensim, achieved a Spearman correlation score of 0.5962, closely aligning with human-level performance. In contrast, the best-performing from-scratch model, Skipgram (Negative Sampling), achieved a lower score of 0.1546, demonstrating the significant advantage of using pretrained embeddings for more reliable results.
