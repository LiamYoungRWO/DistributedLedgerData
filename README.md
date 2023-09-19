---
description: >-
  Developer tooling and resource series summarizing the eight foundational
  algorithms most commonly used in data science and with relational segments
  integrating Solana blockchain data.
cover: .gitbook/assets/AdobeStock_241388513.jpeg
coverY: 0
---

# 📊 Data Science Applications Introduction

These algorithms are key to understanding various aspects of data science, including data exploration, classification, clustering, and regression.



Probably gonna ax this version and redo the repo from the ground up in the next couple of days. Let's test some code blocks in here though to see if they translate

```python
// Some code# Import necessary libraries
import json
import requests
from gensim.models import Word2Vec
import networkx as nx
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Obtain Solana Blockchain Data (Mock Example)
# Normally you'd use Solana's API or another method, but let's mock some data for illustration
solana_data = [
    {'transaction': 'send', 'from': 'A', 'to': 'B', 'amount': 10},
    {'transaction': 'send', 'from': 'B', 'to': 'C', 'amount': 5},
    # ... add more transactional data
]

# Step 2: Preprocess Data
# For Word2Vec
sentences = [[str(item[key]) for key in item] for item in solana_data]

# For Node Embedding
G = nx.Graph()
for item in solana_data:
    G.add_edge(item['from'], item['to'], weight=item['amount'])

# Step 3: Generate Embeddings

# Word2Vec
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)

# Node Embedding using some method (here we mock it)
node_embeddings = {}  # Normally you'd use a library to learn these

# Step 4: Parse and Utilize Embeddings

# Example: find most similar transaction to a given one using Word2Vec
print(word2vec_model.wv.most_similar('send'))

# Example: Utilize Node Embeddings (this part is mocked)
print(node_embeddings.get('A'))

# For visualization, let's consider PCA for Word2Vec vectors for words 'send', 'A', 'B'
words = ['send', 'A', 'B']
vectors = [word2vec_model.wv[word] for word in words]
pca = PCA(n_components=2)
result = pca.fit_transform(vectors)
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()

```
