---
{"dg-publish":true,"dg-permalink":"topic-modeling-with-semantic-vectors","permalink":"/topic-modeling-with-semantic-vectors/","tags":["NLP","Semantic_Analysis","Topic_Modeling","LSA","LDA"]}
---


Keyword-based search fails when queries and documents use different vocabulary to express the same concepts. Searching for "automobile" won't retrieve documents containing only "car", despite identical meaning. TF-IDF addresses frequency but not semantics—"happy" and "joyful" remain orthogonal vectors despite synonymy.

Topic modeling addresses this by discovering latent themes that explain word co-occurrence patterns. Rather than representing documents in vocabulary space (one dimension per word), topic modeling represents them in topic space (one dimension per discovered theme). This captures semantic similarity invisible to keyword matching.

## Limitations of TF-IDF

TF-IDF weighs words by frequency and rarity but treats synonyms as unrelated. "Happy" and "joyful" have zero cosine similarity despite near-identical meaning. Polysemy compounds the problem—"bank" (financial institution) and "bank" (river edge) share identical representations despite completely different semantics.

Examples of polysemy challenges:
- **Homonyms**: "band" (music group) vs "band" (hair accessory)
- **Homographs**: "object" (noun) vs "object" (verb)
- **Context-dependent meaning**: "She stole his heart and his wallet"

TF-IDF lacks the semantic structure to disambiguate these cases.


## Topic Vectors

Topic vectors represent documents in a lower-dimensional space where each dimension corresponds to a latent theme rather than individual words. Starting from a BOW or TF-IDF representation in vocabulary space ($|V|$ dimensions), topic modeling projects into topic space ($k$ dimensions where $k \ll |V|$).

Consider three topics: music, fashion, and politics. Words contribute differently to each topic—"guitar" and "concert" load heavily on music, while "vote" and "president" load on politics. A document about a musician-turned-politician contains "guitar", "model", and "president". Its topic representation reflects contributions to each theme:

```python
# Sample topic vector calculation
topic = {}
doc = "The Politician used to be a model before he became a politician, he plays guitar and is a member of a band"
tfidf = dict(list(zip("politician model guitar band".split(), [0.5, 0.1, 0.4, 0.6])))

# Weights for each topic
topic['music'] = 0*tfidf['politician'] + 0*tfidf['model'] + 0.5*tfidf['guitar'] + 0.5*tfidf['band']
topic['politics'] = 0.6*tfidf['politician'] + 0.5*tfidf['model'] + 0*tfidf['guitar'] + 0*tfidf['band']
topic['fashion'] = 0.1*tfidf['politician'] + 0.6*tfidf['model'] + 0*tfidf['guitar'] + 0*tfidf['band']
```

**Resulting Topic Vectors**:

```python
{'music': 0.5, 'politics': 0.35, 'fashion': 0.11}
```

This tells us that this document is primarily about **music** and **politics**, with a smaller connection to **fashion**. With this kind of representation, we can easily compare documents, search based on topics, and understand their overall meaning.


## LSA and LDA

**Latent Semantic Analysis (LSA)** applies Singular Value Decomposition (SVD) to the term-document matrix. SVD factorizes the $m \times n$ matrix (m terms, n documents) into three matrices:

$$X = U\Sigma V^T$$

where $U$ contains term loadings on topics, $\Sigma$ contains singular values (topic strengths), and $V$ contains document loadings on topics. Truncating to the top $k$ singular values produces the $k$-dimensional topic space. Words appearing in similar contexts cluster together—"car" and "automobile" have similar topic vectors despite different surface forms.

**Latent Dirichlet Allocation (LDA)** takes a probabilistic approach. It models each document as a mixture of topics, and each topic as a distribution over words. For document $d$:

1. Draw topic proportions $\theta_d \sim \text{Dirichlet}(\alpha)$
2. For each word:
   - Draw topic $z \sim \text{Multinomial}(\theta_d)$
   - Draw word $w \sim \text{Multinomial}(\beta_z)$

LDA inference determines topic assignments that best explain observed word patterns. Unlike LSA's deterministic SVD, LDA's probabilistic framework handles uncertainty and provides interpretable topic distributions.

Both are unsupervised—they discover topics without labeled examples. LSA is faster and works well for semantic search. LDA produces more interpretable topics useful for exploration and summarization.

## Applications

Topic vectors enable semantic search (retrieving documents by meaning rather than keywords), document similarity (measuring relatedness despite different vocabulary), and keyword extraction (identifying representative terms per topic). By reducing dimensionality from $|V|$ to $k$ dimensions, topic modeling improves efficiency while capturing latent semantic structure invisible to keyword-based methods.

