---
{"dg-publish":true,"dg-permalink":"transforming-text-into-numbers-for-machine-learning","permalink":"/transforming-text-into-numbers-for-machine-learning/","tags":["NLP","Vectorization","Word_Embeddings","TF-IDF","BOW"]}
---


Machine learning models operate on numerical data. Text, however, consists of discrete symbols (words) with no inherent numerical representation. Vectorization converts text into vectors—arrays of numbers that preserve semantic and syntactic information while enabling numerical computation.

The challenge is preservation of meaning. A naive mapping (assigning each word a unique integer) discards relationships between words. Better representations capture similarity ("dog" and "puppy" are close), context ("bank" means different things in "river bank" vs "savings bank"), and frequency (common words like "the" carry less information than rare domain-specific terms).

Consider three sentences:

1. "I love NLP"
2. "I hate NLP"
3. "I enjoy NLP"

Converting these to vectors requires choosing a representation scheme that balances simplicity, computational efficiency, and information preservation.

## One-Hot Encoding

One-hot encoding assigns each word a unique position in a vocabulary-sized vector. For vocabulary `["I", "love", "hate", "enjoy", "NLP"]`:

- "I": `[1, 0, 0, 0, 0]`
- "love": `[0, 1, 0, 0, 0]`
- "hate": `[0, 0, 1, 0, 0]`
- "enjoy": `[0, 0, 0, 1, 0]`
- "NLP": `[0, 0, 0, 0, 1]`

We can represent each sentence as a collection of these one-hot vectors:

```python
# Sentence: "I love NLP"
[[1, 0, 0, 0, 0],  # "I"
 [0, 1, 0, 0, 0],  # "love"
 [0, 0, 0, 0, 1]]  # "NLP"
```

Limitations are severe. For vocabulary size $|V|$, each word requires a $|V|$-dimensional vector—10,000 words means 10,000-dimensional vectors that are 99.99% zeros (sparse). More critically, one-hot encoding treats all words as equally dissimilar. The distance between "love" and "hate" equals the distance between "love" and "giraffe", discarding semantic relationships.

## Bag of Words

Bag of Words (BOW) represents documents by word frequency, discarding order and grammar. Each document becomes a vector counting occurrences of vocabulary words.

Here’s how it looks with our three sentences:

```python
from sklearn.feature_extraction.text import CountVectorizer

sentences = [
    "I loved the movie",
    "Movie was awesome, movie characters were awesome",
    "Movie was terrible"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

**Output:**

```python
['awesome', 'characters', 'loved', 'movie', 'terrible', 'the', 'was', 'were']
[[0 0 1 1 0 1 0 0]
 [2 1 0 2 0 0 1 1]
 [0 0 0 1 1 0 1 0]]
```

In this example, the words `"movie"`, `"awesome"`, and `"was"` show up multiple times, and each sentence is converted into a vector of word counts.

BOW captures word frequency but loses order and context. "I hate NLP" and "I love NLP" have similar BOW vectors despite opposite sentiment. This limits effectiveness for tasks requiring semantic understanding.

## TF-IDF

Term Frequency-Inverse Document Frequency (TF-IDF) weighs words by importance. Frequent words in a document score high (Term Frequency), but words common across all documents score low (Inverse Document Frequency).

Term Frequency (TF) is calculated as follows:

$$TF(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

where:

- $f_{t,d}$ is the frequency of term $t$ in document $d$.
- $\sum_{t' \in d} f_{t',d}$ is the total number of terms in document $d$.

Inverse Document Frequency (IDF) is calculated as follows:

$$IDF(t, D) = \log\left(\frac{N}{n_t}\right)$$

where:

- $N$ is the total number of documents.
- $n_t$ is the number of documents containing term $t$.

The final TF-IDF score is the product of TF and IDF:

$$TFIDF(t, d, D) = TF(t, d) \times IDF(t, D)$$

Here’s how TF-IDF works in action:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

sentences = [
    "I love NLP",
    "I hate NLP",
    "I enjoy NLP"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

**Output:**

```python
['enjoy', 'hate', 'love', 'nlp']
[[0.         0.         0.861037   0.50854232]
 [0.         0.861037   0.         0.50854232]
 [0.861037   0.         0.         0.50854232]]
```

As you can see, TF-IDF gives higher importance to the unique words ("love," "hate," "enjoy") while giving less weight to common words like "NLP" that appear in all three sentences.

## Word Embeddings

One-hot encoding, BOW, and TF-IDF produce sparse vectors that don't capture semantic relationships. Word embeddings (Word2Vec, GloVe, BERT) create dense vectors where similar words cluster in vector space.

In embedding space, "king" and "queen" are close, as are "man" and "woman". The difference vector ("king" - "man") approximates ("queen" - "woman"), capturing the gender relationship. This semantic structure enables analogical reasoning impossible with sparse representations.

## Document Similarity

Vectorized documents can be compared using cosine similarity. For recommendation systems, this finds documents similar to a user's current reading:

```python
from sklearn.metrics.pairwise import cosine_similarity

sentences = [
    "Cat is drinking milk",
    "Dog is running behind the cat",
    "A man is riding a horse",
    "Cat is playing with the mouse",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

new_doc = "Cat and mouse are playing"
Y = vectorizer.transform([new_doc])

cs = cosine_similarity(X, Y)

sorted_index = np.argsort(cs, axis=0)[::-1].flatten()
for idx in sorted_index:
    print(sentences[idx], cs[idx])
```

**Output:**

```python
Cat is playing with the mouse [0.70710678]
Cat is drinking milk [0.28867513]
Dog is running behind the cat [0.23570226]
A man is riding a horse [0.]
```

Cosine similarity ranks documents by angle in vector space. The query "Cat and mouse are playing" most closely matches "Cat is playing with the mouse" (similarity 0.707), followed by "Cat is drinking milk" (0.289).

## Conclusion

Vectorization techniques trade off simplicity, computational cost, and semantic richness:

- **One-hot encoding**: Simple but sparse, treats all words as equally dissimilar
- **Bag of Words**: Captures frequency, loses order
- **TF-IDF**: Weighs by importance, filters common words
- **Word embeddings**: Dense representations capturing semantic relationships

For tasks requiring semantic understanding (sentiment analysis, question answering), embeddings are necessary. For simpler tasks (spam classification, keyword matching), BOW or TF-IDF suffice and are computationally cheaper.

The choice depends on task requirements, dataset size, and available computational resources.