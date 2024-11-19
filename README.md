# Series Natual Language Processing from Hero to Zero

## Day 1: Word2vec

Statistically-based vs Inference-based (word2vec) 

-> statistics : encoding the similarity of words, 
recompute them from the beginning when data are renewed

-> inference : similarity + encoding by identifying patterns between words, 
learn again with parameters when data are renewed 

- CBOW vs skip-gram

-> Word precision - skip-gram

-> Speed of learning - CBOW

Negative Sampling

not just a positive yes (correct answer),
binary classification that also takes into account
negative yes (incorrect answers)
- Correct answer -> close to output 1
- Wrong answer - > close to output 0
- And proper weights to produce these results
- Learning all negative examples..? -> Select only a few

## Day 2: Recurrent Neural Network