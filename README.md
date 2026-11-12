# Neural Networks and Deep Learning Assignment 3

## Student Information

- **Name** — Suruchi Maharjan
- **Student ID** — [700797006]
- **Course** — Neural Networks and Deep Learning
- **Professor** - [Tsai, I Hua]
- **University** — University of Central Missouri


## Overview

This assignment explores natural language processing (NLP) preprocessing and attention in Transformer models. It includes written answers about sequence-to-sequence models and Transformer architecture, two Python programming exercises, and a step-by-step self-attention calculation based on the lecture slides.

## Questions and Answers

### Part I. Short Answer Questions

1. **How do sequence-to-sequence models use attention?**  
   The encoder represents the source sentence, and the decoder generates the target sentence one token at a time. Attention lets the decoder access the encoder’s hidden states at each decoding step, reducing the information bottleneck of a single fixed-length context vector.

2. **What is the difference between hard and soft attention?**  
   Hard attention makes a discrete selection, while soft attention combines hidden states using attention weights. Soft attention is differentiable, which makes it easier to train with backpropagation. For the exercise, the dot-product scores are `[1, 0, 1]`, the rounded softmax weights are `[0.422, 0.155, 0.422]`, and the context vector is `[0.845, 0.578]` when calculated using the unrounded weights.

3. **How does self-attention use queries, keys, and values?**  
   Self-attention relates tokens within the same sequence. A query is compared with the keys to compute attention scores. These scores determine the weights used to combine the value vectors. This allows distant words to influence each other’s representations.

4. **How does multi-head attention work?**  
   Multiple attention heads learn different patterns of relationships between tokens. With an embedding dimension of 512 and eight equally sized heads, each head has `512 / 8 = 64` dimensions. The head outputs are concatenated and passed through a learned output projection.

5. **Why do Transformers use positional encodings and causal masks?**  
   Positional encodings provide token order. A causal mask prevents decoder positions from attending to future tokens. Transformers can process known positions in parallel during training, while autoregressive generation produces one token at a time.

### Part II. Programming Questions

#### Q1. NLP Preprocessing Pipeline

The Python function tokenizes the sample sentence into words and punctuation, removes punctuation and common English stopwords, and applies the Porter stemmer. It prints the original tokens, the filtered tokens, and the final stems.

- **Stemming vs. lemmatization** — Stemming applies rules to shorten words, while lemmatization uses grammatical information to find a dictionary form. Both can reduce `running` to `run` when it is used as a verb.
- **When stopword removal helps** — It can reduce the number of features in a bag-of-words model. It can also remove meaningful words. For example, removing `not` from `not good` changes the sentiment.

#### Q2. Scaled Dot-Product Attention

The NumPy function computes `Q @ K.T`, divides the scores by the square root of the key dimension, applies softmax to each row, and multiplies the attention weights by `V`. Scaling helps keep large dot products from making the softmax weights overly concentrated.

```text
Attention weights
[[0.7311 0.2689]
 [0.2689 0.7311]]

Final output
[[2.0758 3.0758 4.0758 5.0758]
 [3.9242 4.9242 5.9242 6.9242]]
```

### Part III. Self-Attention Calculation

I used the lecture’s **unscaled dot-product attention** method to calculate the third token’s new representation. The raw scores are `[4, 12, 10]`. After applying softmax, I rounded the weights to one decimal place, giving `[0.0, 0.9, 0.1]`. The weighted sum of the value vectors is `[2.0, 7.8, 0.3]`.

## What I Learned

- I learned how tokenization, stopword removal, and stemming change the text an NLP model receives.
- I learned how queries and keys determine attention weights and how those weights combine value vectors.
- I practiced implementing row-wise softmax and scaled dot-product attention using NumPy.
- I learned how positional encodings preserve word order and how causal masking allows parallel training while preventing access to future tokens.

## Technologies and Libraries Used

| Technology or library | Purpose |
| --- | --- |
| Python | Implement the programming exercises |
| NLTK | Tokenize text, load English stopwords, and apply Porter stemming |
| NumPy | Compute matrix products, softmax weights, and weighted outputs |
| Google Colab or Jupyter | Run the notebook and view its outputs |
| GitHub | Host the source notebook and README |

## How to Run the Notebook

1. Open `your-notebook.ipynb` in Google Colab or Jupyter.
2. If the required libraries are missing, run this command in a notebook code cell:

   ```python
   %pip install nltk numpy
   ```

3. Run all notebook cells from top to bottom. The preprocessing code downloads the NLTK English stopword list.
4. Check the outputs for the three preprocessing lists, the attention weights matrix, and the final output matrix.

## Repository Contents

| File | Description |
| --- | --- |
| `README.md` | Assignment overview, answers, learning outcomes, and run instructions |
| `your-notebook.ipynb` | Commented Python source code for the programming exercises |

## Reference

Course lecture slides, *09-Transformers.pdf*.
