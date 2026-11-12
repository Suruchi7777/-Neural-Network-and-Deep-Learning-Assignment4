# Neural Networks and Deep Learning Assignment 4

## Student Information

- **Name** — Suruchi Maharjan
- **Student ID** — [700797006]
- **Course** — Neural Networks and Deep Learning
- **Professor** - [Tsai, I Hua]
- **University** — University of Central Missouri


## Overview

This notebook contains two Python exercises. The first shows how an NLP preprocessing pipeline turns a sentence into a list of word stems. The second implements scaled dot-product attention and shows how queries and keys determine the contribution of each value vector to the output.

### Part II. Programming Questions

#### Q1. NLP Preprocessing Pipeline
The input sentence is “NLP techniques are used in virtual assistants like Alexa and Siri.”

The pipeline starts by splitting the sentence into words and punctuation using NLTK’s `wordpunct_tokenize`. The period appears as a separate token in the first output. The filtering step then removes punctuation and common English stopwords. Each word is checked against the stopword list in lowercase, while the filtered list preserves the original capitalization.

The final step applies NLTK’s Porter stemmer to the filtered words. Stemming reduces words to a shorter form, such as `assistants` to `assist`. Some stems, such as `techniqu`, are useful for grouping related word forms even though they are not dictionary words.

### Output

```text
Original Tokens: ['NLP', 'techniques', 'are', 'used', 'in', 'virtual', 'assistants', 'like', 'Alexa', 'and', 'Siri', '.']
Tokens Without Stopwords: ['NLP', 'techniques', 'used', 'virtual', 'assistants', 'like', 'Alexa', 'Siri']
Stemmed Words: ['nlp', 'techniqu', 'use', 'virtual', 'assist', 'like', 'alexa', 'siri']
```

#### Q2. Scaled Dot-Product Attention
The attention function takes three matrices: queries (`Q`), keys (`K`), and values (`V`). The product `Q @ K.T` compares each query with each key. Each key in this example has four elements, so the scores are divided by `sqrt(4) = 2`. The raw score matrix `[[2, 0], [0, 2]]` becomes `[[1, 0], [0, 1]]` after scaling.

Softmax converts each row of scores into attention weights. The calculation subtracts the maximum score in each row before exponentiating to improve numerical stability. Multiplying the attention weights by `V` produces one weighted combination of the value vectors for each query. The matrices below are rounded to four decimal places for display.

### Output

```text
Attention weights
[[0.7311 0.2689]
 [0.2689 0.7311]]

Final output
[[2.0758 3.0758 4.0758 5.0758]
 [3.9242 4.9242 5.9242 6.9242]]
```

## Results

- **Question 1** — Tokenization produced 12 tokens, including the period. Filtering removed `are`, `in`, `and`, and the period, leaving 8 words. Stemming transformed those 8 words without changing the number of tokens.
- **Question 2** — Each row of the attention weights sums to 1. The first query assigns about 73% of its weight to `[1, 2, 3, 4]`, so its output is closer to that vector. The second query assigns about 73% of its weight to `[5, 6, 7, 8]`, so its output is closer to the second vector.

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
| `Home Assignment 4_SuruchiMaharjan_700797006.ipynb` | Commented Python source code for the programming exercises |

## Reference

Course lecture slides, *09-Transformers.pdf*.
