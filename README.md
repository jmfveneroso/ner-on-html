
# Named Entity Recognition on the Web


This repository provides implementations for NER models and a novel labeled dataset built by crawling faculty directories from university websites across the world.

## Dataset

We collected 145 computer science faculty pages from 42 different countries in multiple languages, although the English version was preferred when it was available. We gathered faculty webpages randomly in proportion to the number of universities in each country. Each HTML page was preprocessed and converted to the CoNLL 2003 data format. That is, one word per line with empty lines representing sentence boundaries.

**The dataset is available in the [dataset](https://github.com/jmfveneroso/ner-on-html/tree/master/data) directory.**

The dataset is split into 3 different files: train, valid, and test. Also, we provide 11 features alongside each token.

| Feature                          | Type        |
|----------------------------------|-------------|
| Unaccented lowercase token       | Categorical |
| Exact dictionary match           | Binary      |
| Partial dictionary match         | Binary      |
| Email                            | Binary      |
| Number                           | Binary      |
| Honorific (Mr., Mrs., Dr., etc.) | Binary      |
| Matches a URL                    | Binary      |
| Is capitalized                   | Binary      |
| Is a punctuation sign            | Binary      |
| HTML tag + parent                | Categorical |
| CSS class                        | Categorical |

## Running the models

This repository contains implementations for 5 NER models. [This Jupyter Notebook](https://github.com/jmfveneroso/ner-on-html/tree/master/Main.ipynb)  demonstrates how to run the models.

* Multi-state Hidden Markov Model
* Linear Chain Conditional Random Fields
* BI-LSTM-CRF
* BI-LSTM-CRF with CNN character representations
* BI-LSTM-CRF with LSTM character representations

To run them in a Docker container, execute:

```
docker build -t ner .
docker run -d -p 8888:8888 -v $(pwd):/code --rm --user root ner
```

And access the notebook in http://localhost:8888.

## Contact

If you have got any questions, feel free to contact me at **jmfveneroso@gmail.com**.