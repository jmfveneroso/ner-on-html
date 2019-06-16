# Named Entity Recognition on HTML

This repository provides implementations for sequence taggers especially designed for Named Entity Recognition on HTML. We also provide a labeled dataset of faculty listings from universities across the world.

## Dataset

We collected 145 computer science faculty pages from 42 different countries in multiple languages, although the English version was preferred when it was available. We gathered faculty webpages randomly in proportion to the number of universities in each country. Each HTML page was preprocessed and converted to the CoNLL 2003 data format. That is, one word per line with empty lines representing sentence boundaries. Sentence boundaries were determined by line break HTML tags (div, p, table, li, br, etc.) in contrast to inline tags (span, em, a, td, etc.). Sentences that were more than fifty tokens long were also split according to the punctuation.

**The dataset is available in the [dataset](https://github.com/jmfveneroso/ner-on-html/tree/master/dataset) directory.**

## Models

This module implements the following NER methods:

* Multi-state Hidden Markov Model
* Linear Chain Conditional Random Fields
* BI-LSTM-CRF
* BI-LSTM-CRF with CNN character representations
* BI-LSTM-CRF with LSTM character representations

## Contact

If you have any question send an email to **jmfveneroso@gmail.com**.

### Running Models

```
docker build -t ner .
docker run -d -p 0.0.0.0:6006:6006 -p 8888:8888 -v $(pwd):/code --rm --user root ner
docker exec -it $(docker ps | awk '{if (NR == 2) print $1}') bash
docker stop $(docker ps | awk '{if (NR == 2) print $1}')
```
