# Solving the ARC task using Deep Learning

[![Python 3+7 ready](https://img.shields.io/badge/python-3.8%2B-yellowgreen.svg)](https://www.python.org/)

## Authors
[Luca H. Thoms](https://github.com/foger3), [Hannes Rosenbusch](https://hannesrosenbusch.github.io/), [Karel A. Veldkamp](https://github.com/KarelVeldkamp) and [Claire E. Stevenson](https://modelingcreativity.org/people/) 

## The Idea
Analogical reasoning derives information from known relations and generalizes this information to similar yet unfamiliar
situations. One of the first generalized ways in which deep learning models were able to solve verbal analogies was through
vector arithmetic of word embeddings, essentially relating words that were mapped to a vector space (e.g., king – man + woman
=__?). In comparison, most attempts to solve visual analogies are still predominantly task-specific and less generalizable. This
project focuses on visual analogical reasoning and applies the initial generalized mechanism used to solve verbal analogies to
the visual realm. Taking the Abstraction and Reasoning Corpus (ARC) as an example to investigate visual analogy solving,
we use a variational autoencoder (VAE) to transform ARC items into low-dimensional latent vectors, analogous to the word
embeddings used in the verbal approaches. Through simple vector arithmetic, underlying rules of ARC items are discovered
and used to solve them. Results indicate that the approach works well on simple items with fewer dimensions (i.e., few colors
used, uniform shapes), similar input-to-output examples, and high reconstruction accuracy on the VAE. Predictions on more
complex items showed stronger deviations from expected outputs, although, predictions still often approximated parts of the
item’s rule set. Error patterns indicated that the model works as intended. On the official ARC paradigm, the model achieved
a score of 2% (cf. current world record is 21 %) and on ConceptARC it scored 8.8%. Although the methodology proposed
involves basic dimensionality reduction techniques and standard vector arithmetic, this approach demonstrates promising
outcomes on ARC and can easily be generalized to other abstract visual reasoning tasks.

## Keywords
Visual Analogy, ARC, Neural Embeddings, Vector Arithmetic

## Contact
For further information on this project, please consult the [first author](https://github.com/foger3) (owner of this repository)