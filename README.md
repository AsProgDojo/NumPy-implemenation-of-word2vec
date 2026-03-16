# Word2Vec from Scratch in Pure NumPy

A from-scratch implementation of **Word2Vec Skip-Gram with Negative Sampling (SGNS)** using only **NumPy** for the core optimization procedure.  
The goal of this project is to implement the main training mechanics of word2vec without using machine learning frameworks such as PyTorch or TensorFlow.

This project includes:
- text preprocessing
- vocabulary construction
- skip-gram training pair generation
- negative sampling
- forward pass
- loss computation
- gradient computation
- parameter updates with SGD
- simple qualitative evaluation using nearest neighbors

---

## Project Goal

The purpose of this project is to implement the **core training loop of word2vec in pure NumPy** and fully understand each step of the training process.

I chose the **Skip-Gram with Negative Sampling (SGNS)** variant because it is a standard and efficient version of word2vec that is easier to implement and explain than the full softmax formulation.

---

## Model Overview

Word2Vec learns dense vector representations of words such that words appearing in similar contexts obtain similar embeddings.

In the **skip-gram** setting:
- the **input** is a center word
- the model tries to predict its surrounding **context words**

In **negative sampling**:
- for each real `(center, context)` pair, the model also samples several **negative words**
- the model learns to assign:
  - a **high score** to the real context word
  - **low scores** to the sampled negative words

This makes training much more efficient than computing a full softmax over the entire vocabulary.

---

## Mathematical Objective

For a center word `c`, a true context word `o`, and negative samples `n1, n2, ..., nk`, the SGNS objective for one training example is:

$L = -\log \sigma(u_o^T v_c) - \sum_{j=1}^{k} \log \sigma(-u_{n_j}^T v_c)$

where:
- `v_c` is the input embedding of the center word
- `u_o` is the output embedding of the true context word
- `u_{n_j}` are the output embeddings of the negative samples
- `σ` is the sigmoid function

The training process updates the embeddings so that:
- real word-context pairs become more similar
- sampled negative pairs become less similar

---

## Dataset

The training corpus used in this project is:

**_The Bridge on the Drina_ by Ivo Andrić**

I chose this text because it is a meaningful literary corpus with recurring names, places, and themes, while still being manageable for a NumPy-based implementation.

---

## Project Structure

```text
.
├── preprocess.py
├── dataset.py
├── model.py
├── train.py
└── The bridge on the Drina.txt

```
---

## File descriptions

### `preprocess.py`
Responsible for:
- reading the raw text file
- lowercasing the text
- splitting it into sentences
- tokenizing words
- counting word frequencies
- assigning integer IDs to words
- converting tokenized sentences into lists of integer IDs

### `dataset.py`
Responsible for:
- generating positive skip-gram (center, context) pairs
- building the negative sampling distribution
- sampling negative words for each positive pair

### `model.py`
Contains the SGNS model implementation:
- input embedding matrix
- output embedding matrix
- numerically stable sigmoid
- forward pass
- loss computation
- gradient computation
- SGD parameter updates

### `train.py`
Runs the full training pipeline:
- preprocesses the corpus
- builds the dataset
- trains the model for several epochs
- prints average training loss
- evaluates learned embeddings through nearest-neighbor queries

---

## Training Procedure

For each training example, the model performs the following steps:

1. retrieve the center word embedding
2. retrieve the true context word embedding
3. retrieve the negative sample embeddings
4. compute dot-product scores
5. apply the sigmoid function
6. compute the SGNS loss
7. compute gradients
8. update parameters using stochastic gradient descent

Training is repeated for multiple epochs, and the average loss is printed after each epoch.

---

## Example Training Output
A typical training run prints:
- vocabulary size
- number of training pairs
- average loss per epoch
- nearest neighbors for selected words after training

Example behavior observed during training:
- the loss decreases steadily across epochs
- recurring words from the book tend to form meaningful local neighborhoods
- some noise remains due to OCR/text quality

---

## How to Run

Make sure NumPy is installed, then run:

```bash
python train.py
```

The script will:
- preprocess the corpus
- generate SGNS training examples
- train the model
- print the average loss for each epoch
- show nearest neighbors for a few query words







