# Siamese Network for Flow-Field Validation

## Overview


This repository implements a convolutional Siamese neural network tailored for writer-independent offline signature verification. The core objective is to train a model that learns a similarity metric between pairs of signature images, enabling the system to:

Verify Authenticity: Accurately distinguish between genuine signatures and skilled forgeries.

Generalize Across Writers: Operate without retraining for each new signer, making it writer-independent.

Interpret Decisions: Produce similarity scores and localization maps that highlight regions of the signature contributing to match/mismatch.

Key Features:

Pretrained Backbone: Uses a ResNet-50 encoder (ImageNet weights) truncated to produce 256-dimensional embeddings.

Contrastive Loss: Implements a custom DistanceLayer for pairwise contrastive loss (α=5, margin=0.5).

Flexible Data Handling: Data generators load and preprocess images from dataset/real and dataset/forgeries, creating TensorFlow tf.data.Dataset pipelines for efficient training.

Evaluation & Visualization: After training, the model computes cosine similarity distributions on test splits and plots them via utils.plotSimilarities, providing intuitive diagnostics of model performance.

This implementation is motivated by SigNet (Dey et al., 2017).



### 1. Clone your fork

```bash
git clone git@github.com:farzeenka/Siamese-network.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```
