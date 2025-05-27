# Siamese Network for General Image Similarity

## Overview

This repository develops a robust Siamese neural network model for detecting and localizing changes in image data—motivated by two key domains:

1. Simulation Flow-Field Validation: Inspired by industrial simulations where small code modifications can alter flow patterns, we use the Siamese framework to identify deviations from reference flow-field images.

2. Signature Verification: Leveraging ideas from SigNet (Dey et al., 2017), we include an offline signature dataset to demonstrate writer-independent verification.

Because of confidentiality constraints, the actual simulation datasets cannot be included. Instead, a small signature dataset is bundled under dataset/ for general testing.

Key Features:

Generalized Architecture: Convolutional Siamese network with a ResNet-50 backbone producing 256-dimensional embeddings.

Contrastive Loss: Custom DistanceLayer implements contrastive loss (α=5, margin=0.5) to learn meaningful similarity metrics.

Multiple Domains: Although real simulation data is confidential, the model has been validated on synthetic flow-field examples and a public signature dataset.

Visualization & Interpretability: After evaluation, similarity distributions and error-localization heatmaps are generated via utils.plotSimilarities.

### 1. Clone your fork

```bash
git clone git@github.com:farzeenka/Siamese-network.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```
