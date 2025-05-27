# Siamese Network for General Image Similarity

## Overview

Inspired by SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification (Dey et al., 2017) , this project develops a robust Siamese CNN framework to detect and localize changes in simulation outputs. While I cannot include confidential simulation datasets here, I have validated the model on proprietary flow-field simulation data showing high accuracy in identifying deviations. Additionally, a public signature dataset is provided in this repository for demonstration and testing of the same similarity-learning pipeline on offline signature verification tasks.

Key objectives:

Simulation Validation: Learn a similarity metric to flag unacceptable deviations between reference and new simulation images (flow fields, contour plots, etc.).

Generalization: Apply the same network architecture and training pipeline to any image-comparison use case (e.g., signature verification).

Interpretability: Generate similarity scores and visual maps highlighting regions responsible for mismatches.

### 1. Clone your fork

```bash
git clone git@github.com:farzeenka/Siamese-network.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```
