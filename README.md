# Siamese Network for Flow-Field Validation

## Overview

**Motivated by SigNet:** This work is motivated by *SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification* by Dey et al. (2017).

## Overview

This repository implements Siamese and Triplet networks to detect and localize deviations in simulation flow-field outputs. Given a set of reference (correct) simulation images, the models learn to measure similarity and highlight regions of error, providing interpretability alongside validation.



### 1. Clone your fork

```bash
git clone git@github.com:farzeenka/Siamese-network.git
cd Siamese-network
```

### 2. Install dependencies

```bash
# On WSL2 (Windows) or Linux
pip install -r requirements.txt
```
# Siamese-network
