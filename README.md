# When do Graph Neural Networks Outperform Spectral Methods under Structural Noise?

Question we are trying to answer: Can spectral methods outperform GNNs on graph community detection as a graph's structural noise increases? At what level of structural noise does that occur?

We want FAIR comparisons!!!

## Project structure
This repository contains the following:
- ```data/```: Synthetic data generation pipeline. LFR and SBM graphs with targeted (based on centrality) and random edge deletions functioning as our lever for structural noise.
- ```methods/```: Method definitions: All spectral and GNN-based community detection methods
- ```pipeline/```: ML pipeline for training or fitting methods on a given training dataset and evaluating on a held out test set (held-out nodes)

## Synthetic Datasets (Jamie)
TODO: Confirm with Jamie and Finn but I think it should be five base graphs, and for each base graph, have 9 different train/val splits. Each split will remove an increasing number of nodes (increasing structural noise).

## Community Detection Methods (Sabrina)
This project compares 9 different TRANSDUCTIVE methods: 6 spectral and 3 GNN-based methods. These are:
1. Whole-eigenspectrum logistic regression
2. Whole-eigenspectrum label propagation
3. k-cut eigenspectrum logistic regression
4. k-cut eigenspectrum label propagation
5. Regularized eigenspectrum logistic regression
6. Regularized eigenspectrum label propagation
7. Simple Graph Convolution (SGC) - middle-ground between spectral and GNN-based
8. Graph Convolutional Network (GCN)
9. Graph Attention Network (GAT)

## Training and Evaluation Protocol (Finn)
- We compute Adjusted Random Index (ARI) on test nodes/labels only: -1 is perfect disagreement, 0 is chance agreement, 1 is perfect agreement.
- We also compute relative degradation: ARI at noise level x / ARI at noise level 0.
- Hyperparameter tuning via optuna.


## What to compare and how to ensure fair comparisons?
- ARI and relative ARI
- Parameter count (GNNs might have higher expressivity due to param count as opposed to noise robustness)
- FLOP count / computational time.
- GNNs receive either no node features (for now - To discuss)
- Hyperparameter tuning on fixed number of Optuna trials (10).

