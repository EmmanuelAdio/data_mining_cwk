# Data Mining Coursework (COC131)

This repository contains coursework for COC131, demonstrating machine learning techniques applied to image classification using scikit-learn.

## Overview

This project implements a complete machine learning pipeline for classifying forest images using a Multi-Layer Perceptron (MLP) neural network. The workflow includes data preprocessing, hyperparameter optimization, model evaluation, and dimensionality reduction.

## Repository Contents

- **`coc131_cw.py`**: Core implementation file containing the `COC131` class with methods for:
  - Dataset loading and preprocessing
  - Data standardization
  - MLP classifier training and hyperparameter tuning
  - Cross-validation methods
  - Locally Linear Embedding (LLE) for dimensionality reduction

- **`F229639_answer_notebook.ipynb`**: Jupyter notebook demonstrating complete solutions to all coursework questions with visualizations and analysis

- **`README.md`**: This file

## What the Notebook Shows

The notebook (`F229639_answer_notebook.ipynb`) provides a comprehensive walkthrough of the following machine learning tasks:

### Q1: Dataset Loading
- Loads forest image dataset (32×32×3 RGB images)
- Converts images to flattened feature vectors (3072 features)
- Visualizes sample images from different classes

### Q2: Data Standardization
- Applies StandardScaler to normalize features
- Compares data distributions before and after standardization
- Explains the impact on model training efficiency

### Q3: MLP Training & Hyperparameter Tuning
Systematically optimizes the following hyperparameters:
- **Solver**: Compares SGD vs Adam optimizers
- **Activation Function**: Tests identity, logistic, tanh, and ReLU
- **Hidden Layer Sizes**: Evaluates various network architectures (single and multi-layer)
- **Alpha** (L2 regularization): Tunes penalty strength to prevent overfitting
- **Learning Rate**: Finds optimal step size for gradient descent
- **Batch Size**: Determines best mini-batch size for training

**Final Optimized Model**:
- Hidden layers: (300, 150)
- Alpha: 0.1
- Learning rate: 0.0001
- Activation: ReLU
- Solver: Adam
- Batch size: Auto
- **Peak test accuracy**: 67.11% at epoch 47
- **Final test accuracy**: 66.19% (with early stopping)

### Q4: Regularization Analysis
- Investigates the effect of alpha on model performance
- Analyzes weight and bias norms across different alpha values
- Computes macro-averaged precision, recall, and F1-scores
- Demonstrates the bias-variance trade-off

### Q5: Cross-Validation Comparison
- Compares Stratified K-Fold vs standard K-Fold cross-validation
- Uses paired t-test to assess statistical significance
- **Result**: No significant difference found (p-value = 0.6007)

### Q6: Dimensionality Reduction with LLE
- Reduces 3072-dimensional image data to 2D using Locally Linear Embedding
- Evaluates different `n_neighbors` values using Silhouette scores
- Visualizes class separability in 2D space
- **Best result**: n_neighbors = 200 with silhouette score = -0.102

## Key Technologies

- **Python 3.x**
- **scikit-learn**: MLPClassifier, StandardScaler, LLE
- **NumPy**: Array operations and numerical computing
- **Matplotlib**: Data visualization
- **SciPy**: Statistical analysis
- **PIL**: Image processing

## Results Summary

The project successfully demonstrates:
- Effective hyperparameter optimization improving accuracy from ~58% to ~67%
- The importance of data standardization for neural network training
- Proper evaluation using cross-validation and statistical testing
- Visualization of high-dimensional image data in 2D space

---

*This coursework showcases fundamental machine learning skills including data preprocessing, model selection, hyperparameter tuning, and proper evaluation methodology.*
