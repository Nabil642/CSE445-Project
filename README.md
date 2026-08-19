# Custom Image Classification Dataset and Model Development for Five Animal Classes

**CSE445 — Machine Learning | Section 7 | Group 01**

A computationally efficient hybrid pipeline for multi-class image classification of five animal species (Cat, Cow, Dog, Lamb, Zebra), combining **VGG16 deep-feature extraction** (Transfer Learning) with **classical Machine Learning classifiers**. The pipeline achieves up to **100% test accuracy** while empirically proving the absence of overfitting via Learning Curve analysis.

## Team Members

| Name | Student ID | Contribution |
|---|---|---|
| Md. Nazibul Islam Nabil | 2222456042 | K-Nearest Neighbors (KNN) |
| Shawlin Salit | 2212906042 | Hybrid Soft-Voting Ensemble Classifier |
| Ridita Afrin Riya | 2211622042 | Gaussian Naive Bayes |
| Mubtasim Fuad | 2131135642 | Random Forest Ensemble |
| Faiyazul Islam Rimon | 2231898042 | Logistic Regression, Data Loading & VGG16 Feature Extraction |

## Overview

Training deep CNNs from scratch requires massive datasets and heavy compute. This project instead uses **Transfer Learning**: a pre-trained VGG16 network (trained on ImageNet) is stripped of its fully connected layers and used purely as a feature extractor, mapping raw RGB images into a dense 512-dimensional latent space via Global Average Pooling. These feature vectors are then fed into several classical ML classifiers, which are trained, tuned, and evaluated independently by each team member.

## Dataset

- **Total images:** 500, perfectly balanced across 5 classes (100 images/class): Cat, Cow, Dog, Lamb, Zebra
- **Split:**
  - Training: 350 images (70%)
  - Validation: 75 images (15%)
  - Testing: 75 images (15%)
- Images loaded via `image_dataset_from_directory` in micro-batches of 32, with `shuffle=False` to keep extracted features perfectly aligned with their ground-truth labels
- All images resized to `224 × 224 × 3` to match VGG16's input requirements

## Methodology

### 1. Deep Feature Extraction (VGG16)
- VGG16 loaded with `weights='imagenet'`, `include_top=False` — truncating the dense classification head to expose the convolutional base
- Inputs preprocessed via VGG16's `preprocess_input` (RGB → BGR conversion + ImageNet channel-wise mean centering)
- `pooling='avg'` applies Global Average Pooling to collapse the `(7, 7, 512)` convolutional output into a single dense **512-dimensional feature vector** per image

### 2. Classical ML Classifiers
Each team member trained and evaluated a separate classifier on the extracted 512-D features:

- **Logistic Regression** — Multinomial softmax classifier, hyper-tuned with `max_iter=2000` for convergence in high-dimensional space
- **Random Forest** — 100+ estimator bagging ensemble, hyperparameters tuned via `GridSearchCV` (5-fold CV) over tree count, depth, and split criteria
- **K-Nearest Neighbors (KNN)** — Non-parametric distance-based classifier, hyperparameters (`n_neighbors`, `weights`, `metric`) tuned via `GridSearchCV`
- **Gaussian Naive Bayes** — Probabilistic classifier using the Gaussian variant to handle continuous VGG16 feature vectors, requiring minimal tuning
- **Hybrid Soft-Voting Ensemble** — Stacks Logistic Regression, Random Forest, SVM, and XGBoost as base learners; uses `predict_proba()` to average class probabilities (Soft Voting) rather than simple majority voting

### 3. Evaluation
Every model was evaluated on the held-out test set (n = 75) using:
- **Accuracy** and full **Classification Report** (Precision, Recall, F1-score per class)
- **Confusion Matrix** heatmaps for spatial error analysis
- **Learning Curves** (10%–100% of training data, 5-fold CV) to verify generalization and rule out overfitting

## Results

| Classifier | Test Accuracy |
|---|---|
| Gaussian Naive Bayes | See notebook |
| K-Nearest Neighbors (KNN) | 93.33% |
| Random Forest | 97.33% |
| Logistic Regression | 97.33% |
| **Hybrid Soft-Voting Ensemble** | **100.00%** |

Precision and Recall across all classifiers consistently ranged between **0.93 and 1.00**, indicating no class-imbalance bias. Learning Curve analysis showed training and validation accuracy converging as training set size increased, confirming the models generalize well and do not overfit despite the constrained dataset size.

## Repository Structure

```
├── CSE445Report.pdf                          # Full project report / paper
├── CSE445_Presentation.pptx                  # Final presentation slides
├── Final_FAIYAZUL_LogisticRegression.ipynb   # Data loading, VGG16 extraction + Logistic Regression
├── MUBTASIM_RandomForest.ipynb               # Random Forest + GridSearchCV tuning
├── NAZIBUL_KNN.ipynb                         # K-Nearest Neighbors + GridSearchCV tuning
├── Final_RIDITA_NaiveBayes.ipynb             # Gaussian Naive Bayes
├── SHAWLIN_Hybrid.ipynb                      # Hybrid Soft-Voting Ensemble
└── README.md
```

> **Note:** Each notebook includes the shared "General Setup" section (data loading + VGG16 feature extraction) that must be run first, followed by that member's individual classifier implementation and evaluation.

## Tech Stack

- **Deep Learning:** TensorFlow / Keras (VGG16, `image_dataset_from_directory`)
- **Classical ML:** scikit-learn (Logistic Regression, Random Forest, KNN, Gaussian Naive Bayes, Voting Classifier), XGBoost
- **Data & Visualization:** NumPy, Pandas, Matplotlib, Seaborn
- **Environment:** Google Colab (GPU runtime), Google Drive for dataset storage

## How to Run

1. Open any notebook in Google Colab
2. Mount Google Drive and ensure the dataset is available at `/content/drive/MyDrive/CSE445-Project/data` with `train/`, `val/`, and `test/` subfolders (each containing class-named folders: `cat/`, `cow/`, `dog/`, `lamb/`, `zebra/`)
3. Run the **General Setup** cells first (data loading + VGG16 feature extraction)
4. Run the classifier-specific cells below to train and evaluate that member's model

## Key Findings

- Transfer Learning with a frozen, pre-trained VGG16 backbone yields highly discriminative 512-D features without any fine-tuning
- Classical ML classifiers built on top of these features achieve near-perfect accuracy on a small (500-image), highly constrained dataset
- Learning Curve convergence across all models provides strong evidence against overfitting
- The Hybrid Soft-Voting Ensemble, combining multiple base learners' probability estimates, achieved the strongest and most robust performance

## References

1. A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional neural networks," *NeurIPS*, vol. 25, pp. 1097–1105, 2012.
2. K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," *arXiv:1409.1556*, 2014.
3. J. Donahue et al., "DeCAF: A deep convolutional activation feature for generic visual recognition," *ICML*, pp. 647–655, 2014.
4. C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.
5. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
6. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.
7. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *KDD*, pp. 785–794, 2016.
