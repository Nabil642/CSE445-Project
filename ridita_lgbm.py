import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow & VGG16 for Feature Extraction
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Scikit-Learn Models & LightGBM
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve

BASE_DIR = "data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Function to load dataset without shuffling so labels align perfectly for Scikit-Learn
def load_data(folder_name):
    dir_path = os.path.join(BASE_DIR, folder_name)
    return image_dataset_from_directory(
        dir_path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

print("Loading datasets from local directory ...")
train_ds = load_data('train')
val_ds = load_data('val')
test_ds = load_data('test')

# Extract exact integer labels to use for Scikit-Learn training
y_train = np.concatenate([y for x, y in train_ds], axis=0)
y_val = np.concatenate([y for x, y in val_ds], axis=0)
y_test = np.concatenate([y for x, y in test_ds], axis=0)

# Get class names for later
class_names = train_ds.class_names
print(f"Classes found: {class_names}")

# Load VGG16 without the final classification layer (include_top=False)
# pooling='avg' flattens the output into a neat 512-dimensional vector per image.
vgg_feature_extractor = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def extract_features(dataset):
    features = []
    for images, labels in dataset:
        # Preprocess images to match VGG16's specific formatting requirements
        preprocessed_images = preprocess_input(images)
        # Extract features
        extracted = vgg_feature_extractor.predict(preprocessed_images, verbose=0)
        features.append(extracted)
    return np.vstack(features)

print("Extracting features from Train set...")
X_train = extract_features(train_ds)

print("Extracting features from Test set...")
X_test = extract_features(test_ds)

print(f"Feature extraction complete! Training features shape: {X_train.shape}")

print("Training LightGBM...")
lgbm_model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
lgbm_model.fit(X_train, y_train)

# --- Predictions & Accuracy ---
y_pred = lgbm_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"LightGBM finished. Test Accuracy: {acc * 100:.2f}%\n")

# --- 1. Classification Report ---
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=class_names))

# --- 2. Confusion Matrix ---
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('LightGBM Confusion Matrix', fontsize=14, pad=10)
plt.ylabel('Actual Animal', fontsize=11)
plt.xlabel('Predicted Animal', fontsize=11)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("lgbm_confusion_matrix.png")
print("Saved lgbm_confusion_matrix.png")

# --- 3. Learning Curve ---
print("Calculating Learning Curve... (This might take a few seconds)")
np.random.seed(42)
shuffle_indices = np.random.permutation(len(X_train))
X_train_shuffled = X_train[shuffle_indices]
y_train_shuffled = y_train[shuffle_indices]

train_sizes, train_scores, test_scores = learning_curve(
    lgbm_model, X_train_shuffled, y_train_shuffled, 
    cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training Accuracy")
plt.plot(train_sizes, test_mean, 'o-', color="orange", label="Validation Accuracy")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="orange")
plt.title("Learning Curve: LightGBM", fontsize=14, fontweight='bold')
plt.xlabel("Number of Training Images", fontsize=12)
plt.ylabel("Accuracy Score", fontsize=12)
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("lgbm_learning_curve.png")
print("Saved lgbm_learning_curve.png")
