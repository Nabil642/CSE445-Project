import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Configuration Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Class mapping
CLASSES = sorted(['cat', 'cow', 'dog', 'lamb', 'zebra'])
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

def load_images_and_labels(split='train'):
    """
    Load all images and labels from a specific split (train, val, test).
    
    Args:
        split: 'train', 'val', or 'test'
    
    Returns:
        images (np.array): Stacked image arrays [N, 224, 224, 3]
        labels (np.array): Class indices [N]
    """
    split_dir = os.path.join(DATA_DIR, split)
    
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Directory not found: {split_dir}")
    
    images = []
    labels = []
    
    print(f"Loading {split} images...")
    for class_name in CLASSES:
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        
        class_idx = CLASS_TO_IDX[class_name]
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(IMG_SIZE, Image.LANCZOS)
                img_array = np.array(img, dtype=np.float32)
                
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    print(f"Loaded {len(images)} images from {split} split")
    return np.array(images), np.array(labels)

def create_batches(images, labels, batch_size=BATCH_SIZE, shuffle=False):
    """
    Create batches from images and labels.
    
    Args:
        images: Image array [N, 224, 224, 3]
        labels: Label array [N]
        batch_size: Batch size
        shuffle: Whether to shuffle the data
    
    Yields:
        Batches of (images, labels)
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    
    dataset = dataset.batch(batch_size)
    
    for batch_images, batch_labels in dataset:
        yield batch_images.numpy(), batch_labels.numpy()

def load_data(split='train', shuffle=False):
    """
    Load data for a specific split and return as batches.
    
    Args:
        split: 'train', 'val', or 'test'
        shuffle: Whether to shuffle the data (typically True for train)
    
    Returns:
        Generator yielding (images, labels) batches
    """
    images, labels = load_images_and_labels(split)
    return create_batches(images, labels, BATCH_SIZE, shuffle=shuffle)

def prepare_and_save_labels():
    """
    Load labels from train and test sets and save them as .npy files
    for use by modeling scripts.
    """
    print("Preparing and saving labels...")
    
    # Load labels
    _, y_train = load_images_and_labels('train')
    _, y_test = load_images_and_labels('test')
    
    # Save labels
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    
    print(f"Saved y_train.npy with shape {y_train.shape}")
    print(f"Saved y_test.npy with shape {y_test.shape}")
    
    # Print class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print("\nTraining set class distribution:")
    for class_idx, count in zip(unique, counts):
        print(f"  {CLASSES[class_idx]}: {count}")

if __name__ == "__main__":
    # Test data loading
    print("Testing data pipeline...")
    print(f"Base directory: {DATA_DIR}")
    print(f"Classes: {CLASSES}\n")
    
    # Test loading a batch
    train_loader = load_data('train')
    sample_batch = next(iter(train_loader))
    print(f"Sample batch shape: images={sample_batch[0].shape}, labels={sample_batch[1].shape}")
    print(f"Image value range: [{sample_batch[0].min():.1f}, {sample_batch[0].max():.1f}]")
    
    # Prepare labels for modeling
    prepare_and_save_labels()
    print("\n✅ Data pipeline ready!")
