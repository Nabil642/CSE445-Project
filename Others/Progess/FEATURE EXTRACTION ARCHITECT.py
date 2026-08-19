import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from m2_data_pipeline import load_data 

def extract_and_save_features():
    print("Initializing VGG16 Feature Extractor...")
    # Load VGG16 without classification head and apply Global Average Pooling
    vgg_extractor = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    def extract_features(dataset):
        features = []
        for images, labels in dataset:
            preprocessed_images = preprocess_input(images)
            extracted = vgg_extractor.predict(preprocessed_images, verbose=0)
            features.append(extracted)
        return np.vstack(features)

    train_ds = load_data('train')
    test_ds = load_data('test')

    print("Extracting features from Train set...")
    X_train = extract_features(train_ds)
    
    print("Extracting features from Test set...")
    X_test = extract_features(test_ds)

    print(f"Feature extraction complete! Training features shape: {X_train.shape}")
    
    # Save the feature matrices for the modeling team
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    print("Features successfully saved as .npy files.")

if __name__ == "__main__":
    extract_and_save_features()