import os

# Configuration Constants
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def profile_data():
    print(f"Profiling datasets in {BASE_DIR} ...")
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(BASE_DIR, split)
        if os.path.exists(split_dir):
            classes = os.listdir(split_dir)
            total = 0
            print(f"\n--- {split.upper()} SET ---")
            for c in classes:
                count = len(os.listdir(os.path.join(split_dir, c)))
                total += count
                print(f"Class '{c}': {count} images")
            print(f"Total images in {split}: {total}")
        else:
            print(f"Warning: Directory not found: {split_dir}")

if __name__ == "__main__":
    profile_data()