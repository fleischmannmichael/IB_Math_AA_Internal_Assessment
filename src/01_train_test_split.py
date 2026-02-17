import os
import shutil
import random

# --- Configuration ---
SOURCE_DIR = "raw_data"
TRAIN_DIR = "dataset_train"  # Images for Centroids (25)
TEST_DIR = "dataset_test"    # Images for Testing (5)
SPLIT_COUNT = 5              # Number of images to move to Test

CLASSES = ["pizza_slice", "whole_pizza", "pizza_box"]

def split_dataset():
    # 1. Create Train/Test directories
    for folder in [TRAIN_DIR, TEST_DIR]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    for class_name in CLASSES:
        print(f"\n--- Splitting Class: {class_name} ---")
        
        # Source path
        src_path = os.path.join(SOURCE_DIR, class_name)
        
        # Destination paths
        train_path = os.path.join(TRAIN_DIR, class_name)
        test_path = os.path.join(TEST_DIR, class_name)
        
        # Create subfolders
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        
        # Get all images
        if not os.path.exists(src_path):
            print(f"[Error] Missing source folder: {src_path}")
            continue
            
        files = [f for f in os.listdir(src_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Random Shuffle (Ensures fairness)
        random.shuffle(files)
        
        # 2. Perform the Split
        # First 5 go to Test
        test_files = files[:SPLIT_COUNT]
        # Remaining go to Train
        train_files = files[SPLIT_COUNT:]
        
        print(f"  -> Total Images: {len(files)}")
        print(f"  -> Training Set: {len(train_files)} (Calculating Centroids)")
        print(f"  -> Testing Set : {len(test_files)} (Testing Distances)")
        
        # Copy files to destinations
        for f in test_files:
            shutil.copy2(os.path.join(src_path, f), os.path.join(test_path, f))
            
        for f in train_files:
            shutil.copy2(os.path.join(src_path, f), os.path.join(train_path, f))

    print("\n[Done] Dataset split complete.")
    print(f"Use '{TRAIN_DIR}' for Script 1 & 2 (Building Matrix/Centroids)")
    print(f"Use '{TEST_DIR}' for Script 4 (Testing Accuracy)")

if __name__ == "__main__":
    split_dataset()