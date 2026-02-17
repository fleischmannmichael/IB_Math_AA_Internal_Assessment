import numpy as np
from PIL import Image
import os
import csv

# --- Configuration ---
# Mathematical Space: R^3072 (32x32 pixels x 3 channels)
TARGET_SIZE = (32, 32)
CLASSES = ["pizza_slice", "whole_pizza", "pizza_box"]

# Directions
TRAIN_INPUT = "dataset_train"
TEST_INPUT  = "dataset_test"

# Output Locations
TRAIN_OUTPUT = "processed_data/train"
TEST_OUTPUT  = "processed_data/test"

def process_dataset(input_dir, output_dir, create_matrix=False):
    """
    Process a dataset folder.
    Args:
        create_matrix (bool): If True, stacks vectors into a Matrix X (for Training).
                              If False, keeps vectors separate (for Testing).
    """
    # Create main output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n--- Processing Dataset: {input_dir} ---")
    print(f"    Mode: {'Creating Matrix X' if create_matrix else 'Individual Vectors Only'}")

    for class_name in CLASSES:
        # Define specific paths
        class_input_path = os.path.join(input_dir, class_name)
        class_output_path = os.path.join(output_dir, class_name)
        
        # Create subfolder for this class
        if not os.path.exists(class_output_path):
            os.makedirs(class_output_path)
        
        # Check input exists
        if not os.path.exists(class_input_path):
            print(f"[Warning] Missing folder: {class_input_path}")
            continue

        class_vectors = []
        
        # Get all images
        image_files = [f for f in os.listdir(class_input_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"  Class '{class_name}': Found {len(image_files)} images.")

        for idx, filename in enumerate(image_files):
            try:
                # 1. Load Image
                img_path = os.path.join(class_input_path, filename)
                img = Image.open(img_path).convert('RGB')
                
                # 2. Resize to 32x32 (Space R^3072)
                # Using LANCZOS for high-quality mathematical resampling
                img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                
                # 3. Flatten Map phi: Tensor -> Vector
                # Order='C' ensures row-by-row reading (R,G,B)
                tensor = np.array(img)
                vector = tensor.flatten(order='C')
                
                # 4. Save Individual Vector v_i
                # Prefix file with 'train' or 'test' to avoid confusion later
                prefix = "train" if create_matrix else "test"
                
                # Format A: .npy (Computer Readable)
                npy_path = os.path.join(class_output_path, f"{prefix}_{class_name}_{idx}.npy")
                np.save(npy_path, vector)
                
                # Format B: .csv (Examiner/Human Readable)
                csv_path = os.path.join(class_output_path, f"{prefix}_{class_name}_{idx}.csv")
                np.savetxt(csv_path, [vector], delimiter=",", fmt='%d')
                
                # Store for Matrix (Only if needed)
                if create_matrix:
                    class_vectors.append(vector)
                    
            except Exception as e:
                print(f"    [Error] Could not process {filename}: {e}")

        # 5. Create & Save Class Matrix X_k (ONLY for Training Data)
        if create_matrix and len(class_vectors) > 0:
            # Stack vectors as rows: Matrix X_k
            matrix_X_k = np.stack(class_vectors, axis=0) 
            
            # Save Matrix as .npy
            matrix_npy_path = os.path.join(output_dir, f"Matrix_{class_name}.npy")
            np.save(matrix_npy_path, matrix_X_k)
            
            # Save Matrix as .csv (Excel readable)
            matrix_csv_path = os.path.join(output_dir, f"Matrix_{class_name}.csv")
            np.savetxt(matrix_csv_path, matrix_X_k, delimiter=",", fmt='%d')
            
            print(f"    -> Created Matrix X_{class_name}: {matrix_X_k.shape}")

if __name__ == "__main__":
    # 1. Process Training Data (Creates Vectors + Matrices)
    process_dataset(TRAIN_INPUT, TRAIN_OUTPUT, create_matrix=True)
    
    # 2. Process Testing Data (Creates Vectors ONLY)
    process_dataset(TEST_INPUT, TEST_OUTPUT, create_matrix=False)