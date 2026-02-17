import numpy as np
from PIL import Image
import os

# --- Configuration ---
INPUT_DIR = "processed_data/train"  # We ONLY use training data (rigorous!)
OUTPUT_DIR = "centroids"
CLASSES = ["pizza_slice", "whole_pizza", "pizza_box"]
TARGET_SIZE = (32, 32)

def calculate_and_visualize_centroids():
    # Create output folder
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("--- Calculating Centroids (Mean Vectors) ---")

    for class_name in CLASSES:
        # 1. Load the Class Matrix X_k
        matrix_path = os.path.join(INPUT_DIR, f"Matrix_{class_name}.npy")
        
        if not os.path.exists(matrix_path):
            print(f"[Error] Matrix not found for {class_name}")
            continue
            
        # Load the matrix (Rows = Images, Cols = Pixels)
        matrix_X = np.load(matrix_path)
        
        # 2. Calculate the Mean (Centroid)
        # Formula: mu_k = (1/n) * sum(x_i)
        # axis=0 computes the mean down the columns (per pixel)
        centroid_vector = np.mean(matrix_X, axis=0)
        
        # 3. Save the Mathematical Vector
        # Save as .npy (for the classifier script)
        np.save(os.path.join(OUTPUT_DIR, f"Centroid_{class_name}.npy"), centroid_vector)
        
        # Save as .csv (for the Appendix/Examiner)
        # We use %.4f because the mean is a float (e.g., 125.45)
        np.savetxt(os.path.join(OUTPUT_DIR, f"Centroid_{class_name}.csv"), 
                   [centroid_vector], delimiter=",", fmt='%.4f')
        
        # 4. Inverse Map: Vector -> Image
        # We reshape the 3072 vector back into (32, 32, 3)
        # We must cast to 'uint8' (integers 0-255) to be a valid image
        centroid_tensor = centroid_vector.reshape(TARGET_SIZE[0], TARGET_SIZE[1], 3).astype(np.uint8)
        
        # Create and Save the Image
        img = Image.fromarray(centroid_tensor, 'RGB')
        # We resize it up to 256x256 just so you can see it clearly on your screen
        img_display = img.resize((256, 256), Image.Resampling.NEAREST)
        img_display.save(os.path.join(OUTPUT_DIR, f"Visual_{class_name}.png"))
        
        print(f"\nClass: {class_name}")
        print(f"  -> Calculated Mean Vector: {centroid_vector.shape}")
        print(f"  -> Saved Visual Representation: Visual_{class_name}.png")

if __name__ == "__main__":
    calculate_and_visualize_centroids()