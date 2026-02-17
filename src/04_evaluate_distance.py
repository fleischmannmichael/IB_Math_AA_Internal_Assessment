import numpy as np
import os
import csv
import pandas as pd # Optional, but helps with matrices. We'll stick to standard lib for safety.

# --- Configuration ---
TEST_DIR = "processed_data/test"
CENTROID_DIR = "centroids"
OUTPUT_DIR = "evaluation_results" # New dedicated folder
CLASSES = ["pizza_slice", "whole_pizza", "pizza_box"]

# --- Mathematical Distance Functions ---
def d_euclidean(v1, v2):
    # Formula: sqrt(sum((x - mu)^2)) [cite: 1190]
    return np.sqrt(np.sum((v1 - v2) ** 2))

def d_manhattan(v1, v2):
    # Formula: sum(|x - mu|) [cite: 1200]
    return np.sum(np.abs(v1 - v2))

def d_cosine(v1, v2):
    # Formula: 1 - (dot_product / (|x| * |mu|)) [cite: 1211]
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0: return 1.0 
    
    similarity = dot_product / (norm_v1 * norm_v2)
    return 1.0 - similarity 

def run_evaluation():
    # 1. Setup Output Directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[System] Created output folder: {OUTPUT_DIR}")

    # 2. Load Centroids (The "Knowledge Base")
    centroids = {}
    for c in CLASSES:
        path = os.path.join(CENTROID_DIR, f"Centroid_{c}.npy")
        if os.path.exists(path):
            centroids[c] = np.load(path)
        else:
            print(f"[Error] Missing centroid for {c}. Run Step 3 first.")
            return

    # 3. Initialize Data Structures
    results = []
    
    # Structure for Confusion Matrices: { Metric: { TrueClass: { PredClass: Count } } }
    confusion_matrices = {
        'Euclidean': {t: {p: 0 for p in CLASSES} for t in CLASSES},
        'Manhattan': {t: {p: 0 for p in CLASSES} for t in CLASSES},
        'Cosine':    {t: {p: 0 for p in CLASSES} for t in CLASSES}
    }

    correct_counts = {'Euclidean': 0, 'Manhattan': 0, 'Cosine': 0}
    total_images = 0

    print("--- Starting Evaluation ---")

    # 4. Iterate through Test Sets
    for true_class in CLASSES:
        class_dir = os.path.join(TEST_DIR, true_class)
        if not os.path.exists(class_dir): continue
        
        # Get individual test vector files (csv or npy - we use npy for speed)
        files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        
        for filename in files:
            total_images += 1
            x = np.load(os.path.join(class_dir, filename))
            
            # Distances for this image
            dists = {'Euclidean': {}, 'Manhattan': {}, 'Cosine': {}}
            
            # Compare x against EVERY centroid mu_k
            for k in CLASSES:
                mu_k = centroids[k]
                dists['Euclidean'][k] = d_euclidean(x, mu_k)
                dists['Manhattan'][k] = d_manhattan(x, mu_k)
                dists['Cosine'][k]    = d_cosine(x, mu_k)

            # Argmin Decision Rule [cite: 1222]
            preds = {}
            for metric in ['Euclidean', 'Manhattan', 'Cosine']:
                # Find class with minimum distance
                preds[metric] = min(dists[metric], key=dists[metric].get)
                
                # Update Accuracy Count
                if preds[metric] == true_class:
                    correct_counts[metric] += 1
                
                # Update Confusion Matrix
                confusion_matrices[metric][true_class][preds[metric]] += 1

            # Log Data for Detailed Report
            row = {
                'Image': filename,
                'True_Class': true_class,
                'Pred_E': preds['Euclidean'],
                'Pred_M': preds['Manhattan'],
                'Pred_C': preds['Cosine'],
                'Dist_E_Slice': round(dists['Euclidean']['pizza_slice'], 2),
                'Dist_E_Whole': round(dists['Euclidean']['whole_pizza'], 2),
                'Dist_E_Box':   round(dists['Euclidean']['pizza_box'], 2),
                'Dist_C_Slice': round(dists['Cosine']['pizza_slice'], 4),
                'Dist_C_Whole': round(dists['Cosine']['whole_pizza'], 4),
                'Dist_C_Box':   round(dists['Cosine']['pizza_box'], 4),
            }
            results.append(row)

    # 5. Save Files

    # A. Detailed Predictions CSV
    detailed_path = os.path.join(OUTPUT_DIR, "detailed_predictions.csv")
    keys = results[0].keys()
    with open(detailed_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"[Saved] Detailed Report -> {detailed_path}")

    # B. Confusion Matrices (3 Separate Files)
    for metric in ['Euclidean', 'Manhattan', 'Cosine']:
        cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{metric}.csv")
        with open(cm_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(["True Class \\ Predicted"] + CLASSES)
            # Rows
            for t_class in CLASSES:
                row = [t_class] + [confusion_matrices[metric][t_class][p] for p in CLASSES]
                writer.writerow(row)
        print(f"[Saved] Confusion Matrix ({metric}) -> {cm_path}")

    # C. Accuracy Summary Text
    summary_path = os.path.join(OUTPUT_DIR, "accuracy_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("--- Classification Accuracy Summary ---\n")
        f.write(f"Total Test Images: {total_images}\n\n")
        for metric in ['Euclidean', 'Manhattan', 'Cosine']:
            acc = (correct_counts[metric] / total_images) * 100
            f.write(f"{metric}: {acc:.2f}% ({correct_counts[metric]}/{total_images})\n")
    print(f"[Saved] Summary -> {summary_path}")

if __name__ == "__main__":
    run_evaluation()