import os
import random
from sklearn.model_selection import train_test_split
import shutil

def prepare_datasets(frames_dir, output_dir, test_size=0.2, val_size=0.1):
    """
    Prepares train, validation, and test datasets from the extracted frames.

    Parameters:
    - frames_dir: Directory containing 'real' and 'manipulated' folders.
    - output_dir: Directory where the prepared datasets will be saved.
    - test_size: Proportion of the test set.
    - val_size: Proportion of the validation set (from remaining training data).
    """
    categories = ['real', 'manipulated']
    data = []
    
    # Gather all image paths and labels
    for category in categories:
        category_path = os.path.join(frames_dir, category)
        label = 0 if category == 'real' else 1
        
        for img_file in os.listdir(category_path):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(category_path, img_file)
                data.append((img_path, label))
    
    # Shuffle the data for randomness
    random.shuffle(data)
    
    # Split data into training, validation, and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, stratify=[x[1] for x in data])
    train_data, val_data = train_test_split(train_data, test_size=val_size, stratify=[x[1] for x in train_data])
    
    # Helper function to save images into folders
    def save_data(data, subset_name):
        subset_path = os.path.join(output_dir, subset_name)
        os.makedirs(subset_path, exist_ok=True)
        
        for img_path, label in data:
            label_folder = 'real' if label == 0 else 'manipulated'
            dest_folder = os.path.join(subset_path, label_folder)
            os.makedirs(dest_folder, exist_ok=True)
            shutil.copy(img_path, dest_folder)
    
    # Save data into train, val, and test folders
    save_data(train_data, 'train')
    save_data(val_data, 'val')
    save_data(test_data, 'test')
    
    print("Dataset preparation complete!")
    print(f"Training set: {len(train_data)} images")
    print(f"Validation set: {len(val_data)} images")
    print(f"Test set: {len(test_data)} images")

# Define paths
frames_path = "C:/Users/yashas/Downloads/archive/frames"
output_dataset_path = "C:/Users/yashas/Downloads/archive/dataprep"

# Prepare datasets
prepare_datasets(frames_path, output_dataset_path, test_size=0.2, val_size=0.1)
