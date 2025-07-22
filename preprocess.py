import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import sys

# Configuration - UPDATED PATHS
IMG_SIZE = 128
BASE_DIR = Path(__file__).parent.parent  # Goes up to Fruit_QualityGrading
DATASET_PATH = BASE_DIR / "app" / "dataset" / "MY_data"  # Corrected path
AUGMENTATION = True
CLASS_LIMIT = 800

def load_dataset():
    """Robust dataset loader with detailed error messages"""
    X, y = [], []
    class_counts = {}
    
    # Verify dataset path exists
    if not DATASET_PATH.exists():
        print(f"\n❌ Error: Dataset folder not found at:")
        print(f"Expected path: {DATASET_PATH}")
        print("\n🔍 Please verify:")
        print(f"1. The folder 'MY_data' exists in: {BASE_DIR / 'app' / 'dataset'}")
        print(f"2. It contains subfolders for each fruit class")
        print(f"3. Current working directory: {os.getcwd()}")
        sys.exit(1)

    # Get class folders
    classes = [d.name for d in DATASET_PATH.iterdir() 
               if d.is_dir() and not d.name.startswith('.')]
    
    if not classes:
        print(f"\n❌ Error: No fruit class folders found in: {DATASET_PATH}")
        print("Each fruit should have its own folder (e.g., apple/, banana/)")
        sys.exit(1)

    label_map = {cls: i for i, cls in enumerate(sorted(classes))}
    
    print(f"\n🔄 Loading dataset from: {DATASET_PATH}")
    print(f"📦 Found {len(classes)} classes: {', '.join(classes)}")

    for cls in classes:
        cls_path = DATASET_PATH / cls
        images_loaded = 0
        valid_extensions = ('.jpg', '.jpeg', '.png')

        # Count available images first
        total_images = len([f for f in cls_path.glob("*") 
                          if f.suffix.lower() in valid_extensions])
        
        print(f"\n📂 Processing {cls} ({total_images} images available)...")

        for img_file in cls_path.glob("*"):
            if img_file.suffix.lower() not in valid_extensions:
                continue
                
            if images_loaded >= CLASS_LIMIT:
                print(f"⚠️ Reached class limit ({CLASS_LIMIT}) for {cls}")
                break

            try:
                # Load and validate image
                img = cv2.imread(str(img_file))
                if img is None:
                    print(f"⚠️ Corrupted image: {img_file.name}")
                    continue

                # Preprocessing pipeline
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype('float32') / 255.0

                X.append(img)
                y.append(label_map[cls])
                images_loaded += 1

                # Basic augmentation
                if AUGMENTATION and images_loaded < CLASS_LIMIT:
                    X.append(cv2.flip(img, 1))  # Horizontal flip
                    y.append(label_map[cls])
                    images_loaded += 1

            except Exception as e:
                print(f"⚠️ Error processing {img_file.name}: {str(e)}")
                continue

        class_counts[cls] = images_loaded
        print(f"✅ Loaded {images_loaded} images")

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Balance classes (simple version)
    unique, counts = np.unique(y, return_counts=True)
    min_samples = min(counts)
    
    balanced_X = []
    balanced_y = []
    
    for class_id in unique:
        class_indices = np.where(y == class_id)[0]
        selected_indices = np.random.choice(
            class_indices, 
            min_samples, 
            replace=False
        )
        balanced_X.extend(X[selected_indices])
        balanced_y.extend(y[selected_indices])
    
    X = np.array(balanced_X)
    y = np.array(balanced_y)

    # Save visualization
    plot_class_distribution(class_counts)
    
    return (X, y), label_map

def plot_class_distribution(class_counts):
    """Create a bar plot of class distribution"""
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    
    plt.title("Final Class Distribution", pad=20)
    plt.xlabel("Fruit Classes")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45, ha='right')
    
    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save to static folder
    os.makedirs(BASE_DIR / "app" / "static", exist_ok=True)
    plt.savefig(BASE_DIR / "app" / "static" / "class_distribution.png")
    plt.close()
    print(f"\n📊 Saved distribution plot to: {BASE_DIR / 'app' / 'static' / 'class_distribution.png'}")

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print("FRUIT QUALITY GRADING - DATA PREPROCESSOR")
    print(f"{'='*50}\n")
    
    try:
        (X, y), label_map = load_dataset()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            stratify=y, 
            random_state=42
        )
        
        # Save artifacts
        os.makedirs(BASE_DIR / "app" / "model", exist_ok=True)
        with open(BASE_DIR / "app" / "model" / "label_map.pkl", "wb") as f:
            pickle.dump(label_map, f)
            
        print("\n🎉 Preprocessing completed successfully!")
        print(f"┌ {'─'*40} ┐")
        print(f"│ {'Training samples:':<20} {X_train.shape[0]:>18} │")
        print(f"│ {'Testing samples:':<20} {X_test.shape[0]:>18} │")
        print(f"│ {'Number of classes:':<20} {len(label_map):>18} │")
        print(f"└ {'─'*40} ┘")
        print(f"\n💾 Saved label map to: {BASE_DIR / 'app' / 'model' / 'label_map.pkl'}")
        
    except Exception as e:
        print(f"\n❌ Critical error during preprocessing: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Verify all fruit folders contain valid images")
        print("2. Check image formats (supported: .jpg, .jpeg, .png)")
        print("3. Ensure sufficient disk space")
        sys.exit(1)