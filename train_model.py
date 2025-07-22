import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle

# ====== CORRECT PATH CONFIGURATION ======
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Gets to Fruit_QualityGrading
DATASET_PATH = os.path.join(BASE_DIR, "app", "dataset", "MY_data")  # Updated path
IMG_SIZE = 128
# =======================================

def load_dataset():
    """Load dataset with error handling"""
    X, y = [], []
    
    # Verify dataset path exists
    if not os.path.exists(DATASET_PATH):
        print(f"\n❌ Error: Dataset folder not found at:\n{DATASET_PATH}")
        print("\n🔍 Please verify:")
        print(f"1. The folder structure: app/dataset/MY_data/[fruit_folders]")
        print(f"2. Current working directory: {os.getcwd()}")
        return None, None, None

    # Get class folders
    CLASSES = [d for d in os.listdir(DATASET_PATH) 
               if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    if not CLASSES:
        print(f"\n❌ Error: No fruit folders found in:\n{DATASET_PATH}")
        return None, None, None

    label_map = {cls: i for i, cls in enumerate(CLASSES)}
    print(f"\n🔄 Loading dataset from:\n{DATASET_PATH}")
    print(f"📦 Found {len(CLASSES)} classes: {', '.join(CLASSES)}")

    for cls in CLASSES:
        cls_path = os.path.join(DATASET_PATH, cls)
        print(f"\n📂 Processing {cls}...")
        
        for img_file in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_file)
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️ Could not read {img_file} - skipping")
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(label_map[cls])
                
            except Exception as e:
                print(f"⚠️ Error processing {img_file}: {str(e)}")
                continue

    return np.array(X), np.array(y), label_map

# --- Main Training --- 
X, y, label_map = load_dataset()
if X is None:
    exit(1)  # Exit if dataset loading failed

# Preprocessing
X = X / 255.0
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train
print("\n🔮 Training model...")
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test), 
                    epochs=10,
                    batch_size=32)

# Save
model_dir = os.path.join(BASE_DIR, "app", "model")
os.makedirs(model_dir, exist_ok=True)

model.save(os.path.join(model_dir, "fruit_classifier.h5"))
with open(os.path.join(model_dir, "label_map.pkl"), "wb") as f:
    pickle.dump(label_map, f)

print(f"\n✅ Model saved to:\n{model_dir}")
print(f"➡️ Next step: Run `streamlit run app/main.py`")