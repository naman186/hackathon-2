import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# =============================
# 1️⃣ Configuration
# =============================
DATASET_PATH = "data"
IMG_SIZE = 64
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15

# =============================
# 2️⃣ Load Dataset
# =============================
X = []
y = []

# SORT classes for consistent label order
class_names = sorted(os.listdir(DATASET_PATH))

print("Detected Classes:", class_names)

for class_name in class_names:
    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X.append(img)
        y.append(class_name)

print("Total images loaded:", len(X))

# =============================
# 3️⃣ Convert to NumPy
# =============================
X = np.array(X, dtype="float32")
y = np.array(y)

# =============================
# 4️⃣ Normalize
# =============================
X = X / 255.0

# =============================
# 5️⃣ Encode Labels
# =============================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

num_classes = len(np.unique(y_encoded))
y_categorical = to_categorical(y_encoded, num_classes)

print("Encoded classes:", label_encoder.classes_)
print("Number of classes:", num_classes)

# =============================
# 6️⃣ Train / Val / Test Split
# =============================

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_categorical,
    test_size=TEST_SPLIT,
    stratify=y_encoded,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=VAL_SPLIT,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

# =============================
# 7️⃣ Save Everything
# =============================

np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
np.save("X_test.npy", X_test)

np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)
np.save("y_test.npy", y_test)

# Save class names (IMPORTANT for camera detection)
np.save("class_names.npy", label_encoder.classes_)

print("🔥 Data preparation complete and saved successfully!")