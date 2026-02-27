import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =============================
# 1️⃣ Load Data
# =============================

X = np.load('X_train.npy')
y = np.load('y_train.npy')   # ⚠️ Already one-hot encoded

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)

# =============================
# 2️⃣ Normalize
# =============================

X = X.astype("float32") / 255.0

# =============================
# 3️⃣ Train / Validation Split
# =============================

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y, axis=1)
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# =============================
# 4️⃣ Build Model (Modern Style)
# =============================

model = Sequential([
    Input(shape=(64, 64, 3)),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.4),

    Dense(64, activation='relu'),

    Dense(y.shape[1], activation='softmax')
])

# =============================
# 5️⃣ Compile
# =============================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

# =============================
# 6️⃣ Callbacks
# =============================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_cnn_model.keras',
    monitor='val_loss',
    save_best_only=True
)

# =============================
# 7️⃣ Train
# =============================

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint]
)

# =============================
# 8️⃣ Save Final Model
# =============================

model.save("cnn_model.keras")

print("🔥 Training Complete & Model Saved")