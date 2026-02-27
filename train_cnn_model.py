import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =============================
# 1️⃣ Data Generators
# =============================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2   # 🔥 Auto split
)

train_generator = train_datagen.flow_from_directory(
    'data',
    target_size=(64,64),
    batch_size=128,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = train_datagen.flow_from_directory(
    'data',
    target_size=(64,64),
    batch_size=128,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)

print("Class indices:", train_generator.class_indices)

# =============================
# 2️⃣ Build CNN Model
# =============================

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dense(train_generator.num_classes, activation='softmax'))

# =============================
# 3️⃣ Compile
# =============================

model.compile(
    optimizer=Adam(learning_rate=0.0011),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

# =============================
# 4️⃣ Callbacks
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
# 5️⃣ Train
# =============================

history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint]
)

# =============================
# 6️⃣ Save Final Model
# =============================

model.save("cnn_model.keras")

print("🔥 Training Complete")