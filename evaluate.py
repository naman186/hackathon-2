import numpy as np
import tensorflow as tf

# =============================
# Load Model
# =============================

model = tf.keras.models.load_model("cnn_model.keras")

# =============================
# Load Data
# =============================

X = np.load('X_train.npy')
y = np.load('y_train.npy')

X = X.astype("float32") / 255.0

# =============================
# Evaluate
# =============================

loss, acc = model.evaluate(X, y)

print("Loss:", loss)
print("Accuracy:", acc)