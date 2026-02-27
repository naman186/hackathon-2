import cv2
import numpy as np
import tensorflow as tf

# =============================
# 1️⃣ Load Model
# =============================
model = tf.keras.models.load_model("cnn_model.h5")

# ⚠️ Replace with your real class names
class_names = [
    "Class1",
    "Class2",
    "Class3",
    "Class4",
    "Class5",
    "Class6",
    "Class7",
    "Class8"
]

# =============================
# 2️⃣ Start Camera
# =============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # =============================
    # 3️⃣ Define Green Box Area (Center)
    # =============================
    box_size = 300
    x1 = width // 2 - box_size // 2
    y1 = height // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    # Draw Green Rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # =============================
    # 4️⃣ Crop ROI (Inside Box Only)
    # =============================
    roi = frame[y1:y2, x1:x2]

    # Resize to model size
    img = cv2.resize(roi, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # =============================
    # 5️⃣ Predict
    # =============================
    predictions = model.predict(img, verbose=0)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)

    label = f"{class_names[class_index]} ({confidence*100:.2f}%)"

    # Show prediction above box
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.imshow("Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()