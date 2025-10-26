import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from tensorflow.keras.models import load_model

# ---------------------- Paths ----------------------
model_path = r"C:\Users\user\Sign-Language-detection\Model\keras_model.h5"
labels_path = r"C:\Users\user\Sign-Language-detection\Model\labels.txt"

# ---------------------- Load Model & Labels ----------------------
model = load_model(model_path)
labels = [line.strip() for line in open(labels_path)]

# ---------------------- Camera & Hand Detector ----------------------
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Get model input size (batch, height, width, channels)
_, h_in, w_in, _ = model.input_shape
view_sz = 300
offset = 20

while True:
    success, img = cap.read()
    if not success:
        break

    img_output = img.copy()
    hands, _ = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        img_white = np.ones((view_sz, view_sz, 3), dtype=np.uint8) * 255
        img_crop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        if img_crop.size > 0:
            aspect = h / w

            if aspect > 1:
                k = view_sz / h
                w_new = math.ceil(w * k)
                img_resized = cv2.resize(img_crop, (w_new, view_sz))
                gap = math.ceil((view_sz - w_new) / 2)
                img_white[:, gap:gap + w_new] = img_resized
            else:
                k = view_sz / w
                h_new = math.ceil(h * k)
                img_resized = cv2.resize(img_crop, (view_sz, h_new))
                gap = math.ceil((view_sz - h_new) / 2)
                img_white[gap:gap + h_new, :] = img_resized

            # Preprocess for model
            img_input = cv2.resize(img_white, (w_in, h_in))
            img_input = img_input.astype("float32") / 255.0
            img_input = np.expand_dims(img_input, axis=0)

            # Predict
            preds = model.predict(img_input, verbose=0)[0]
            index = np.argmax(preds)
            confidence = preds[index]

            # Display Prediction
            cv2.putText(img_output, f"{labels[index]}  {confidence:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.rectangle(img_output, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)

    cv2.imshow("Output", img_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
