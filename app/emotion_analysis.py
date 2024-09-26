import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

def load_emotion_model():
    model = load_model('models/emotion_model.h5')
    return model

def detect_emotion(frame, face_cascade, emotion_model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_model.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame

def run_emotion_analysis():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_model = load_emotion_model()

    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        frame = detect_emotion(frame, face_cascade, emotion_model)
        cv2.imshow('Emotion Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

EMOTIONS = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

if __name__ == "__main__":
    run_emotion_analysis()