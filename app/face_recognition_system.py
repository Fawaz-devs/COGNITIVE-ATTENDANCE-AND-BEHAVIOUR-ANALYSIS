import cv2
import face_recognition
import numpy as np
import sqlite3
from datetime import datetime

def get_enrolled_faces():
    conn = sqlite3.connect('../data/attendance.db')
    c = conn.cursor()
    c.execute("SELECT name, face_encoding FROM users")
    enrolled_faces = {name: np.frombuffer(encoding, dtype=np.float64) for name, encoding in c.fetchall()}
    conn.close()
    return enrolled_faces

def mark_attendance(name):
    conn = sqlite3.connect('../data/attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute("INSERT INTO attendance (name) VALUES (?)", (name,))
    conn.commit()
    conn.close()

def run_recognition():
    enrolled_faces = get_enrolled_faces()
    known_face_encodings = list(enrolled_faces.values())
    known_face_names = list(enrolled_faces.keys())

    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                mark_attendance(name)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()