import cv2
import face_recognition
import sqlite3
import os


def enroll_face(name):
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow('Enrollment', frame)

        # Capture image when 'c' is pressed
        if cv2.waitKey(1) & 0xFF == ord('c'):
            # Detect faces in the image
            face_locations = face_recognition.face_locations(frame)
            if len(face_locations) == 1:
                # Encode the face
                face_encoding = face_recognition.face_encodings(frame, face_locations)[0]

                # Save the encoding to the database
                conn = sqlite3.connect('data/attendance.db')
                c = conn.cursor()
                c.execute('''CREATE TABLE IF NOT EXISTS users
                             (id INTEGER PRIMARY KEY AUTOINCREMENT,
                              name TEXT,
                              face_encoding BLOB)''')
                c.execute("INSERT INTO users (name, face_encoding) VALUES (?, ?)",
                          (name, face_encoding.tobytes()))
                conn.commit()
                conn.close()

                print(f"Enrolled {name} successfully!")
                break
            else:
                print("Please ensure only one face is visible in the frame.")

        # Exit enrollment when 'q' is pressed
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    name = input("Enter the name of the person to enroll: ")
    enroll_face(name)