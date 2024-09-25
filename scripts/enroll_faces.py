import cv2
import face_recognition
import os
import numpy as np
import sqlite3

def enroll_face(name, image_path):
    # Load the image
    image = face_recognition.load_image_file(image_path)
    
    # Find face locations and encodings
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    if len(face_encodings) == 0:
        print(f"No face found in the image for {name}")
        return None
    
    # Take the first face encoding (assuming one face per image)
    face_encoding = face_encodings[0]
    
    # Connect to the database
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute('''CREATE TABLE IF NOT EXISTS users
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       name TEXT NOT NULL,
                       face_encoding BLOB NOT NULL)''')
    
    # Insert the face encoding
    cursor.execute("INSERT INTO users (name, face_encoding) VALUES (?, ?)",
                   (name, face_encoding.tobytes()))
    
    conn.commit()
    conn.close()
    
    print(f"Enrolled {name} successfully")
    return face_encoding

# Example usage
enroll_face("John Doe", "path/to/john_doe.jpg")