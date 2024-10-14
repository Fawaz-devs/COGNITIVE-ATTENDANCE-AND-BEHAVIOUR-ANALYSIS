import os
import pickle
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import face_recognition
from deepface import DeepFace

# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set up the template folder path
template_folder = os.path.join(current_dir, 'templates')

# Define paths for storing face data
face_data_file = os.path.join(current_dir, 'face_data.pkl')

app = Flask(__name__, template_folder=template_folder)
app.config['SECRET_KEY'] = '123456'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Hardcoded user credentials
USERS = {
    'admin': 'password123',
    'user1': 'password456'
}

# Initialize face data
known_face_encodings = []
known_face_names = []

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

def load_face_data():
    global known_face_encodings, known_face_names
    if os.path.exists(face_data_file):
        with open(face_data_file, 'rb') as f:
            data = pickle.load(f)
            known_face_encodings = data['encodings']
            known_face_names = data['names']
        print(f"Loaded {len(known_face_names)} face(s) from storage.")

def save_face_data():
    with open(face_data_file, 'wb') as f:
        pickle.dump({
            'encodings': known_face_encodings,
            'names': known_face_names
        }, f)
    print(f"Saved {len(known_face_names)} face(s) to storage.")

# Load existing face data when the app starts
load_face_data()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in USERS and USERS[username] == password:
            user = User(username)
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/enroll', methods=['GET', 'POST'])
@login_required
def enroll():
    if request.method == 'POST':
        username = request.form['username']
        
        # Capture multiple images for enrollment
        camera = cv2.VideoCapture(0)
        face_encodings = []
        for _ in range(5):  # Capture 5 images
            success, frame = camera.read()
            if success:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                if face_locations:
                    encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                    face_encodings.append(encoding)
        camera.release()

        if face_encodings:
            # Save the average encoding and username
            average_encoding = np.mean(face_encodings, axis=0)
            known_face_encodings.append(average_encoding)
            known_face_names.append(username)
            save_face_data()
            flash(f'Enrollment successful for {username}', 'success')
        else:
            flash('No face detected, enrollment failed', 'error')

        return redirect(url_for('index'))
    return render_template('enrollment.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

import time

def generate_frames():
    camera = cv2.VideoCapture(0)
    
    # Set the frame rate to 40 FPS
    camera.set(cv2.CAP_PROP_FPS, 60)
    
    # Get the actual frame rate (it might be lower than requested depending on the camera)
    actual_fps = camera.get(cv2.CAP_PROP_FPS)
    print(f"Actual camera FPS: {actual_fps}")
    
    # Calculate the time to wait between frames
    frame_time = 1 / 40  # 40 FPS

    while True:
        start_time = time.time()
        
        success, frame = camera.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Use face_distance to get the best match
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    min_distance = face_distances[best_match_index]
                    
                    if min_distance < 0.5:  # Adjust this threshold as needed
                        name = known_face_names[best_match_index]
                    else:
                        name = "Unknown"
                else:
                    name = "Unknown"

                # Emotion Analysis
                try:
                    analysis = DeepFace.analyze(rgb_frame[top:bottom, left:right], actions=['emotion'], enforce_detection=False)
                    emotion = analysis[0]['dominant_emotion']
                    cv2.putText(frame, f'{name}: {emotion}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                except Exception as e:
                    print("Error analyzing emotions:", e)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            # Calculate the time taken for processing this frame
            process_time = time.time() - start_time
            
            # If processing took less time than the frame time, wait for the remaining time
            if process_time < frame_time:
                time.sleep(frame_time - process_time)

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recognition', methods=['GET', 'POST'])
@login_required
def start_recognition():
    return render_template('start_recognition.html')  # Template with video feed

@app.route('/start_emotion_analysis', methods=['GET', 'POST'])
@login_required
def start_emotion_analysis():
    return render_template('start_emotion_analysis.html')  # Template with emotion analysis

@app.route('/enrolled_faces')
@login_required
def enrolled_faces():
    return render_template('enrolled_faces.html', faces=list(zip(known_face_names, range(len(known_face_names)))))

@app.route('/delete_face/<int:index>')
@login_required
def delete_face(index):
    if 0 <= index < len(known_face_names):
        del known_face_names[index]
        del known_face_encodings[index]
        save_face_data()
        flash('Face deleted successfully', 'success')
    else:
        flash('Invalid face index', 'error')
    return redirect(url_for('enrolled_faces'))

if __name__ == '__main__':
    app.run(debug=True)