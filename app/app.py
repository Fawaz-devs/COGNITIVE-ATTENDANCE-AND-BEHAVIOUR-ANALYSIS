import os
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

app = Flask(__name__, template_folder=template_folder)
app.config['SECRET_KEY'] = '123456'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Hardcoded user credentials and encodings
USERS = {
    'admin': 'password123',
    'user1': 'password456'
}

# Store known face encodings and names
known_face_encodings = []
known_face_names = []

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

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
        
        # Capture image for enrollment
        camera = cv2.VideoCapture(0)
        success, frame = camera.read()
        camera.release()

        if success:
            # Encode the captured image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_frame)

            if face_encodings:
                # Save the encoding and username
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(username)
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

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # Read the camera frame
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces and recognize emotions
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Check if the face matches any known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Use the first match
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                # Emotion Analysis
                try:
                    analysis = DeepFace.analyze(rgb_frame[top:bottom, left:right], actions=['emotion'], enforce_detection=False)
                    emotions = analysis[0]['dominant_emotion']
                    cv2.putText(frame, f'{name}: {emotions}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                except Exception as e:
                    print("Error analyzing emotions:", e)

                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Encode the frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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

if __name__ == '__main__':
    app.run(debug=True)
