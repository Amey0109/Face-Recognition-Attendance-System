import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import face_recognition
import numpy as np
import dlib
from imutils import face_utils
import psycopg2
import os
from datetime import datetime, timedelta, date

# ----------------- CONFIG -----------------
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "amey",
    "host": "localhost",
    "port": "5432"
}
DATASET_PATH = "Students" 
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 1
BLINK_WINDOW_SECONDS = 3  
FACE_DISTANCE_THRESHOLD = 0.6
CAM_WIDTH = 640
CAM_HEIGHT = 480


# --- Database connection ---
try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("Connected to DB")
except Exception as e:
    print("DB connection error:", e)
    raise

# --- EAR function ---
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# --- Load known faces dynamically ---
known_encodings = []
known_names = []

for person in os.listdir(DATASET_PATH):
    person_folder = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_folder):
        continue
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        try:
            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person)
        except Exception as e:
            print("Error loading", img_path, ":", e)

# --- Load dlib models ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Blink + marking states
blink_counter = 0
last_blink_time = {}
marked_today = set()  # names already marked today

# Check if already marked today
def already_marked_today(name):
    try:
        cursor.execute(
            "SELECT 1 FROM attendance WHERE name=%s AND DATE(timestamp)=%s LIMIT 1",
            (name, date.today())
        )
        return cursor.fetchone() is not None
    except Exception as e:
        print("DB error (check marked):", e)
        return False

for person in known_names:
    if already_marked_today(person):
        marked_today.add(person)

# --- GUI Setup ---
root = tk.Tk()
root.title("Face Recognition Attendance")

camera_label = tk.Label(root)
camera_label.pack()

notif_frame = tk.Frame(root, pady=8)
notif_frame.pack(fill="x")
notification_label = tk.Label(notif_frame, text="Waiting...", font=("Arial", 14), fg="black")
notification_label.pack()

status_label = tk.Label(root, text="", font=("Arial", 9), fg="gray")
status_label.pack()

def update_notification(msg, color="black"):
    notification_label.config(text=msg, fg=color)


# --- Video capture ---
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

if not video.isOpened():
    update_notification("Could not open webcam", "red")
    raise SystemExit("Webcam not available")

def process_frame():
    global blink_counter

    ret, frame = video.read()
    if not ret:
        update_notification("Failed to read from webcam", "red")
        root.after(100, process_frame)
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small_frame = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blink detection
    rects = detector(gray, 0)
    blink_detected = False
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEAR = eye_aspect_ratio(shape[lStart:lEnd])
        rightEAR = eye_aspect_ratio(shape[rStart:rEnd])
        ear = (leftEAR + rightEAR) / 2.0
        status_label.config(text=f"EAR: {ear:.3f}")

        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= EAR_CONSEC_FRAMES:
                for name in known_names:
                    last_blink_time[name] = datetime.now()
                blink_detected = True
                update_notification("Blink detected", "blue")
            blink_counter = 0

    # Face recognition
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    if not face_locations:
        update_notification("No face detected", "black")

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        top *= 2; right *= 2; bottom *= 2; left *= 2

        name = "Unknown"
        if known_encodings:
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_idx = np.argmin(face_distances)
            if face_distances[best_idx] < FACE_DISTANCE_THRESHOLD:
                name = known_names[best_idx]

        # Draw face box
        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if name == "Unknown":
            update_notification("Face not matched ", "red")
        else:
            last_time = last_blink_time.get(name, None)
            if last_time and datetime.now() - last_time <= timedelta(seconds=BLINK_WINDOW_SECONDS):
                if name not in marked_today:
                    try:
                        cursor.execute(
                            "INSERT INTO attendance (name, timestamp) VALUES (%s, %s)",
                            (name, datetime.now())
                        )
                        conn.commit()
                        marked_today.add(name)
                        update_notification(f"Hey {name}, your attendance has been marked", "green")
                    except Exception as e:
                        conn.rollback()
                        update_notification(f"DB Error: {e}", "red")
                else:
                    update_notification(f"{name} already marked today", "orange")
            else:
                if name not in marked_today:
                    update_notification(f"{name} recognized — blink to mark", "orange")

    # Show frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)

    root.after(10, process_frame)

root.after(0, process_frame)
root.mainloop()

video.release()
cursor.close()
conn.close()
