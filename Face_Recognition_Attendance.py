import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import face_recognition
import numpy as np
import dlib
from imutils import face_utils
import psycopg2
import os
from datetime import date 


# ---------------- CONFIG ----------------
DB_CONFIG = {
    "dbname": "Attendance_System",
    "user": "postgres",
    "password": "amey",
    "host": "localhost",
    "port": "5432"
}

# ---------------- Hash verify ----------------
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


DATASET_PATH = "Students"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")


EAR_THRESHOLD = 0.25
FACE_DISTANCE_THRESHOLD = 0.6

os.makedirs(DATASET_PATH, exist_ok=True)

# ---------------- GLOBAL CONTEXT ----------------
CURRENT_INSTITUTE_ID = None
CURRENT_FACULTY_EMAIL = None
CURRENT_FACULTY_ID = None

# ---------------- DB ----------------
conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

# ---------------- FACE DATA ----------------
known_encodings = []
known_student_ids = {}
student_names = {}

def load_students():
    known_encodings.clear()
    known_student_ids.clear()
    student_names.clear()

    cursor.execute("""
        SELECT student_id, full_name, image_folder
        FROM student_details
        WHERE institute_id=%s AND is_active=true
    """, (CURRENT_INSTITUTE_ID,))

    for sid, name, folder in cursor.fetchall():
        student_names[sid] = name
        if folder and os.path.isdir(folder):
            for img in os.listdir(folder):
                path = os.path.join(folder, img)
                image = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(image)
                if encs:
                    known_encodings.append(encs[0])
                    known_student_ids[len(known_encodings)-1] = sid

# ---------------- BLINK ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1]-eye[5])
    B = np.linalg.norm(eye[2]-eye[4])
    C = np.linalg.norm(eye[0]-eye[3])
    return (A+B)/(2*C)

def detect_blink(gray, rect):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    leftEAR = eye_aspect_ratio(shape[lStart:lEnd])
    rightEAR = eye_aspect_ratio(shape[rStart:rEnd])
    return ((leftEAR + rightEAR) / 2.0) < EAR_THRESHOLD

# ---------------- GUI ----------------
root = tk.Tk()
root.title("NeuroFace AI")
root.geometry("900x600")

camera_label = tk.Label(root)
camera_label.pack()

status = tk.Label(root, text="Please login", font=("Arial", 14))
status.pack()

video = cv2.VideoCapture(0)

# ---------------- FACULTY LOGIN ----------------
def faculty_login():
    login = tk.Toplevel(root)
    login.title("Faculty Login")
    login.geometry("300x220")

    tk.Label(login, text="Email").pack()
    email = tk.Entry(login)
    email.pack()

    tk.Label(login, text="Password").pack()
    password = tk.Entry(login, show="*")
    password.pack()

    def verify():
        global CURRENT_INSTITUTE_ID, CURRENT_FACULTY_EMAIL, CURRENT_FACULTY_ID

        entered_email = email.get().strip()
        entered_password = password.get().strip()

        cursor.execute("""
            SELECT institute_id, email, password_hash
            FROM users
            WHERE email=%s AND UPPER(role)='FACULTY'
        """, (entered_email,))

        user = cursor.fetchone()
        if not user:
            messagebox.showerror("Error", "Invalid login")
            return

        stored_hash = user[2]

        # ✅ PASSLIB VERIFICATION
        if not verify_password(entered_password, stored_hash):
            messagebox.showerror("Error", "Invalid login")
            return

        # ✅ Login success
        CURRENT_INSTITUTE_ID = user[0]
        CURRENT_FACULTY_EMAIL = user[1]

        cursor.execute("""
            SELECT id
            FROM faculty
            WHERE email=%s AND institute_id=%s
        """, (CURRENT_FACULTY_EMAIL, CURRENT_INSTITUTE_ID))

        faculty = cursor.fetchone()
        if not faculty:
            messagebox.showerror("Error", "Faculty record not found")
            return

        CURRENT_FACULTY_ID = faculty[0]

        load_students()
        status.config(
            text=f"Logged in | Institute: {CURRENT_INSTITUTE_ID} | Faculty ID: {CURRENT_FACULTY_ID}"
        )
        login.destroy()
        process_camera()


    tk.Button(login, text="Login", command=verify).pack(pady=10)

faculty_login()

# ---------------- IMAGE CAPTURE ----------------
def capture_images(folder):
    count = 0
    while count < 5:
        ret, frame = video.read()
        cv2.imshow("Capture - Press SPACE", frame)
        if cv2.waitKey(1) == ord(' '):
            cv2.imwrite(os.path.join(folder, f"{count}.jpg"), frame)
            count += 1
    cv2.destroyAllWindows()

# ---------------- CAMERA LOOP ----------------
def process_camera():
    ret, frame = video.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, faces)

    for enc, (t, r, b, l) in zip(encs, faces):
        distances = face_recognition.face_distance(known_encodings, enc)
        if len(distances) > 0:
            idx = np.argmin(distances)
            if distances[idx] < FACE_DISTANCE_THRESHOLD:
                sid = known_student_ids[idx]
                rect = dlib.rectangle(l, t, r, b)

                if detect_blink(gray, rect):
                    cursor.execute("""
                        INSERT INTO attendance (
                            student_id, attendance_date, institute_id
                        ) VALUES (%s,%s,%s)
                        ON CONFLICT DO NOTHING
                    """, (sid, date.today(), CURRENT_INSTITUTE_ID))
                    conn.commit()

                    status.config(
                        text=f"Attendance marked: {student_names[sid]}"
                    )

                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    student_names[sid],
                    (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

    img = ImageTk.PhotoImage(Image.fromarray(frame))
    camera_label.configure(image=img)
    camera_label.image = img

    root.after(10, process_camera)

root.mainloop()

video.release()
cursor.close()
conn.close()
