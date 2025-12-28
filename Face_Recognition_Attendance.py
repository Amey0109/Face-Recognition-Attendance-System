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
from datetime import datetime, timedelta

# ---------------- CONFIG ----------------
DB_CONFIG = {
    "dbname": "Attendance_System",
    "user": "postgres",
    "password": "amey",
    "host": "localhost",
    "port": "5432"
}

DATASET_PATH = os.path.join(os.path.dirname(__file__), "Students")
PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")

EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 1
FACE_DISTANCE_THRESHOLD = 0.6
BLINK_WINDOW_SECONDS = 3

os.makedirs(DATASET_PATH, exist_ok=True)
capture_mode = False
student_names = {}


# ---------------- DB ----------------
conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

# ---------------- FACE DATA ----------------
known_encodings = []
known_student_ids = []

# Load student names
def load_student_names():
    student_names.clear()
    cursor.execute("SELECT student_id, full_name FROM student_details")
    for sid, name in cursor.fetchall():
        student_names[sid] = name

load_student_names()

def load_faces():
    known_encodings.clear()
    known_student_ids.clear()
    load_student_names()

    for folder in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder)
        if not os.path.isdir(folder_path):
            continue

        student_id = int(folder.split("_")[0])

        for img in os.listdir(folder_path):
            path = os.path.join(folder_path, img)
            image = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(image)
            if encs:
                known_encodings.append(encs[0])
                known_student_ids.append(student_id)

load_faces()

def detect_blink(gray, face_rect, sid):
    shape = predictor(gray, face_rect)
    shape = face_utils.shape_to_np(shape)

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0

    if sid not in blink_counter:
        blink_counter[sid] = 0
        last_blink[sid] = datetime.min

    if ear < EAR_THRESHOLD:
        blink_counter[sid] += 1
    else:
        if blink_counter[sid] >= EAR_CONSEC_FRAMES:
            blink_counter[sid] = 0
            last_blink[sid] = datetime.now()
            return True

        blink_counter[sid] = 0

    return False


# ---------------- DLIB ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# ---------------- GUI ----------------
root = tk.Tk()
root.title("NeuroFace AI")

menu = tk.Menu(root)
root.config(menu=menu)

student_menu = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Student", menu=student_menu)
student_menu.add_command(label="Add New Student", command=lambda: open_faculty_login())

camera_label = tk.Label(root)
camera_label.pack()

status = tk.Label(root, text="Waiting...", font=("Arial", 14))
status.pack()

# ---------------- CAMERA ----------------
video = cv2.VideoCapture(0)

# ---------------- FACULTY LOGIN ----------------
def open_faculty_login():
    login_win = tk.Toplevel(root)
    login_win.title("Faculty Login")
    login_win.geometry("300x220")

    tk.Label(login_win, text="Faculty ID").pack(pady=5)
    fid = tk.Entry(login_win)
    fid.pack()

    tk.Label(login_win, text="Password").pack(pady=5)
    fpass = tk.Entry(login_win, show="*")
    fpass.pack()

    def verify():
        faculty_id = fid.get().strip()
        password = fpass.get().strip()

        if not faculty_id or not password:
            messagebox.showwarning("Error", "All fields required")
            return

        try:
            cursor.execute(
                "SELECT faculty_id FROM faculty WHERE faculty_id=%s AND password=%s",
                (faculty_id, password)
            )
            if cursor.fetchone():
                login_win.destroy()  # destroy AFTER reading values
                open_student_registration(faculty_id)
            else:
                messagebox.showerror("Error", "Invalid credentials")
        except Exception as e:
            messagebox.showerror("DB Error", str(e))

    tk.Button(login_win, text="Login", command=verify).pack(pady=15)


# ---------------- STUDENT REGISTRATION ----------------
def open_student_registration(faculty_id):
    reg = tk.Toplevel(root)
    reg.title("Student Registration")

    fields = {}
    for label in ["Name", "Roll No", "Class"]:
        tk.Label(reg, text=label).pack()
        fields[label] = tk.Entry(reg)
        fields[label].pack()

    tk.Label(reg, text="Stream").pack()
    stream = ttk.Combobox(reg, values=["Science", "Commerce", "Arts", "None"], state="readonly")
    stream.current(3)
    stream.pack()

    def save_student():
        cursor.execute("""
            INSERT INTO student_details (full_name, roll_no, class, stream, image_folder, registered_by)
            VALUES (%s,%s,%s,%s,'',%s) RETURNING student_id
        """, (
            fields["Name"].get(),
            fields["Roll No"].get(),
            fields["Class"].get(),
            stream.get(),
            faculty_id
        ))

        student_id = cursor.fetchone()[0]
        folder = f"{student_id}_{fields['Name'].get().replace(' ', '_')}"
        folder_path = os.path.join(DATASET_PATH, folder)
        os.makedirs(folder_path, exist_ok=True)

        cursor.execute(
            "UPDATE student_details SET image_folder=%s WHERE student_id=%s",
            (folder_path, student_id)
        )
        conn.commit()

        capture_images(folder_path)
        load_faces()
        messagebox.showinfo("Success", "Student Added Successfully")

    tk.Button(reg, text="Save & Capture Images", command=save_student).pack(pady=10)

# ---------------- IMAGE CAPTURE ----------------
def capture_images(folder):
    global capture_mode
    capture_mode = True

    count = 0
    while count < 6:
        ret, frame = video.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow("Press SPACE to Capture | Q to Quit", frame)
        key = cv2.waitKey(1)

        if key == ord(' '):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)

            if boxes:
                top, right, bottom, left = boxes[0]
                face_img = frame[top:bottom, left:right]
                cv2.imwrite(os.path.join(folder, f"{count}.jpg"), face_img)
                count += 1

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    capture_mode = False

# ---------------- ATTENDANCE LOOP ----------------
blink_counter = {}
last_blink = {}

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1]-eye[5])
    B = np.linalg.norm(eye[2]-eye[4])
    C = np.linalg.norm(eye[0]-eye[3])
    return (A+B)/(2*C)

def process():
    ret, frame = video.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)

    faces = face_recognition.face_locations(small)
    encs = face_recognition.face_encodings(small, faces)

    for enc, (t, r, b, l) in zip(encs, faces):
        t, r, b, l = t*2, r*2, b*2, l*2
        name = "Unknown"
        color = (0, 0, 255)  # Red for unknown

        distances = face_recognition.face_distance(known_encodings, enc)
        if len(distances) > 0:
            idx = np.argmin(distances)

            if distances[idx] < FACE_DISTANCE_THRESHOLD:
                sid = known_student_ids[idx]
                name = student_names.get(sid, "Student")
                color = (0, 255, 255)  # Yellow box

                face_rect = dlib.rectangle(l, t, r, b)
                blinked = detect_blink(gray, face_rect, sid)

                if blinked:
                    try:
                        cursor.execute("""
                            INSERT INTO attendance (student_id, timestamp)
                            VALUES (%s, %s)
                            ON CONFLICT DO NOTHING
                        """, (sid, datetime.now()))
                        conn.commit()
                        status.config(text=f"Attendance Marked: {name}")
                    except:
                        conn.rollback()

        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(frame, name, (l, t-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    camera_label.configure(image=img)
    camera_label.image = img

    root.after(10, process)


process()
root.mainloop()

video.release()
cursor.close()
conn.close()
