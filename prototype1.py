import cv2
import mediapipe as mp
import numpy as np
import pygame
import math

# === UPDATE THIS PATH ===
BUZZER_SOUND_PATH = r"C:\Users\asus\Downloads\buzzer.mp3"

# Landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_OUTER = [61, 291, 0, 17, 314, 405, 321, 375, 291]
MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(eye):
    A = euclidean_distance(eye[1], eye[5])
    B = euclidean_distance(eye[2], eye[4])
    C = euclidean_distance(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth_outer, mouth_inner):
    outer_vertical = euclidean_distance(mouth_outer[2], mouth_outer[6])
    inner_vertical = euclidean_distance(mouth_inner[2], mouth_inner[6])
    horizontal = euclidean_distance(mouth_outer[0], mouth_outer[4])
    return (outer_vertical + inner_vertical) / (2.0 * horizontal)

def calculate_head_angle(face_landmarks, frame_shape):
    h, w = frame_shape[:2]
    
    # 3D model points (generic head model)
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),      # Chin
        (-225.0, 170.0, -135.0),   # Left eye left corner
        (225.0, 170.0, -135.0),    # Right eye right corner
        (-150.0, -150.0, -125.0),  # Mouth left corner
        (150.0, -150.0, -125.0)    # Mouth right corner
    ], dtype=np.float64)
    
    # Corresponding 2D image points
    image_points = np.array([
        (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),    # Nose tip
        (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h), # Chin
        (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),   # Left eye left
        (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h), # Right eye right
        (face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h),   # Mouth left
        (face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h)  # Mouth right
    ], dtype=np.float64)
    
    # Camera matrix approximation
    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros((4,1))  # Assuming no lens distortion
    
    # Solve PnP to get rotation and translation vectors
    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    if not success:
        return 0  # Return neutral angle if estimation fails
    
    # Convert rotation vector to rotation matrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    
    # Calculate Euler angles
    # Extract pitch (nodding up/down)
    pitch = math.degrees(math.asin(rotation_mat[2, 1]))
    return pitch

# Initialize pygame mixer
pygame.mixer.init()
buzzer_playing = False

def play_buzzer():
    global buzzer_playing
    if not buzzer_playing:
        buzzer_playing = True
        pygame.mixer.music.load(BUZZER_SOUND_PATH)
        pygame.mixer.music.play(-1)

def stop_buzzer():
    global buzzer_playing
    if buzzer_playing:
        pygame.mixer.music.stop()
        buzzer_playing = False

# MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Thresholds
EAR_THRESHOLD = 0.18
MAR_THRESHOLD = 0.75
HEAD_DOWN_ANGLE_THRESHOLD = 25  # Degrees

CLOSED_FRAMES_THRESHOLD = 30
YAWN_FRAMES_THRESHOLD = 20
HEAD_DOWN_FRAMES_THRESHOLD = 30

closed_frames = 0
yawn_frames = 0
head_down_frames = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            
            # Get landmarks
            left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]
            mouth_outer = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in MOUTH_OUTER]
            mouth_inner = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in MOUTH_INNER]
            
            # Calculate ratios
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(mouth_outer, mouth_inner)
            
            # Calculate head pitch angle (with error handling)
            try:
                pitch_angle = calculate_head_angle(face_landmarks, frame.shape)
            except:
                pitch_angle = 0
            
            # Eye closure detection
            if avg_ear < EAR_THRESHOLD:
                closed_frames += 1
            else:
                closed_frames = 0
                
            # Yawning detection
            if mar > MAR_THRESHOLD:
                yawn_frames += 1
            else:
                yawn_frames = 0
                
            # Head down detection
            if abs(pitch_angle) > HEAD_DOWN_ANGLE_THRESHOLD:
                head_down_frames += 1
            else:
                head_down_frames = 0
            
            # Display metrics
            cv2.putText(frame, f'EAR: {avg_ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f'MAR: {mar:.2f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f'Head Pitch: {pitch_angle:.1f}°', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Visual feedback for head position
            if abs(pitch_angle) > HEAD_DOWN_ANGLE_THRESHOLD:
                cv2.putText(frame, "HEAD DOWN", (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Check for sleepiness conditions
            sleepy = False
            if closed_frames > CLOSED_FRAMES_THRESHOLD:
                cv2.putText(frame, "SLEEPY (Eyes Closed)!", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                sleepy = True
            elif yawn_frames > YAWN_FRAMES_THRESHOLD:
                cv2.putText(frame, "SLEEPY (Yawning)!", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                sleepy = True
            elif head_down_frames > HEAD_DOWN_FRAMES_THRESHOLD:
                cv2.putText(frame, "SLEEPY (Head Down)!", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                sleepy = True
            
            # Control buzzer
            if sleepy:
                play_buzzer()
            else:
                stop_buzzer()

    cv2.imshow("Sleepiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
stop_buzzer()
pygame.mixer.quit()