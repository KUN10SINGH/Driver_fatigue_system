import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import time
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from datetime import datetime
from collections import deque
import threading
import csv

class DrowsinessDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Drowsiness Detection System")
        self.root.geometry("1280x720")
        self.root.configure(bg="#f5f5f5")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # === Configuration and Variables ===
        # Thresholds
        self.EAR_THRESHOLD = 0.22  # Eye aspect ratio threshold
        self.HEAD_DOWN_ANGLE_THRESHOLD = 15  # Degrees for head pitch
        self.CLOSED_FRAMES_THRESHOLD = 36    # About 1.2 seconds at 30fps
        self.HEAD_DOWN_FRAMES_THRESHOLD = 24  # About 0.8 seconds at 30fps
        
        # Drowsiness Score System
        self.MAX_DROWSINESS_SCORE = 100
        self.ALERT_THRESHOLD = 70
        self.CRITICAL_THRESHOLD = 90
        
        # Score accumulation rates (points per frame)
        self.EYE_CLOSED_SCORE_RATE = 1.5
        self.HEAD_DOWN_SCORE_RATE = 1.2
        
        # Score decay rate (points per frame when not drowsy)
        self.SCORE_DECAY_RATE = 0.5
        
        # Working variables
        self.running = False
        self.camera_id = 0
        self.drowsiness_score = 0
        self.ear_values = deque(maxlen=10)
        self.head_angle_values = deque(maxlen=10)
        self.current_alert_level = 0
        self.last_alert_time = 0
        self.REPEAT_ALERT_DELAY = 8  # seconds
        self.buzzer_start_time = None
        
        # Alert levels
        self.LEVEL_NORMAL = 0
        self.LEVEL_WARNING = 1
        self.LEVEL_ALERT = 2
        self.LEVEL_CRITICAL = 3
        
        # MediaPipe Setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=1, 
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Set up directories
        self.LOGS_DIR = "drowsiness_logs"
        self.IMAGES_DIR = os.path.join(self.LOGS_DIR, "images")
        self.REPORTS_DIR = os.path.join(self.LOGS_DIR, "reports")
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        os.makedirs(self.IMAGES_DIR, exist_ok=True)
        os.makedirs(self.REPORTS_DIR, exist_ok=True)
        
        # Log file path
        self.LOG_FILE = os.path.join(self.LOGS_DIR, f"drowsiness_log_{datetime.now().strftime('%Y%m%d')}.txt")
        
        # Default sound path - will be updated by settings
        self.BUZZER_SOUND_PATH = ""
        
        # Alert counters
        self.session_alerts = {
            "warnings": 0,
            "alerts": 0,
            "critical": 0
        }
        
        self.session_start_time = datetime.now()
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Create GUI elements
        self.create_widgets()
        
        # Load settings
        self.load_settings()
    
    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel (video feed and controls)
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right panel (metrics and settings)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
        
        # Video feed frame
        video_frame = ttk.LabelFrame(left_panel, text="Camera Feed")
        video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_canvas = tk.Canvas(video_frame, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        control_frame = ttk.Frame(left_panel)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start Detection", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        self.camera_label = ttk.Label(control_frame, text="Camera:")
        self.camera_label.pack(side=tk.LEFT, padx=5)
        
        self.camera_var = tk.StringVar(value="0")
        self.camera_combo = ttk.Combobox(control_frame, textvariable=self.camera_var, values=["0", "1", "2"], width=5)
        self.camera_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Generate Report", command=self.generate_report).pack(side=tk.RIGHT, padx=5)
        
        # Metrics panel
        metrics_frame = ttk.LabelFrame(right_panel, text="Drowsiness Metrics")
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Drowsiness progress bar
        ttk.Label(metrics_frame, text="Drowsiness Level:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.drowsiness_bar = ttk.Progressbar(metrics_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.drowsiness_bar.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=3)
        self.drowsiness_label = ttk.Label(metrics_frame, text="0%")
        self.drowsiness_label.grid(row=0, column=2, sticky=tk.E, padx=5, pady=3)
        
        # Eye aspect ratio
        ttk.Label(metrics_frame, text="Eye Openness (EAR):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.ear_bar = ttk.Progressbar(metrics_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.ear_bar.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=3)
        self.ear_label = ttk.Label(metrics_frame, text="0.00")
        self.ear_label.grid(row=1, column=2, sticky=tk.E, padx=5, pady=3)
        
        # Head angle
        ttk.Label(metrics_frame, text="Head Position:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        self.head_bar = ttk.Progressbar(metrics_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.head_bar.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=3)
        self.head_label = ttk.Label(metrics_frame, text="0.0°")
        self.head_label.grid(row=2, column=2, sticky=tk.E, padx=5, pady=3)
        
        # Status indicator
        self.status_frame = ttk.Frame(metrics_frame)
        self.status_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.status_label = ttk.Label(self.status_frame, text="SYSTEM READY", font=("Arial", 14, "bold"))
        self.status_label.pack()
        
        # Session statistics
        stats_frame = ttk.LabelFrame(right_panel, text="Session Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(stats_frame, text="Session Duration:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.duration_label = ttk.Label(stats_frame, text="00:00:00")
        self.duration_label.grid(row=0, column=1, sticky=tk.E, padx=5, pady=3)
        
        ttk.Label(stats_frame, text="Warning Alerts:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.warnings_label = ttk.Label(stats_frame, text="0")
        self.warnings_label.grid(row=1, column=1, sticky=tk.E, padx=5, pady=3)
        
        ttk.Label(stats_frame, text="Drowsiness Alerts:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        self.alerts_label = ttk.Label(stats_frame, text="0")
        self.alerts_label.grid(row=2, column=1, sticky=tk.E, padx=5, pady=3)
        
        ttk.Label(stats_frame, text="Critical Alerts:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=3)
        self.critical_label = ttk.Label(stats_frame, text="0")
        self.critical_label.grid(row=3, column=1, sticky=tk.E, padx=5, pady=3)
        
        # Settings panel
        settings_frame = ttk.LabelFrame(right_panel, text="Settings")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # EAR Threshold
        ttk.Label(settings_frame, text="EAR Threshold:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.ear_threshold_var = tk.DoubleVar(value=self.EAR_THRESHOLD)
        ear_scale = ttk.Scale(settings_frame, from_=0.15, to=0.30, variable=self.ear_threshold_var, 
                             command=lambda v: self.ear_threshold_label.config(text=f"{float(v):.2f}"))
        ear_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=3)
        self.ear_threshold_label = ttk.Label(settings_frame, text=f"{self.EAR_THRESHOLD:.2f}")
        self.ear_threshold_label.grid(row=0, column=2, sticky=tk.E, padx=5, pady=3)
        
        # Head Angle Threshold
        ttk.Label(settings_frame, text="Head Angle Threshold:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.head_threshold_var = tk.DoubleVar(value=self.HEAD_DOWN_ANGLE_THRESHOLD)
        head_scale = ttk.Scale(settings_frame, from_=5, to=30, variable=self.head_threshold_var,
                              command=lambda v: self.head_threshold_label.config(text=f"{float(v):.1f}°"))
        head_scale.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=3)
        self.head_threshold_label = ttk.Label(settings_frame, text=f"{self.HEAD_DOWN_ANGLE_THRESHOLD:.1f}°")
        self.head_threshold_label.grid(row=1, column=2, sticky=tk.E, padx=5, pady=3)
        
        # Alert Threshold
        ttk.Label(settings_frame, text="Alert Threshold:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        self.alert_threshold_var = tk.IntVar(value=self.ALERT_THRESHOLD)
        alert_scale = ttk.Scale(settings_frame, from_=50, to=90, variable=self.alert_threshold_var,
                              command=lambda v: self.alert_threshold_label.config(text=f"{int(float(v))}%"))
        alert_scale.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=3)
        self.alert_threshold_label = ttk.Label(settings_frame, text=f"{self.ALERT_THRESHOLD}%")
        self.alert_threshold_label.grid(row=2, column=2, sticky=tk.E, padx=5, pady=3)
        
        # Critical Threshold
        ttk.Label(settings_frame, text="Critical Threshold:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=3)
        self.critical_threshold_var = tk.IntVar(value=self.CRITICAL_THRESHOLD)
        critical_scale = ttk.Scale(settings_frame, from_=70, to=100, variable=self.critical_threshold_var,
                                 command=lambda v: self.critical_threshold_label.config(text=f"{int(float(v))}%"))
        critical_scale.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=3)
        self.critical_threshold_label = ttk.Label(settings_frame, text=f"{self.CRITICAL_THRESHOLD}%")
        self.critical_threshold_label.grid(row=3, column=2, sticky=tk.E, padx=5, pady=3)
        
        # Alert Sound
        ttk.Label(settings_frame, text="Alert Sound:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=3)
        self.sound_path_var = tk.StringVar(value="")
        self.sound_path_entry = ttk.Entry(settings_frame, textvariable=self.sound_path_var, width=20)
        self.sound_path_entry.grid(row=4, column=1, sticky=tk.EW, padx=5, pady=3)
        ttk.Button(settings_frame, text="Browse", command=self.browse_sound_file).grid(row=4, column=2, sticky=tk.E, padx=5, pady=3)
        
        # Save settings button
        ttk.Button(settings_frame, text="Save Settings", command=self.save_settings).grid(row=5, column=0, columnspan=3, pady=10)
        
        # Start statistics updater
        self.update_session_stats()
    
    def start_detection(self):
        """Start the drowsiness detection process"""
        try:
            self.camera_id = int(self.camera_var.get())
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Could not open camera ID {self.camera_id}")
                return
                
            # Update threshold values from settings
            self.EAR_THRESHOLD = self.ear_threshold_var.get()
            self.HEAD_DOWN_ANGLE_THRESHOLD = self.head_threshold_var.get()
            self.ALERT_THRESHOLD = self.alert_threshold_var.get()
            self.CRITICAL_THRESHOLD = self.critical_threshold_var.get()
            
            # Reset session counters if starting new session
            if not self.running:
                self.session_start_time = datetime.now()
                self.session_alerts = {
                    "warnings": 0,
                    "alerts": 0,
                    "critical": 0
                }
                self.update_session_labels()
                
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.camera_combo.config(state=tk.DISABLED)
            
            # Start detection thread
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection: {str(e)}")
    
    def stop_detection(self):
        """Stop the drowsiness detection process"""
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.camera_combo.config(state=tk.NORMAL)
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            
        # Reset status
        self.status_label.config(text="SYSTEM READY", foreground="black")
        
        # Stop sound if playing
        if self.buzzer_start_time is not None:
            pygame.mixer.music.stop()
            self.buzzer_start_time = None
    
    def detection_loop(self):
        """Main drowsiness detection process"""
        while self.running and hasattr(self, 'cap') and self.cap.isOpened():
            success, frame = self.cap.read()
            
            if not success:
                self.running = False
                messagebox.showerror("Error", "Failed to capture frame from camera")
                self.root.after(0, self.stop_detection)
                break
                
            frame = cv2.flip(frame, 1)  # Mirror for more intuitive display
            
            self.process_frame(frame)
            
            # Convert frame to display on GUI
            self.display_frame(frame)
            
            # Update metrics on GUI (using main thread)
            self.root.after(0, self.update_metrics_display)
            
            # Check if buzzer should be stopped
            if self.buzzer_start_time is not None and time.time() - self.buzzer_start_time >= 3:
                pygame.mixer.music.stop()
                self.buzzer_start_time = None
    
    def process_frame(self, frame):
        """Process a video frame for drowsiness detection"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        current_time = time.time()
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                
                # Get eye landmarks
                left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) 
                            for i in self.LEFT_EYE]
                right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) 
                             for i in self.RIGHT_EYE]
                
                # Calculate raw metrics
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                try:
                    pitch_angle = self.calculate_head_angle(face_landmarks, frame.shape)
                except:
                    pitch_angle = 0
                
                # Add to moving average queues
                self.ear_values.append(avg_ear)
                self.head_angle_values.append(pitch_angle)
                
                # Calculate smoothed values
                smoothed_ear = sum(self.ear_values) / len(self.ear_values) if self.ear_values else avg_ear
                smoothed_pitch = sum(self.head_angle_values) / len(self.head_angle_values) if self.head_angle_values else pitch_angle
                
                # Store current values for GUI update
                self.current_ear = smoothed_ear
                self.current_head_angle = smoothed_pitch
                
                # Update drowsiness score based on indicators
                drowsiness_contributors = []
                
                # Check each drowsiness indicator and update score
                is_eyes_closed = smoothed_ear < self.EAR_THRESHOLD
                is_head_down = abs(smoothed_pitch) > self.HEAD_DOWN_ANGLE_THRESHOLD
                
                # Calculate score adjustments
                if is_eyes_closed:
                    self.drowsiness_score += self.EYE_CLOSED_SCORE_RATE
                    drowsiness_contributors.append("Eyes Closed")
                
                if is_head_down:
                    self.drowsiness_score += self.HEAD_DOWN_SCORE_RATE
                    drowsiness_contributors.append("Head Down")
                    
                # Apply score decay if no drowsiness indicators
                if not drowsiness_contributors:
                    self.drowsiness_score = max(0, self.drowsiness_score - self.SCORE_DECAY_RATE)
                    
                # Cap the maximum score
                self.drowsiness_score = min(self.drowsiness_score, self.MAX_DROWSINESS_SCORE)
                
                # Display metrics on frame
                cv2.putText(frame, f'EAR: {smoothed_ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f'Head Pitch: {smoothed_pitch:.1f}°', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Determine alert color based on drowsiness score
                if self.drowsiness_score < 30:
                    score_color = (0, 255, 0)  # Green
                elif self.drowsiness_score < self.ALERT_THRESHOLD:
                    score_color = (0, 255, 255)  # Yellow
                elif self.drowsiness_score < self.CRITICAL_THRESHOLD: 
                    score_color = (0, 165, 255)  # Orange
                else:
                    score_color = (0, 0, 255)  # Red
                    
                # Display drowsiness score with dynamic color
                cv2.putText(frame, f'Drowsiness: {int(self.drowsiness_score)}%', (w-220, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
                
                # Visual feedback for head position
                if abs(smoothed_pitch) > self.HEAD_DOWN_ANGLE_THRESHOLD:
                    cv2.putText(frame, "HEAD DOWN", (w-200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Determine current alert level
                new_alert_level = self.LEVEL_NORMAL
                alert_type = None
                alert_values = {}
                alert_message = ""
                
                if self.drowsiness_score >= self.CRITICAL_THRESHOLD:
                    new_alert_level = self.LEVEL_CRITICAL
                    alert_message = "CRITICAL DROWSINESS!"
                    alert_type = "CRITICAL_DROWSINESS"
                    alert_values = {
                        "Score": self.drowsiness_score,
                        "Factors": ", ".join(drowsiness_contributors),
                        "Head Pitch": f"{smoothed_pitch:.1f}°",
                        "EAR": f"{smoothed_ear:.2f}"
                    }
                elif self.drowsiness_score >= self.ALERT_THRESHOLD:
                    new_alert_level = self.LEVEL_ALERT
                    alert_message = "DANGER - DROWSY!"
                    alert_type = "HIGH_DROWSINESS"
                    alert_values = {
                        "Score": self.drowsiness_score,
                        "Factors": ", ".join(drowsiness_contributors),
                        "Head Pitch": f"{smoothed_pitch:.1f}°",
                        "EAR": f"{smoothed_ear:.2f}"
                    }
                elif self.drowsiness_score >= 30:
                    new_alert_level = self.LEVEL_WARNING
                    alert_message = "Warning - Drowsiness Detected"
                    
                # Check if we can trigger a new alert (based on time elapsed and level change)
                can_alert = (current_time - self.last_alert_time > self.REPEAT_ALERT_DELAY and 
                             new_alert_level > self.current_alert_level)
                    
                # Display alert message if score is high enough
                if new_alert_level >= self.LEVEL_WARNING:
                    message_color = score_color
                    font_size = 0.8 + (new_alert_level * 0.2)  # Bigger font for higher alert levels
                    cv2.putText(frame, alert_message, (int(w/2)-150, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_size, message_color, 2)
                
                # Handle alerting and logging
                if new_alert_level >= self.LEVEL_ALERT and can_alert:
                    self.play_buzzer_for_3_seconds()
                    
                    # Log the event
                    self.log_drowsiness_event(alert_type, alert_values)
                    
                    # Save screenshot
                    img_path = self.save_image(frame, alert_type)
                    
                    # Update alert counters
                    if new_alert_level == self.LEVEL_CRITICAL:
                        self.session_alerts["critical"] += 1
                    else:
                        self.session_alerts["alerts"] += 1
                    
                    # Update alert state
                    self.last_alert_time = current_time
                    self.current_alert_level = new_alert_level
                
                # Update warning counter if needed
                elif new_alert_level == self.LEVEL_WARNING and self.current_alert_level < self.LEVEL_WARNING:
                    self.session_alerts["warnings"] += 1
                    self.current_alert_level = new_alert_level
                
                # Draw eye contours
                cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 255), 1)
                cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 255), 1)
                
        else:
            # No face detected
            cv2.putText(frame, "No Face Detected", (int(frame.shape[1]/2)-100, int(frame.shape[0]/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    def update_metrics_display(self):
        """Update the metrics display in the GUI"""
        if hasattr(self, 'current_ear'):
            # Update EAR display
            ear_percent = min(self.current_ear / 0.4 * 100, 100)  # Normalize to 0-100%
            self.ear_bar['value'] = ear_percent
            self.ear_label.config(text=f"{self.current_ear:.2f}")
            
            # Update head angle display
            angle_percent = min(abs(self.current_head_angle) / 30 * 100, 100)  # Normalize to 0-100%
            self.head_bar['value'] = angle_percent
            self.head_label.config(text=f"{self.current_head_angle:.1f}°")
            
            # Update drowsiness score
            self.drowsiness_bar['value'] = self.drowsiness_score
            self.drowsiness_label.config(text=f"{int(self.drowsiness_score)}%")
            
            # Update status text and color
            if self.drowsiness_score < 30:
                self.status_label.config(text="ALERT", foreground="green")
            elif self.drowsiness_score < self.ALERT_THRESHOLD:
                self.status_label.config(text="WARNING", foreground="orange")
            elif self.drowsiness_score < self.CRITICAL_THRESHOLD:
                self.status_label.config(text="DROWSY", foreground="red")
            else:
                self.status_label.config(text="CRITICAL", foreground="red")
                
            # Update alert counters
            self.update_session_labels()
    
    def update_session_labels(self):
        """Update the session statistics labels"""
        self.warnings_label.config(text=str(self.session_alerts["warnings"]))
        self.alerts_label.config(text=str(self.session_alerts["alerts"]))
        self.critical_label.config(text=str(self.session_alerts["critical"]))
    
    def update_session_stats(self):
        """Update session duration"""
        if self.running:
            duration = datetime.now() - self.session_start_time
            # Format as HH:MM:SS
            hours, remainder = divmod(duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.duration_label.config(text=f"{hours:02}:{minutes:02}:{seconds:02}")
        
        # Schedule next update
        self.root.after(1000, self.update_session_stats)
    
    def display_frame(self, frame):
        """Display the opencv frame in the GUI"""
        try:
            # Resize frame to fit the canvas
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:  # Make sure canvas has been drawn
                # Calculate aspect ratio preserving resize
                h, w = frame.shape[:2]
                aspect_ratio = w / h
                
                if canvas_width / canvas_height > aspect_ratio:
                    # Canvas is wider than frame
                    new_height = canvas_height
                    new_width = int(new_height * aspect_ratio)
                else:
                    # Canvas is taller than frame
                    new_width = canvas_width
                    new_height = int(new_width / aspect_ratio)
                
                frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert frame to PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Update canvas
                self.video_canvas.config(width=new_width, height=new_height)
                self.video_canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=img_tk)
                self.video_canvas.image = img_tk  # Keep a reference
        except Exception as e:
            print(f"Error displaying frame: {e}")
    
    def eye_aspect_ratio(self, eye):
        """Calculate the eye aspect ratio"""
        # Compute the euclidean distances between the vertical eye landmarks
        A = self.euclidean_distance(eye[1], eye[5])
        B = self.euclidean_distance(eye[2], eye[4])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        C = self.euclidean_distance(eye[0], eye[3])
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def euclidean_distance(self, p1, p2):
        """Calculate the euclidean distance between two points"""
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    def calculate_head_angle(self, face_landmarks, frame_shape):
        """Calculate the head angle (pitch)"""
        h, w = frame_shape[:2]
        model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),   # Right eye right corner
            (-150.0, -150.0, -125.0), # Mouth left corner
            (150.0, -150.0, -125.0)   # Mouth right corner
        ], dtype=np.float64)
        
        image_points = np.array([
            (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),
            (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h),
            (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),
            (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h),
            (face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h),
            (face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h)
        ], dtype=np.float64)
    
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
    
        dist_coeffs = np.zeros((4,1))
        success, rotation_vec, translation_vec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        
        if not success:
            return 0
        
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pitch = math.degrees(math.asin(rotation_mat[2, 1]))
        return pitch
    
    def log_drowsiness_event(self, event_type, values):
        """Log drowsiness events to a file with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {event_type} - Values: {values}\n"
        
        with open(self.LOG_FILE, "a") as f:
            f.write(log_entry)
        
        print(f"Logged: {log_entry.strip()}")
    
    def save_image(self, frame, event_type):
        """Save an image of the detected drowsiness event"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(self.IMAGES_DIR, f"{event_type}_{timestamp}.jpg")
        cv2.imwrite(img_path, frame)
        return img_path
    
    def play_buzzer_for_3_seconds(self):
        """Play the alert sound for 3 seconds"""
        if self.BUZZER_SOUND_PATH and os.path.exists(self.BUZZER_SOUND_PATH):
            try:
                if self.buzzer_start_time is None:
                    pygame.mixer.music.load(self.BUZZER_SOUND_PATH)
                    pygame.mixer.music.play()
                    self.buzzer_start_time = time.time()
            except Exception as e:
                print(f"Error playing sound: {e}")
    
    def browse_sound_file(self):
        """Open file dialog to select alert sound file"""
        sound_file = filedialog.askopenfilename(
            title="Select Alert Sound",
            filetypes=[("MP3 files", "*.mp3"), ("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if sound_file:
            self.sound_path_var.set(sound_file)
    
    def save_settings(self):
        """Save the current settings to a file"""
        settings = {
            "EAR_THRESHOLD": self.ear_threshold_var.get(),
            "HEAD_DOWN_ANGLE_THRESHOLD": self.head_threshold_var.get(),
            "ALERT_THRESHOLD": self.alert_threshold_var.get(),
            "CRITICAL_THRESHOLD": self.critical_threshold_var.get(),
            "BUZZER_SOUND_PATH": self.sound_path_var.get(),
            "CAMERA_ID": self.camera_var.get()
        }
        
        try:
            with open(os.path.join(self.LOGS_DIR, "settings.txt"), "w") as f:
                for key, value in settings.items():
                    f.write(f"{key}={value}\n")
            
            # Update current settings
            self.EAR_THRESHOLD = settings["EAR_THRESHOLD"]
            self.HEAD_DOWN_ANGLE_THRESHOLD = settings["HEAD_DOWN_ANGLE_THRESHOLD"]
            self.ALERT_THRESHOLD = settings["ALERT_THRESHOLD"]
            self.CRITICAL_THRESHOLD = settings["CRITICAL_THRESHOLD"]
            self.BUZZER_SOUND_PATH = settings["BUZZER_SOUND_PATH"]
            
            messagebox.showinfo("Settings", "Settings saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
    
    def load_settings(self):
        """Load settings from file if it exists"""
        settings_file = os.path.join(self.LOGS_DIR, "settings.txt")
        if os.path.exists(settings_file):
            try:
                settings = {}
                with open(settings_file, "r") as f:
                    for line in f:
                        if "=" in line:
                            key, value = line.strip().split("=", 1)
                            settings[key] = value
                
                # Update GUI variables with loaded settings
                if "EAR_THRESHOLD" in settings:
                    self.ear_threshold_var.set(float(settings["EAR_THRESHOLD"]))
                    self.ear_threshold_label.config(text=f"{float(settings['EAR_THRESHOLD']):.2f}")
                    self.EAR_THRESHOLD = float(settings["EAR_THRESHOLD"])
                
                if "HEAD_DOWN_ANGLE_THRESHOLD" in settings:
                    self.head_threshold_var.set(float(settings["HEAD_DOWN_ANGLE_THRESHOLD"]))
                    self.head_threshold_label.config(text=f"{float(settings['HEAD_DOWN_ANGLE_THRESHOLD']):.1f}°")
                    self.HEAD_DOWN_ANGLE_THRESHOLD = float(settings["HEAD_DOWN_ANGLE_THRESHOLD"])
                
                if "ALERT_THRESHOLD" in settings:
                    self.alert_threshold_var.set(int(float(settings["ALERT_THRESHOLD"])))
                    self.alert_threshold_label.config(text=f"{int(float(settings['ALERT_THRESHOLD']))}%")
                    self.ALERT_THRESHOLD = int(float(settings["ALERT_THRESHOLD"]))
                
                if "CRITICAL_THRESHOLD" in settings:
                    self.critical_threshold_var.set(int(float(settings["CRITICAL_THRESHOLD"])))
                    self.critical_threshold_label.config(text=f"{int(float(settings['CRITICAL_THRESHOLD']))}%")
                    self.CRITICAL_THRESHOLD = int(float(settings["CRITICAL_THRESHOLD"]))
                
                if "BUZZER_SOUND_PATH" in settings:
                    self.sound_path_var.set(settings["BUZZER_SOUND_PATH"])
                    self.BUZZER_SOUND_PATH = settings["BUZZER_SOUND_PATH"]
                
                if "CAMERA_ID" in settings:
                    self.camera_var.set(settings["CAMERA_ID"])
                
                print("Settings loaded successfully.")
            except Exception as e:
                print(f"Error loading settings: {e}")
    
    def generate_report(self):
        """Generate a report of drowsiness events"""
        if not os.path.exists(self.LOG_FILE):
            messagebox.showinfo("Report", "No drowsiness data available to generate report.")
            return
        
        try:
            # Get current date for report filename
            today = datetime.now().strftime("%Y%m%d")
            report_file = os.path.join(self.REPORTS_DIR, f"drowsiness_report_{today}.csv")
            
            # Parse log file and extract events
            events = []
            with open(self.LOG_FILE, "r") as f:
                for line in f:
                    try:
                        parts = line.strip().split(" - ", 2)
                        if len(parts) >= 3:
                            timestamp, event_type, values_str = parts
                            
                            # Parse the values string (format: "Values: {'key': value, ...}")
                            values_dict = {}
                            if values_str.startswith("Values: {"):
                                values_text = values_str[8:].strip()  # Remove "Values: " prefix
                                
                                # Extract key-value pairs
                                items = values_text.strip("{}").split(", ")
                                for item in items:
                                    if ": " in item:
                                        k, v = item.split(": ", 1)
                                        k = k.strip("'\"")
                                        values_dict[k] = v.strip("'\"")
                            
                            events.append({
                                "Timestamp": timestamp,
                                "Event Type": event_type,
                                "Drowsiness Score": values_dict.get("Score", ""),
                                "Factors": values_dict.get("Factors", ""),
                                "EAR": values_dict.get("EAR", ""),
                                "Head Pitch": values_dict.get("Head Pitch", "")
                            })
                    except Exception as e:
                        print(f"Error parsing log line: {e}")
            
            # Write to CSV
            with open(report_file, "w", newline="") as csvfile:
                fieldnames = ["Timestamp", "Event Type", "Drowsiness Score", "Factors", "EAR", "Head Pitch"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for event in events:
                    writer.writerow(event)
            
            # Calculate session statistics
            session_duration = datetime.now() - self.session_start_time
            hours, remainder = divmod(session_duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{hours:02}:{minutes:02}:{seconds:02}"
            
            # Append session summary
            with open(report_file, "a", newline="") as csvfile:
                csvfile.write("\n\nSession Summary\n")
                csvfile.write(f"Start Time,{self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                csvfile.write(f"End Time,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                csvfile.write(f"Duration,{duration_str}\n")
                csvfile.write(f"Warning Alerts,{self.session_alerts['warnings']}\n")
                csvfile.write(f"Drowsiness Alerts,{self.session_alerts['alerts']}\n")
                csvfile.write(f"Critical Alerts,{self.session_alerts['critical']}\n")
            
            messagebox.showinfo("Report", f"Report generated successfully:\n{report_file}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.running:
            self.stop_detection()
        
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
            
        self.root.destroy()


if __name__ == "__main__":
    # Set up better looking theme
    try:
        # Try to use a modern theme if available
        from ttkthemes import ThemedTk
        root = ThemedTk(theme="arc")
    except ImportError:
        # If ttkthemes is not available, use regular Tk
        root = tk.Tk()
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
    
    app = DrowsinessDetectionApp(root)
    root.mainloop()