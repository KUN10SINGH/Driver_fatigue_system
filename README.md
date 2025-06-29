# Driver_fatigue_system 
1. Introduction
   
Driver fatigue is a leading cause of road accidents. Fatigue Detection utilizes
MediaPipe’s Face Mesh and vision-based indicators—Eye Aspect Ratio (EAR), yawning
frequency, and posture—to detect early signs of drowsiness and issue timely alerts.

2. Objectives

  Implement MediaPipe Face Mesh for real-time facial landmark detection,
  Compute EAR to monitor prolonged eye closure,
  Detects yawning via lip aperture analysis,
  Estimate gaze direction to assess driver attentiveness,
  Issue audio/visual alerts when fatigue indicators are met,

  
Current Features
1. Eye & Head Tracking
● Uses MediaPipe Face Mesh for precise facial landmark detection.
● Calculates:
○ Eye Aspect Ratio (EAR) to detect if eyes are closed.
○ Head pitch angle to determine if the driver’s head is tilted downward.
● Visual overlays show eye contours and alert messages on screen.
2. Drowsiness Scoring System
● Maintains a drowsiness score (0–100), which:
○ Increases if EAR is below threshold or head is down.
○ Decays gradually when signs of alertness return.
● Configurable scoring rates:
○ +1.5 points/frame for eye closure.
○ +1.2 points/frame for head down.
○ -0.5 points/frame decay when no signs are detected.
3. Multi-Level Alerts
● Alert thresholds:
○ Warning: score ≥ 30
○ Drowsy: score ≥ ALERT_THRESHOLD (default 70)
○ Critical: score ≥ CRITICAL_THRESHOLD (default 90)
● Triggers visual alerts and plays buzzer sounds.
● Limits alerts to avoid repetition within 8 seconds.
4. Customizable Settings
● GUI sliders allow real-time adjustment of:
○ EAR threshold
○ Head angle threshold
○ Alert and critical score levels
● Users can choose and test custom alert sounds.
● All settings are saved and loaded from a local file.
5. Session Metrics and Statistics
● Tracks and displays:
○ Session duration
○ Number of warning, alert, and critical events
● Visual indicators include progress bars for EAR, head pitch, and drowsiness
score.
● Status labels change color based on alert level.
6. Event Logging and Report Generation
● Logs all drowsiness events with:
○ Timestamp
○ EAR and head pitch values
○ Drowsiness score
○ Contributing factors
● Saves annotated screenshots of critical events.
● Generates CSV reports with:
○ Event logs
○ Session summary (duration, alerts count)
