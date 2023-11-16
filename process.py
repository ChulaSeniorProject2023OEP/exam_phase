import pandas as pd
import cv2
import mediapipe as mp
import numpy as np
import os

# Create a directory to save the frames if it doesn't exist
frames_dir = 'extracted_frames'
os.makedirs(frames_dir, exist_ok=True)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define threshold values for yaw and pitch
# yaw_threshold = 25   # degrees, adjust based on your requirement
# pitch_threshold = 15 # degrees, adjust based on your requirement

# Intervals (start_time, end_time, cheat_type)
cheating_intervals = [
    (135, 204, 1),
    (205, 220, 1),
    (221, 223, 2),
    (226, 244, 2),
    (252, 302, 2),
    (505, 517, 5),
    (519, 532, 5),
    (535, 536, 5),
    (544, 547, 5),
    (607, 649, 1),
    (1245, 1257, 1),
    (1316, 1339, 6),
    (324, 3350, 3),
    (930, 1013, 3),
    (1024, 1054, 3)


   
]
frame_rate = 10

# Convert time intervals to frame intervals
frame_intervals = [(int(start * frame_rate), int(end * frame_rate), cheat_type) for start, end, cheat_type in cheating_intervals]

def estimate_head_pose(frame):
    img_h, img_w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    # Define the 3D model points of a generic human face
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corne
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmarks for head pose estimation
            face_2d = np.array([
                (face_landmarks.landmark[1].x * img_w, face_landmarks.landmark[1].y * img_h),  # Nose tip
                (face_landmarks.landmark[152].x * img_w, face_landmarks.landmark[152].y * img_h),  # Chin
                (face_landmarks.landmark[226].x * img_w, face_landmarks.landmark[226].y * img_h),  # Left eye left corner
                (face_landmarks.landmark[446].x * img_w, face_landmarks.landmark[446].y * img_h),  # Right eye right corner
                (face_landmarks.landmark[57].x * img_w, face_landmarks.landmark[57].y * img_h),  # Left Mouth corner
                (face_landmarks.landmark[287].x * img_w, face_landmarks.landmark[287].y * img_h),  # Right mouth corner
            ], dtype="double")

            # Camera matrix and distortion coefficients
            focal_length = img_w
            center = (img_w / 2, img_h / 2)
            cam_matrix = np.array([[focal_length, 0, center[0]],
                                   [0, focal_length, center[1]],
                                   [0, 0, 1]], dtype="double")
            dist_matrix = np.zeros((4, 1), dtype="double")

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(model_points, face_2d, cam_matrix, dist_matrix)

            # Convert rotation vector to rotation matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Calculate Euler angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Convert to degrees
            x_angle = angles[0] * 360
            y_angle = angles[1] * 360
            z_angle = angles[2] * 360

            return {'yaw': y_angle, 'pitch': x_angle, 'roll': z_angle}

    return None

def extract_and_label_frames(video_path, frame_intervals, interval=1):
    frames = []
    labels = []
    vidcap = cv2.VideoCapture(video_path)
    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
    
    success, image = vidcap.read()
    frame_count = 0

    while success:
        # Label the frame based on cheating intervals
        label = 'not_cheating'
        for start_frame, end_frame, cheat_type in frame_intervals:
            if start_frame <= frame_count <= end_frame:
                label = f'cheating_type_{cheat_type}'
                break

        if frame_count % interval == 0:
            frames.append(image)
            labels.append(label)
        
        success, image = vidcap.read()
        frame_count += 1

    vidcap.release()
    return frames, labels

video_paths = ['Yousef1.mp4']
dataset = []
frame_id = 0

for video_path in video_paths:
    frames, labels = extract_and_label_frames(video_path, frame_intervals)
    for frame, label in zip(frames, labels):
        frame_path = os.path.join(frames_dir, f'frame_{frame_id}.jpg')
        cv2.imwrite(frame_path, frame)
        dataset.append((frame_path, label))
        frame_id += 1

df = pd.DataFrame(dataset, columns=['frame_path', 'label'])
df.to_csv('dataset.csv', index=False)
