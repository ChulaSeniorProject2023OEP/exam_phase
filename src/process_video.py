import cv2
import numpy as np
import mediapipe as mp
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Create a directory to save the frames if it doesn't exist
frames_dir = "extracted_frames"
os.makedirs(frames_dir, exist_ok=True)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True
)


def extract_face_keypoints(results):
    if results.multi_face_landmarks:
        face = [[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark]
        return np.array(face).flatten()
        # return np.array(face)
    else:
        # return np.zeros(468 * 3)  # Assuming 468 landmarks, each with x, y, z
        return np.zeros(
            478 * 3
        )  # Assuming 468 landmarks, each with x, y, z for refine_landmarks=True
        # return np.zeros((468, 3))  # Assuming 468 landmarks, each with x, y, z


def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def extract_and_save_keypoints(video_path, video_idx, interval=30):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0

    while success:
        if count % interval == 0:
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            keypoints = extract_face_keypoints(results)
            npy_path = os.path.join(
                frames_dir, f"video_{video_idx}_keypoints_{count}.npy"
            )
            np.save(npy_path, keypoints)

        success, image = vidcap.read()
        count += 1

    vidcap.release()


# Example labels for each video (these should be assigned appropriately)
video_labels = ["0", "1", "2", "3", "5", "6"]

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(video_labels)

# Process videos and generate dataset
video_paths = [
    "video/not_cheating.mp4",
    "video/cheating_1.mp4",
    "video/cheating_2.mp4",
    "video/cheating_3.mp4",
    "video/cheating_5.mp4",
    "video/cheating_6.mp4",
]
dataset = []
keypoints_data = []
labels = []

for i, video_path in tqdm(enumerate(video_paths)):
    frame_count = get_frame_count(video_path)
    extract_and_save_keypoints(video_path, i, 30)

    for frame_num in range(0, frame_count, 30):  # Adjust for the interval
        npy_path = os.path.join(frames_dir, f"video_{i}_keypoints_{frame_num}.npy")
        print(f"Checking for file: {npy_path}")
        if os.path.exists(npy_path):
            keypoints = np.load(npy_path)
            keypoints_data.append(keypoints)
            labels.append(encoded_labels[i])
        else:
            print(f"File not found: {npy_path}")

print(f"Total number of keypoints arrays loaded: {len(keypoints_data)}")
