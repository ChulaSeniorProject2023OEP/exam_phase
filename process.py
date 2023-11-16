import cv2
import numpy as np
import mediapipe as mp
import os
import pandas as pd

# Create a directory to save the frames if it doesn't exist
frames_dir = 'extracted_frames'
os.makedirs(frames_dir, exist_ok=True)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define threshold values for yaw and pitch
yaw_threshold = 30   # degrees
pitch_threshold = 15 # degrees

def estimate_head_pose(frame, img_w, img_h):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_2d = [(face_landmarks.landmark[i].x * img_w, face_landmarks.landmark[i].y * img_h) for i in [1, 152, 226, 446, 57, 287]]
            face_3d = [(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)]

            # Convert lists to numpy arrays and reshape
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64).reshape(-1, 3)  # Reshape to Nx3

            cam_matrix = np.array([[img_w, 0, img_w/2], [0, img_w, img_h/2], [0, 0, 1]], dtype='double')
            dist_matrix = np.zeros((4, 1), dtype='double')

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x_angle = np.degrees(angles[0])
            y_angle = np.degrees(angles[1])
            z_angle = np.degrees(angles[2])

            return {'yaw': y_angle, 'pitch': x_angle, 'roll': z_angle}
    
    return None


def extract_and_label_frames(video_path, interval=30, display_video=False):
    frames = []
    labels = []
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0

    while success:
        if count % interval == 0:
            img_h, img_w, _ = image.shape
            head_pose = estimate_head_pose(image, img_w, img_h)
            if head_pose:
                # Display video with head pose angles if enabled
                if display_video:
                    cv2.putText(image, f"Yaw: {head_pose['yaw']:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f"Pitch: {head_pose['pitch']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f"Roll: {head_pose['roll']:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("Video", image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
                        break

                frames.append(image)
                is_cheating = (head_pose['yaw'] < -1439 or head_pose['yaw'] > 2000) or (head_pose['pitch'] < -9000 or head_pose['pitch'] > -7000)
                label = 'cheating' if is_cheating else 'not_cheating'
                labels.append(label)

        success, image = vidcap.read()
        count += 1

    vidcap.release()
    cv2.destroyAllWindows()
    return frames, labels

# Process videos and generate dataset
video_paths = ['Yousef1.mp4','Jourabloo1.mp4','XiYin1.mp4','meowseph1.mp4']
dataset = []
frame_id = 0

for video_path in video_paths:
    frames, labels = extract_and_label_frames(video_path, display_video=True)
    for frame, label in zip(frames, labels):
        frame_path = os.path.join(frames_dir, f'frame_{frame_id}.jpg')
        cv2.imwrite(frame_path, frame)
        dataset.append((frame_path, label))
        frame_id += 1

# Convert to DataFrame and save
df = pd.DataFrame(dataset, columns=['frame_path', 'label'])
df.to_csv('dataset.csv', index=False)
