from mediapipe_face_mesh_detector import MediapipeFaceMeshDetector
from eye_gaze_estimator import EyeGazeEstimator
from head_pose_estimator import HeadPoseEstimator
from mouth_state_detector import MouthStateDetection
import cv2
import pandas as pd
import logging

# Initialize log configuration
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Initialize the estimators
face_mesh_detector = MediapipeFaceMeshDetector()
eye_gaze_estimator = EyeGazeEstimator()
head_pose_estimator = HeadPoseEstimator()
mouth_state_detector = MouthStateDetection()

# Load video from file instead of webcam
video_file_path = '/Users/sarmkunatham/Desktop/1. Projects/GithubCodes/CU_Work/senior_project/exam_phase/data/processed/OEP_dataset/subject1/XiYin1.mp4'
cap = cv2.VideoCapture(video_file_path)

# Check if video opened successfully
if not cap.isOpened():
    logging.error("Error opening video stream or file")
    exit()

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
# Calculate the number of frames to skip based on fps to process one frame per second
skip_frames = int(fps)

# Prepare DataFrame to store results
columns = ['time_MMSS', 'pitch', 'yaw', 'roll', 'classify_head_pose', 'left_iris_x', 'right_iris_x', 'left_eye_outer_corner_x', 'right_eye_outer_corner_x', 'left_diff', 'right_diff', 'classify_eye_gaze', 'mouth_classify', 'mouth_opening_distance']
results_df = pd.DataFrame(columns=columns)

frame_count = 0
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    # Only process frame at one-second intervals
    if frame_count % skip_frames == 0:
        # is cheating flag
        is_cheating = 0
        # Calculate the timestamp in MMSS format
        seconds = int(frame_count / fps)
        time_MMSS = f'{seconds // 60:02d}{seconds % 60:02d}'

        mediapipe_result = face_mesh_detector.detect_landmarks(image)

        # Collect estimations
        estimations = {'time_MMSS': time_MMSS}
        
        # Head pose estimation
        face_features = head_pose_estimator.extract_head_pose_features(mediapipe_result)
        if len(face_features):
            face_features_df = pd.DataFrame([face_features], columns=head_pose_estimator.feature_columns)
            face_features_normalized_df = head_pose_estimator.normalize(face_features_df)
            head_pose = head_pose_estimator.predict_head_pose(face_features_normalized_df)
            estimations.update({'pitch': head_pose["pitch"], 'yaw': head_pose["yaw"], 'roll': head_pose["roll"], 'classify_head_pose': head_pose["classify"]})
            if head_pose["classify"] != 'Forward':
                is_cheating = 1

        # Eye gaze estimation
        eye_features = eye_gaze_estimator.extract_eye_features(mediapipe_result)
        if len(eye_features):
            eye_features_df = pd.DataFrame([eye_features], columns=eye_gaze_estimator.feature_columns)
            eye_features_normalized_df = eye_gaze_estimator.normalize(eye_features_df)
            eye_gaze = eye_gaze_estimator.predict_eye_gaze(eye_features_normalized_df)
            estimations.update({'left_iris_x': eye_gaze['left_iris_x'], 'right_iris_x': eye_gaze['right_iris_x'], 'left_eye_outer_corner_x': eye_gaze['left_eye_outer_corner_x'], 'right_eye_outer_corner_x': eye_gaze['right_eye_outer_corner_x'], 'left_diff': eye_gaze['left_diff'], 'right_diff': eye_gaze['right_diff'], 'classify_eye_gaze': eye_gaze['classify']})
            if eye_gaze['classify'] != 'Looking Straight':
                is_cheating = 1

        # Mouth state detection
        mouth_output = mouth_state_detector.extract_mouth_state(mediapipe_result)
        estimations.update({'mouth_classify': mouth_output["classify"], 'mouth_opening_distance': mouth_output["mouth_opening_distance"]})
        if mouth_output["classify"] != 'not talking':
            is_cheating = 1

        # Add is_cheating flag to the DataFrame
        estimations.update({'is_cheating': is_cheating})
        
        # Append the collected estimations for the current frame to the DataFrame
        results_df = results_df.append(estimations, ignore_index=True)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Optionally, save the DataFrame to a CSV file
results_df.to_csv('/Users/sarmkunatham/Desktop/1. Projects/GithubCodes/CU_Work/senior_project/exam_phase/reports/estimations/subject1.csv', index=False)
print(results_df)