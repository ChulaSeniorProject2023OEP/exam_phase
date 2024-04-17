import cv2
import pandas as pd
import numpy as np
import logging
import time
from datetime import timedelta
from mediapipe_face_mesh_detector import MediapipeFaceMeshDetector
from eye_gaze_estimator import EyeGazeEstimator
from head_pose_estimator import HeadPoseEstimator
from mouth_state_detector import MouthStateDetection

def initialize_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def initialize_estimators():
    return MediapipeFaceMeshDetector(), EyeGazeEstimator(), HeadPoseEstimator(), MouthStateDetection()

def process_video(video_file_path, estimators):
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        logging.error("Error opening video stream or file")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = int(fps)

    columns = ['time_MMSS', 'pitch', 'yaw', 'roll', 'classify_head_pose', 'left_iris_x', 'right_iris_x',
               'left_eye_outer_corner_x', 'right_eye_outer_corner_x', 'left_diff', 'right_diff',
               'classify_eye_gaze', 'mouth_classify', 'mouth_opening_distance', 'is_cheating']
    results = []

    start_time = time.time()
    frame_count = 0
    processed_frames = 0

    while True:
        success, image = cap.read()
        if not success:
            break

        if frame_count % skip_frames == 0:
            time_MMSS, estimations = process_frame(image, frame_count, fps, *estimators)
            # Convert the estimations dictionary to a list and prepend time_MMSS
            estimations_list = list(estimations.values())
            results.append(estimations_list)
            processed_frames += 1
            estimate_time_left(start_time, processed_frames, total_frames // skip_frames)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    results_df = pd.DataFrame(results, columns=columns)
    return results_df

def process_frame(image, frame_count, fps, face_mesh_detector, eye_gaze_estimator, head_pose_estimator, mouth_state_detector):
    # Initialize the cheating flag and calculate the time in MMSS format
    is_cheating = 0
    seconds = int(frame_count / fps)
    time_MMSS = f'{seconds // 60:02d}{seconds % 60:02d}'

    # Detect landmarks using the face mesh detector
    mediapipe_result = face_mesh_detector.detect_landmarks(image)

    # Initialize the estimations dictionary
    estimations = {'time_MMSS': time_MMSS}

    # Head pose estimation
    face_features = head_pose_estimator.extract_head_pose_features(mediapipe_result)
    if face_features:
        face_features_df = pd.DataFrame([face_features], columns=head_pose_estimator.feature_columns)
        face_features_normalized_df = head_pose_estimator.normalize(face_features_df)
        head_pose = head_pose_estimator.predict_head_pose(face_features_normalized_df)
        estimations.update({
            'pitch': head_pose["pitch"],
            'yaw': head_pose["yaw"],
            'roll': head_pose["roll"],
            'classify_head_pose': head_pose["classify"]
        })
        if head_pose["classify"] != 'Forward':
            is_cheating = 1

    # Eye gaze estimation
    eye_features = eye_gaze_estimator.extract_eye_features(mediapipe_result)
    if eye_features:
        eye_features_df = pd.DataFrame([eye_features], columns=eye_gaze_estimator.feature_columns)
        eye_features_normalized_df = eye_gaze_estimator.normalize(eye_features_df)
        eye_gaze = eye_gaze_estimator.predict_eye_gaze(eye_features_normalized_df)
        estimations.update({
            'left_iris_x': eye_gaze['left_iris_x'],
            'right_iris_x': eye_gaze['right_iris_x'],
            'left_eye_outer_corner_x': eye_gaze['left_eye_outer_corner_x'],
            'right_eye_outer_corner_x': eye_gaze['right_eye_outer_corner_x'],
            'left_diff': eye_gaze['left_diff'],
            'right_diff': eye_gaze['right_diff'],
            'classify_eye_gaze': eye_gaze['classify']
        })
        if eye_gaze['classify'] != 'Looking Straight':
            is_cheating = 1

    # Mouth state detection
    mouth_output = mouth_state_detector.extract_mouth_state(mediapipe_result)
    if mouth_output:
        estimations.update({
            'mouth_classify': mouth_output["classify"],
            'mouth_opening_distance': mouth_output["mouth_opening_distance"]
        })
        if mouth_output["classify"] != 'not talking':
            is_cheating = 1

    # Update the cheating flag in the estimations
    estimations['is_cheating'] = is_cheating

    return time_MMSS, estimations


def estimate_time_left(start_time, processed_frames, total_frames):
    elapsed_time = time.time() - start_time
    estimated_total_time = elapsed_time / processed_frames * total_frames
    remaining_time = estimated_total_time - elapsed_time
    logging.info(f"Estimated Time Left: {str(timedelta(seconds=remaining_time))}")

def save_results(df, file_path):
    df.to_csv(file_path, index=False)
    logging.info(f"Results saved to {file_path}")

if __name__ == "__main__":
    initialize_logging()
    estimators = initialize_estimators()
    video_file_path = '/Users/sarmkunatham/Desktop/1. Projects/GithubCodes/CU_Work/senior_project/exam_phase/data/processed/OEP_dataset/subject10/huangpi21.mp4'
    output_file_path = '/Users/sarmkunatham/Desktop/1. Projects/GithubCodes/CU_Work/senior_project/exam_phase/reports/output/subject10.csv'
    results_df = process_video(video_file_path, estimators)
    if results_df is not None:
        print(results_df.head())
        save_results(results_df, output_file_path)
