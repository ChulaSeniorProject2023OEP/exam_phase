from mediapipe_face_mesh_detector import MediapipeFaceMeshDetector
from eye_gaze_estimator import EyeGazeEstimator
from head_pose_estimator import HeadPoseEstimator
from mouth_state_detector import MouthStateDetection
import cv2
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
face_mesh_detector = MediapipeFaceMeshDetector()
eye_gaze_estimator = EyeGazeEstimator()
head_pose_estimator = HeadPoseEstimator()
mouth_state_detector = MouthStateDetection()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        logging.error("Error reading from camera, Exiting...")
        break
    
    image = cv2.flip(image, 1)
    image_h, image_w, _ = image.shape
    
    mediapipe_result = face_mesh_detector.detect_landmarks(image)
    
    # head pose estimation
    face_features = head_pose_estimator.extract_head_pose_features(mediapipe_result)
    if len(face_features):
        face_features_df = pd.DataFrame([face_features], columns=head_pose_estimator.feature_columns)
        face_features_normalized_df = head_pose_estimator.normalize(face_features_df)
        head_pose = head_pose_estimator.predict_head_pose(face_features_normalized_df)
        nose_x = face_features_df['nose_x'].values[0] * image_w
        nose_y = face_features_df['nose_y'].values[0] * image_h
        image = head_pose_estimator.draw_axes(image, head_pose, nose_x, nose_y)
        
        cv2.putText(image, f'Pitch: {head_pose["pitch"]:.2f}', (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, f'Yaw: {head_pose["yaw"]:.2f}', (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, f'Roll: {head_pose["roll"]:.2f}', (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, f'Classify: {head_pose["classify"]}', (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # eye gaze estimation
    eye_features = eye_gaze_estimator.extract_eye_features(mediapipe_result)
    if len(eye_features):
        eye_features_df = pd.DataFrame([eye_features], columns=eye_gaze_estimator.feature_columns)
        eye_features_normalized_df = eye_gaze_estimator.normalize(eye_features_df)
        eye_gaze = eye_gaze_estimator.predict_eye_gaze(eye_features_normalized_df)
        eye_gaze_estimator.draw_eye_landmarks(image, mediapipe_result)

        cv2.putText(image, f"Left iris X: {eye_gaze['left_iris_x']:.4f}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
        cv2.putText(image, f"Right iris X: {eye_gaze['right_iris_x']:.4f}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
        cv2.putText(image, f"Left corner X: {eye_gaze['left_eye_outer_corner_x']:.4f}", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
        cv2.putText(image, f"Right corner X: {eye_gaze['right_eye_outer_corner_x']:.4f}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
        cv2.putText(image, f"Left diff: {eye_gaze['left_diff']:.4f}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
        cv2.putText(image, f"Right diff: {eye_gaze['right_diff']:.4f}", (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
        cv2.putText(image, f"Classify: {eye_gaze['classify']}", (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 200), 2)
    
    # mouth state detection
    output = mouth_state_detector.extract_mouth_state(mediapipe_result)
    # Display the mouth state on the frame
    cv2.putText(image, output["classify"], (10, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f'Mouth Opening Distance: {output["mouth_opening_distance"]:.2f}', (10, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Misbehavior Detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()