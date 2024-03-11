import cv2
import numpy as np
import pandas as pd
import pickle
import logging

from constants import (CHIN_INDEX, DEFAULT_HEAD_POSE_MODEL_PATH, 
                        FOREHEAD_INDEX, LEFT_EYE_INDEX, LEFT_MOUTH_CORNER_INDEX, 
                        NOSE_TIP_INDEX, RIGHT_EYE_INDEX, RIGHT_MOUTH_CORNER_INDEX)
from mediapipe_face_mesh_detector import MediapipeFaceMeshDetector

class HeadPoseEstimator:
    '''
    A class to estimate head pose angles (pitch, yaw, roll) using facial landmarks identified in images. 
    Utilizes a pretrained model for the estimation.

    Attributes:
        model_path (str): Path to the serialized head pose estimation model.
        vertical_threshold (float): The vertical movement classification threshold.
        horizontal_threshold (float): The horizontal movement classification threshold.
        model (pickle.Pickle): Loaded model for head pose estimation.
        feature_columns (list): Names of the feature columns for the model.
    '''

    def __init__(self, model_path=DEFAULT_HEAD_POSE_MODEL_PATH, vertical_threshold=0.3, horizontal_threshold=0.5):
        '''
        Initialize the head pose estimator
        
        Args:
            model_path (str): Path to the trained head pose estimation model.
            vertical_threshold (float): Threshold for classifying vertical head movement (looking up/down).
            horizontal_threshold (float): Threshold for classifying horizontal head movement (looking left/right).
        '''
        self.model = pickle.load(open(model_path, 'rb'))
        self.vertical_threshold = vertical_threshold
        self.horizontal_threshold = horizontal_threshold
        self.feature_columns = [f"{pos}{dim}" for pos in ['nose_', 'forehead_', 'left_eye_', 'mouth_left_', 'chin_', 'right_eye_', 'mouth_right_']  
                                for dim in ['x', 'y']]
    
    def extract_head_pose_features(self, mediapipe_result):
        '''
        Extracts relevant facial landmark features for head pose estimation from a MediaPipe FaceMesh result.

        Parameters:
            mediapipe_result (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): 
                The facial landmarks detected by MediaPipe.

        Returns:
            list: Extracted facial landmark features relevant for head pose estimation.
        '''
        face_features = []
        if mediapipe_result.multi_face_landmarks is not None:
            for face_landmarks in mediapipe_result.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [FOREHEAD_INDEX, NOSE_TIP_INDEX, LEFT_MOUTH_CORNER_INDEX, RIGHT_MOUTH_CORNER_INDEX, CHIN_INDEX, LEFT_EYE_INDEX, RIGHT_EYE_INDEX]:
                        face_features.append(lm.x)
                        face_features.append(lm.y) 
        return face_features

    def normalize(self, face_features_df):
        '''
        Normalizes facial landmark features for head pose estimation.

        Parameters:
            face_features_df (pandas.DataFrame): DataFrame containing facial landmark features.

        Returns:
            pandas.DataFrame: Normalized feature DataFrame.
        '''
        
        face_features_normalized_df = face_features_df.copy()
        for dim in ['x', 'y']:
            for feature in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim, 'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
                face_features_normalized_df[feature] = face_features_df[feature] - face_features_df['nose_'+dim]
                
            diff = face_features_normalized_df['mouth_right_'+dim] - face_features_normalized_df['left_eye_'+dim]
            for feature in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim, 'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
                face_features_normalized_df[feature] = face_features_normalized_df[feature] / diff
        
        return face_features_normalized_df

    def predict_head_pose(self, face_features_normalized_df):
        '''
        Predicts head pose angles using the normalized facial features.

        Parameters:
            face_features_normalized_df (pandas.DataFrame): DataFrame of normalized facial features.

        Returns:
            dict: Predicted head pose angles (pitch, yaw, roll) and classification.
        '''
        pitch_pred, yaw_pred, roll_pred = self.model.predict(face_features_normalized_df).ravel()
        classify_text = self._classify_movement(pitch_pred, yaw_pred)
        output = {'pitch': pitch_pred, 'yaw': yaw_pred, 'roll': roll_pred, 'classify': classify_text}
        logging.info(f'Head Pose: {output}')
        return output
    
    def _classify_movement(self, pitch, yaw):
        '''
        Classifies head movement based on predicted pitch and yaw angles.
        
        Parameters:
            pitch (float): Predicted pitch angle.
            yaw (float): Predicted yaw angle.
        
        Returns:
            str: Classification of head movement.
        '''
        if pitch > self.vertical_threshold:
            return 'Top' + (' Left' if yaw > self.horizontal_threshold else ' Right' if yaw < -self.horizontal_threshold else '')
        elif pitch < -self.vertical_threshold:
            return 'Bottom' + (' Left' if yaw > self.horizontal_threshold else ' Right' if yaw < -self.horizontal_threshold else '')
        return 'Left' if yaw > self.horizontal_threshold else 'Right' if yaw < -self.horizontal_threshold else 'Forward'
    
    def draw_axes(self, img, output, tx, ty, size=50):
        '''
        Draws 3D axes on the image to visualize estimated head pose.

        Parameters:
            img (numpy.ndarray): The image to draw axes on.
            output (dict): Contains predicted pitch, yaw, and roll angles.
            tx (int): X-coordinate for the origin of the axes (typically nose_x).
            ty (int): Y-coordinate for the origin of the axes (typically nose_y).
            size (int): Length of the axes lines.
        
        Returns:
            numpy.ndarray: Image with 3D axes drawn.
        '''
        output['yaw'] = -output['yaw']
        rotation_matrix = cv2.Rodrigues(np.array([output['pitch'], output['yaw'], output['roll']]))[0].astype(np.float64)
        axes_points = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ], dtype=np.float64)
        axes_points = rotation_matrix @ axes_points
        axes_points = (axes_points[:2, :] * size).astype(int)
        axes_points[0, :] = axes_points[0, :] + tx
        axes_points[1, :] = axes_points[1, :] + ty
        
        new_img = img.copy()
        cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 0].ravel()), (255, 0, 0), 3)    
        cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 1].ravel()), (0, 255, 0), 3)    
        cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 2].ravel()), (0, 0, 255), 3)
        return new_img

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    head_pose_estimator = HeadPoseEstimator()
    face_mesh_detector = MediapipeFaceMeshDetector()
    
    cap = cv2.VideoCapture(0)  # From Camera
    while(cap.isOpened()):
        success, image = cap.read()
        if not success:
            logging.error("Error reading from camera, Exiting...")
            break
        
        image = cv2.flip(image, 1)
        image_h, image_w, _ = image.shape
        
        mediapipe_result = face_mesh_detector.detect_landmarks(image)
        face_features = head_pose_estimator.extract_head_pose_features(mediapipe_result)
        if len(face_features):
            face_features_df = pd.DataFrame([face_features], columns=head_pose_estimator.feature_columns)
            face_features_normalized_df = head_pose_estimator.normalize(face_features_df)
            head_pose = head_pose_estimator.predict_head_pose(face_features_normalized_df)
            nose_x = face_features_df['nose_x'].values[0] * image_w
            nose_y = face_features_df['nose_y'].values[0] * image_h
            image = head_pose_estimator.draw_axes(image, head_pose, nose_x, nose_y)
            
            cv2.putText(image, f'Pitch: {head_pose["pitch"]:.2f}', (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, f'Yaw: {head_pose["yaw"]:.2f}', (25, 175), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, f'Roll: {head_pose["roll"]:.2f}', (25, 225), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, f'Classify: {head_pose["classify"]}', (25, 275), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Head Pose Estimation', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
