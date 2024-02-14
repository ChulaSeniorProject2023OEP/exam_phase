import mediapipe as mp
import time
import cv2
import numpy as np
import pandas as pd
import pickle

class HeadPoseEstimator:
    """
    Class for estimating head pose using facial landmarks.

    Args:
        model_path (str): Path to the head pose estimation model (default: 'src/model/head_pose_model/model.pkl')

    Attributes:
        model (object): Head pose estimation model
        cols (list): List of feature column names
        threshold (float): Threshold for classifying the head pose

    Methods:
        extract_features: Extracts facial landmarks from an image
        normalize: Normalizes the extracted features
        draw_axes: Draws 3D axes on an image based on the estimated head pose
        predict: Predicts the head pose angles (pitch, yaw, roll) based on the normalized features
        run: Runs the head pose estimation in real-time using the computer's camera
    """

    def __init__(self, model_path='src/model/head_pose_model/model.pkl'):
        self.model = pickle.load(open(model_path, 'rb'))
        self.cols = []
        self.threshold = 0.3
        for pos in ['nose_', 'forehead_', 'left_eye_', 'mouth_left_', 'chin_', 'right_eye_', 'mouth_right_']:
            for dim in ('x', 'y'):
                self.cols.append(pos + dim)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
            )

    def extract_features(self, img):
        """
        Extracts facial landmarks from an image.

        Args:
            img (numpy.ndarray): Input image

        Returns:
            list: List of extracted facial landmarks
        """
        NOSE = 1
        FOREHEAD = 10
        LEFT_EYE = 33
        MOUTH_LEFT = 61
        CHIN = 199
        RIGHT_EYE = 263
        MOUTH_RIGHT = 291

        result = self.face_mesh.process(img)
        face_features = []
        
        if result.multi_face_landmarks is not None:
            for face_landmarks in result.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [FOREHEAD, NOSE, MOUTH_LEFT, MOUTH_RIGHT, CHIN, LEFT_EYE, RIGHT_EYE]:
                        face_features.append(lm.x)
                        face_features.append(lm.y)

        return face_features

    def normalize(self, poses_df):
        """
        Normalizes the extracted features.

        Args:
            poses_df (pandas.DataFrame): DataFrame containing the extracted features

        Returns:
            pandas.DataFrame: Normalized DataFrame
        """
        normalized_df = poses_df.copy()
        
        for dim in ['x', 'y']:
            for feature in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim, 'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
                normalized_df[feature] = poses_df[feature] - poses_df['nose_'+dim]
                
            diff = normalized_df['mouth_right_'+dim] - normalized_df['left_eye_'+dim]
            for feature in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim, 'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
                normalized_df[feature] = normalized_df[feature] / diff
        
        return normalized_df

    def draw_axes(self, img, pitch, yaw, roll, tx, ty, size=50):
        """
        Draws 3D axes on an image based on the estimated head pose.

        Args:
            img (numpy.ndarray): Input image
            pitch (float): Estimated pitch angle
            yaw (float): Estimated yaw angle
            roll (float): Estimated roll angle
            tx (float): X-coordinate of the nose landmark
            ty (float): Y-coordinate of the nose landmark
            size (int): Size of the axes (default: 50)

        Returns:
            numpy.ndarray: Image with 3D axes drawn
        """
        yaw = -yaw
        rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll]))[0].astype(np.float64)
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

    def predict(self, face_features_df):
        """
        Predicts the head pose angles (pitch, yaw, roll) based on the normalized features.

        Args:
            face_features_df (pandas.DataFrame): DataFrame containing the normalized features

        Returns:
            tuple: Predicted pitch, yaw, and roll angles
        """
        pitch_pred, yaw_pred, roll_pred = self.model.predict(face_features_df).ravel()
        return pitch_pred, yaw_pred, roll_pred

    def run(self):
        """
        Runs the head pose estimation in real-time using the computer's camera.
        """
        cap = cv2.VideoCapture(0)  # From Camera
        while(cap.isOpened()):
            ret, img = cap.read()
            start = time.time()
            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.flip(img, 1)
                img_h, img_w, img_c = img.shape
                text = ''
                
                face_features = self.extract_features(img)
                if len(face_features):
                    face_features_df = pd.DataFrame([face_features], columns=self.cols)
                    face_features_normalized = self.normalize(face_features_df)
                    pitch_pred, yaw_pred, roll_pred = self.predict(face_features_normalized)
                    nose_x = face_features_df['nose_x'].values * img_w
                    nose_y = face_features_df['nose_y'].values * img_h
                    img = self.draw_axes(img, pitch_pred, yaw_pred, roll_pred, nose_x, nose_y)
                                    
                    if pitch_pred > self.threshold:
                        text = 'Top'
                        if yaw_pred > self.threshold:
                            text = 'Top Left'
                        elif yaw_pred < -self.threshold:
                            text = 'Top Right'
                    elif pitch_pred < -self.threshold:
                        text = 'Bottom'
                        if yaw_pred > self.threshold:
                            text = 'Bottom Left'
                        elif yaw_pred < -self.threshold:
                            text = 'Bottom Right'
                    elif yaw_pred > self.threshold:
                        text = 'Left'
                    elif yaw_pred < -self.threshold:
                        text = 'Right'
                    else:
                        text = 'Forward'
                    cv2.putText(img, f'Pitch: {pitch_pred:.2f}', (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(img, f'Yaw: {yaw_pred:.2f}', (25, 175), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(img, f'Roll: {roll_pred:.2f}', (25, 225), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        
                cv2.putText(img, text, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                end = time.time()
                fps = 1 / (end - start)
                cv2.putText(img, f'FPS: {int(fps)}', (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                cv2.imshow('img', img)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
            else:
                break

        cv2.destroyAllWindows()
        cap.release()

if __name__ == "__main__":
    head_pose_estimator = HeadPoseEstimator()
    head_pose_estimator.run()
