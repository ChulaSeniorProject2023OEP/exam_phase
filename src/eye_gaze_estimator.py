import cv2
import numpy as np
import logging
import pandas as pd

from mediapipe_face_mesh_detector import MediapipeFaceMeshDetector
from constants import LEFT_EYE_BOUNDARY_LANDMARKS, LEFT_EYE_IRIS_LANDMARKS, LEFT_EYE_OUTER_CORNER_INDEX, LEFT_IRIS_INDEX, NOSE_TIP_INDEX, RIGHT_EYE_BOUNDARY_LANDMARKS, RIGHT_EYE_IRIS_LANDMARKS, RIGHT_EYE_OUTER_CORNER_INDEX, RIGHT_IRIS_INDEX

class EyeGazeEstimator:
    """
    Class for estimating eye gaze direction using facial landmarks provided by MediaPipe.
    """
    def __init__(self, ratio_threshold=0.35):
        """
        Initializes the EyeGazeEstimator.

        Args:
            ratio_threshold: The threshold for determining gaze direction based on 
                                the distance between the iris and the eye corners.
        """
        self.ratio_threshold = ratio_threshold
        self.feature_columns = [f"{pos}{dim}" for pos in ['nose_', 'left_iris_', 'right_iris_', 'left_eye_outer_corner_', 'right_eye_outer_corner_'] 
                                for dim in ['x', 'y']]
    
    def extract_eye_features(self, mediapipe_result):
        """
        Extracts relevant eye landmark coordinates from MediaPipe results.

        Args:
            mediapipe_result: The output from MediaPipe's face mesh detector.

        Returns:
            A list of extracted eye landmark coordinates (x, y) or None if no faces are detected.
        """
        eye_features = []
        if mediapipe_result.multi_face_landmarks is not None:
            for face_landmarks in mediapipe_result.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [NOSE_TIP_INDEX, LEFT_IRIS_INDEX, RIGHT_IRIS_INDEX, LEFT_EYE_OUTER_CORNER_INDEX, RIGHT_EYE_OUTER_CORNER_INDEX]:
                        eye_features.append(lm.x)
                        eye_features.append(lm.y)
        return eye_features

    def normalize(self, eye_features_df):
        """
        Normalizes eye features to make them scale and translation invariant.

        Args:
            eye_features_df: A Pandas DataFrame containing eye features.

        Returns:
            A DataFrame containing the normalized eye features.
        """
        eye_features_normalized_df = eye_features_df.copy()
        for dim in ['x', 'y']:
            for feature in ['nose_'+dim, 'left_iris_'+dim, 'right_iris_'+dim, 'left_eye_outer_corner_'+dim, 'right_eye_outer_corner_'+dim]:
                eye_features_normalized_df[feature] = eye_features_df[feature] - eye_features_df['nose_'+dim]
            
            diff = (eye_features_normalized_df['right_eye_outer_corner_'+dim] - eye_features_normalized_df['left_eye_outer_corner_'+dim]) / 2
            for feature in ['nose_'+dim, 'left_iris_'+dim, 'right_iris_'+dim, 'left_eye_outer_corner_'+dim, 'right_eye_outer_corner_'+dim]:
                eye_features_normalized_df[feature] = eye_features_normalized_df[feature] / diff
        
        return eye_features_normalized_df
    
    def get_landmark_distance(self, landmark1, landmark2):
        """
        Calculates the Euclidean distance between two facial landmarks.

        Args:
            landmark1: The first landmark.
            landmark2: The second landmark.

        Returns:
            The Euclidean distance between the two landmarks.
        """
        dx = landmark1.x - landmark2.x
        dy = landmark1.y - landmark2.y
        return np.sqrt(dx * dx + dy * dy)
    
    def predict_eye_gaze(self, eye_features_normalized_df):
            '''
            Predicts the direction of eye gaze based on the normalized eye features.

            Parameters:
            eye_features_normalized_df (DataFrame): A DataFrame containing the normalized eye features.

            Returns:
            dict: A dictionary containing the eye gaze prediction and related information.

            '''
            classify_text = ''
            if eye_features_normalized_df is not None:
                left_iris_center_x = eye_features_normalized_df['left_iris_x'].values[0]
                right_iris_center_x = eye_features_normalized_df['right_iris_x'].values[0]
                left_eye_outer_corner_x = eye_features_normalized_df['left_eye_outer_corner_x'].values[0]
                right_eye_outer_corner_x = eye_features_normalized_df['right_eye_outer_corner_x'].values[0]
                left_diff = abs(left_iris_center_x - left_eye_outer_corner_x)
                right_diff = abs(right_iris_center_x - right_eye_outer_corner_x)
                
                if left_diff < self.ratio_threshold:
                    classify_text = 'Looking Left'
                elif right_diff < self.ratio_threshold:
                    classify_text = 'Looking Right'
                else:
                    classify_text = 'Looking Straight'
                
                output = {'left_diff': left_diff, 'right_diff': right_diff, 'left_iris_x': left_iris_center_x, 'right_iris_x': right_iris_center_x,'left_eye_outer_corner_x': left_eye_outer_corner_x, 'right_eye_outer_corner_x': right_eye_outer_corner_x, 'classify': classify_text}
                logging.info(f'Eye Gaze: {output}')
                return output
        
    def draw_iris_circle(self, img, iris_center_landmark, surrounding_landmarks):
        """
        Draws an iris circle on the given image.

        Parameters:
        img (numpy.ndarray): The input image.
        iris_center_landmark (Landmark): The landmark representing the center of the iris.
        surrounding_landmarks (list): A list of landmarks representing the surrounding points of the iris.

        Returns:
        numpy.ndarray: The image with the iris circle drawn on it.
        """
        
        center_x = int(iris_center_landmark.x * img.shape[1])
        center_y = int(iris_center_landmark.y * img.shape[0])
        center = (center_x, center_y)

        distances = [np.linalg.norm(np.array([landmark.x * img.shape[1], landmark.y * img.shape[0]]) - np.array(center))
                for landmark in surrounding_landmarks]
        radius = int(np.mean(distances))

        cv2.circle(img, center, radius, (255, 255, 0), 2)  # Yellow color for iris circle
        return img
    
    def draw_eye_landmarks(self, img, mediapipe_result):
        """
        Draws landmarks on the eyes of a face in an image.

        Args:
            img (numpy.ndarray): The input image.
            mediapipe_result (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): The mediapipe result containing the face landmarks.

        Returns:
            numpy.ndarray: The image with eye landmarks drawn.
        """
        if mediapipe_result.multi_face_landmarks is not None:
            for face_landmarks in mediapipe_result.multi_face_landmarks:
                for landmark in LEFT_EYE_BOUNDARY_LANDMARKS + LEFT_EYE_IRIS_LANDMARKS + RIGHT_EYE_BOUNDARY_LANDMARKS + RIGHT_EYE_IRIS_LANDMARKS:
                    cv2.circle(img, (int(face_landmarks.landmark[landmark].x * img.shape[1]), int(face_landmarks.landmark[landmark].y * img.shape[0])), 1, (0, 255, 0), -1)
                # Draw circles around the irises for additional visualization.
                img = self.draw_iris_circle(img, face_landmarks.landmark[468], [face_landmarks.landmark[i] for i in LEFT_EYE_IRIS_LANDMARKS])
                img = self.draw_iris_circle(img, face_landmarks.landmark[473], [face_landmarks.landmark[i] for i in RIGHT_EYE_IRIS_LANDMARKS])
        return img

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    eye_gaze_estimator = EyeGazeEstimator()
    face_mesh_detector = MediapipeFaceMeshDetector()
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            logging.error("Error reading from camera, Exiting...")
            break
        
        image = cv2.flip(image, 1)
        image_h, image_w, _ = image.shape
        
        mediapipe_result = face_mesh_detector.detect_landmarks(image)
        eye_features = eye_gaze_estimator.extract_eye_features(mediapipe_result)
        if len(eye_features):
            eye_features_df = pd.DataFrame([eye_features], columns=eye_gaze_estimator.feature_columns)
            eye_features_normalized_df = eye_gaze_estimator.normalize(eye_features_df)
            eye_gaze = eye_gaze_estimator.predict_eye_gaze(eye_features_normalized_df)
            eye_gaze_estimator.draw_eye_landmarks(image, mediapipe_result)

            cv2.putText(image, f"Left iris X: {eye_gaze['left_iris_x']:.4f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
            cv2.putText(image, f"Right iris X: {eye_gaze['right_iris_x']:.4f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
            cv2.putText(image, f"Left corner X: {eye_gaze['left_eye_outer_corner_x']:.4f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
            cv2.putText(image, f"Right corner X: {eye_gaze['right_eye_outer_corner_x']:.4f}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
            cv2.putText(image, f"Left diff: {eye_gaze['left_diff']:.4f}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
            cv2.putText(image, f"Right diff: {eye_gaze['right_diff']:.4f}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
            cv2.putText(image, f"Classify: {eye_gaze['classify']}", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 200), 2)
        
        cv2.imshow('Eye Gaze Estimation', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
