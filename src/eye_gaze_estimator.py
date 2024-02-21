import mediapipe as mp
import cv2
import numpy as np

# Assuming constants.py contains predefined lists of landmark indices for eye boundaries and irises.
from constants import (
    LEFT_EYE_BOUNDARY_LANDMARKS,
    LEFT_EYE_IRIS_LANDMARKS,
    RIGHT_EYE_BOUNDARY_LANDMARKS,
    RIGHT_EYE_IRIS_LANDMARKS,
)

class GazeEstimator:
    """
    A class for estimating the gaze direction using facial landmarks detected by MediaPipe.
    It determines if a person is looking straight, left, or right based on the position of the iris
    relative to the eye corners.
    """
    def __init__(self):
        """Initializes MediaPipe FaceMesh model with specific parameters for high accuracy."""
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def get_normalized_distance(self, landmark1, landmark2):
        """
        Calculates the Euclidean distance between two landmarks.

        Parameters:
            landmark1, landmark2: The two landmarks between which distance is to be calculated.

        Returns:
            The Euclidean distance between the two landmarks.
        """
        dx = landmark1.x - landmark2.x
        dy = landmark1.y - landmark2.y
        return np.sqrt(dx * dx + dy * dy)

    def detect_gaze_direction(self, img, face_landmarks):
        """
        Determines the gaze direction by analyzing the position of the iris within the eye.

        Parameters:
            img: The image frame from the video capture.
            face_landmarks: The facial landmarks detected by MediaPipe.

        Returns:
            A string indicating the detected behavior ("Looking Away" or "Focused").
        """
        # Extract landmarks for the iris centers and eye corners.
        left_iris_center = face_landmarks.landmark[468]
        right_iris_center = face_landmarks.landmark[473]
        left_eye_outer_corner = face_landmarks.landmark[33]  
        left_eye_inner_corner = face_landmarks.landmark[133]  
        right_eye_outer_corner = face_landmarks.landmark[263] 
        right_eye_inner_corner = face_landmarks.landmark[362] 

        # Calculate the horizontal ratio to determine gaze direction.
        left_eye_horizontal_ratio = self.get_normalized_distance(left_iris_center, left_eye_outer_corner) / \
            (self.get_normalized_distance(left_iris_center, left_eye_outer_corner) +
             self.get_normalized_distance(left_iris_center, left_eye_inner_corner))

        right_eye_horizontal_ratio = self.get_normalized_distance(right_iris_center, right_eye_outer_corner) / \
            (self.get_normalized_distance(right_iris_center, right_eye_outer_corner) +
             self.get_normalized_distance(right_iris_center, right_eye_inner_corner))

        # Annotate the image with the ratio for debugging and visualization.
        cv2.putText(img, f"Left eye horizontal ratio: {left_eye_horizontal_ratio:.2f}", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Right eye horizontal ratio: {right_eye_horizontal_ratio:.2f}", (10, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Determine gaze direction based on the ratio.
        left_gaze = "Straight" if np.isclose(left_eye_horizontal_ratio, 0.5, atol=0.1) else \
                    ("Left" if left_eye_horizontal_ratio < 0.5 else "Right")
        right_gaze = "Straight" if np.isclose(right_eye_horizontal_ratio, 0.5, atol=0.1) else \
                     ("Left" if right_eye_horizontal_ratio > 0.5 else "Right")

        behavior = "Looking Away" if left_gaze != right_gaze else "Focused"
        return behavior

    def draw_iris_circle(self, img, iris_center_landmark, surrounding_landmarks):
        """
        Draws a circle around the iris for visualization.

        Parameters:
            img: The image frame from the video capture.
            iris_center_landmark: The center landmark of the iris.
            surrounding_landmarks: Landmarks surrounding the iris to calculate the circle radius.

        Returns:
            The image with the iris circle drawn.
        """
        center_x = int(iris_center_landmark.x * img.shape[1])
        center_y = int(iris_center_landmark.y * img.shape[0])
        center = (center_x, center_y)

        distances = [np.linalg.norm(np.array([landmark.x * img.shape[1], landmark.y * img.shape[0]]) - np.array(center))
                    for landmark in surrounding_landmarks]
        radius = int(np.mean(distances))

        cv2.circle(img, center, radius, (255, 255, 0), 2)  # Yellow color for iris circle
        return img

    def visualize_eye_positions(self, img, face_landmarks):
        """
        Visualizes the positions of eye landmarks on the image.

        Parameters:
            img: The image frame from the video capture.
            face_landmarks: The facial landmarks detected by MediaPipe.
        """
        # Draw landmarks for eye boundaries and irises.
        for landmark in LEFT_EYE_BOUNDARY_LANDMARKS + LEFT_EYE_IRIS_LANDMARKS + RIGHT_EYE_BOUNDARY_LANDMARKS + RIGHT_EYE_IRIS_LANDMARKS:
            cv2.circle(img, (int(face_landmarks.landmark[landmark].x * img.shape[1]),
                             int(face_landmarks.landmark[landmark].y * img.shape[0])), 2, (0, 255, 0), -1)

        # Draw circles around the irises for additional visualization.
        img = self.draw_iris_circle(img, face_landmarks.landmark[468], [face_landmarks.landmark[i] for i in LEFT_EYE_IRIS_LANDMARKS])
        img = self.draw_iris_circle(img, face_landmarks.landmark[473], [face_landmarks.landmark[i] for i in RIGHT_EYE_IRIS_LANDMARKS])
        
        # Additional text for debugging and visualization.
        cv2.putText(img, f"Left eye iris: {face_landmarks.landmark[468].x:.2f}, {face_landmarks.landmark[468].y:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 3)
        cv2.putText(img, f"Right eye iris: {face_landmarks.landmark[473].x:.2f}, {face_landmarks.landmark[473].y:.2f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 3)
        return img

    def run(self):
        """
        Captures video from the default camera, processes each frame to detect facial landmarks,
        determines gaze behavior, and visualizes the results.
        """
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    behavior = self.detect_gaze_direction(frame, face_landmarks)
                    cv2.putText(frame, behavior, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 200), 2)
                    frame = self.visualize_eye_positions(frame, face_landmarks)

            cv2.imshow("Gaze Estimation", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    gaze_estimator = GazeEstimator()
    gaze_estimator.run()
