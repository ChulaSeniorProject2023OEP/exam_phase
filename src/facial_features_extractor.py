import cv2
import mediapipe as mp
import numpy as np
from constants import CHIN_INDEX, LEFT_MOUTH_CORNER_INDEX, LOWER_LIP_INDEX, NOSE_TIP_INDEX, LEFT_EYE_INDEX, RIGHT_EYE_INDEX, RIGHT_MOUTH_CORNER_INDEX, UPPER_LIP_INDEX
import time


class FacialFeatureExtractor:
    """
    FacialFeatureExtractor uses MediaPipe to extract facial features including
    head posture, irises, and mouth from video frames.

    Methods:
        process_frame(frame): Process a video frame, extract and visualize facial features.
        extract_head_posture(landmarks): Placeholder method to extract head posture.
        extract_irises(landmarks): Placeholder method to extract irises information.
        extract_mouth(landmarks): Placeholder method to extract mouth information.
    """

    def __init__(self):
        """
        Initializes the facial feature extractor by setting up MediaPipe Face Mesh.
        """
        # Initialize MediaPipe Face Mesh.
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # For drawing utility
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize params
        self.key_coordinates = [
            NOSE_TIP_INDEX, LEFT_EYE_INDEX, RIGHT_EYE_INDEX, LEFT_MOUTH_CORNER_INDEX, RIGHT_MOUTH_CORNER_INDEX, CHIN_INDEX
        ]
        self.img_h = 0
        self.img_w = 0
        self.face_3d = None
        self.face_2d = None
        self.numpy_face_3d = None
        self.numpy_face_2d = None
        self.nose_2d = (0, 0)
        self.nose_3d = (0, 0, 0)
        self.rotation_vector = None
        self.translation_vector = None
        self.camera_matrix = None
        self.distortion_matrix = None
        self.mouth_opening_distances = []
        self.TALKING_THRESHOLD = 0.02

    def process_frame(self, frame):
        """
        Processes a video frame, extracts facial landmarks, and visualizes the landmarks.

        Args:
            frame (numpy.ndarray): The video frame to process.

        Returns:
            numpy.ndarray: The video frame with facial landmarks visualized.
        """
        
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        self.img_h, self.img_w, _ = frame.shape
        
        # Improve performance
        frame.flags.writeable = False
        
        # Process the frame and get the face landmarks
        results = self.face_mesh.process(frame)
        
        # Improve performance
        frame.flags.writeable = True

        self.face_3d = []
        self.face_2d = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                # Draw face landmarks
                self.mp_drawing.draw_landmarks(frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS)

                # Extract features
                head_pose_estimation = self.extract_head_posture(face_landmarks.landmark)
                # irises = self.extract_irises(face_landmarks)
                mouth = self.extract_mouth(face_landmarks.landmark)
                
                # Process the extracted features as needed
                self.plot_head_posture(frame, head_pose_estimation)
                self.plot_mouth_status(frame, mouth)

        return frame

    def extract_head_posture(self, landmarks):
        """
        Extracts head posture information based on facial landmarks.

        Args:
            landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
                Facial landmarks detected by MediaPipe.

        Returns:
            tuple: Information about the head posture (pitch, yaw, roll).
        """
        face_2d = []
        face_3d = []
        for key_coordinate in self.key_coordinates:
            key_landmark = landmarks[key_coordinate]
            if key_coordinate == NOSE_TIP_INDEX:
                self.nose_2d = (key_landmark.x * self.img_w, key_landmark.y * self.img_h)
                self.nose_3d = (key_landmark.x * self.img_w, key_landmark.y * self.img_h, key_landmark.z * 3000)
            
            x, y = int(key_landmark.x * self.img_w), int(key_landmark.y * self.img_h)
            
            # get the 2d coordinates
            face_2d.append([x, y])
            
            # get 3d coordinate
            face_3d.append([x, y, key_landmark.z])
            
        # Convert it to the numpy array
        self.face_2d = np.array(face_2d, dtype=np.float64)
        self.face_3d = np.array(face_3d, dtype=np.float64)
        
        # Camera matrix
        focal_length = 1 * self.img_w  # Approximate focal length based on typical webcam focal length
        self.camera_matrix = np.array(
            [[focal_length, 0, self.img_h / 2], [0, focal_length, self.img_w / 2], [0, 0, 1]], dtype=np.float64
        )
        
        # The distortion coefficients
        self.distortion_matrix = np.zeros((4, 1), dtype=np.float64)
        
        # Calculate rotation and translation vectors using solvePnP
        success, self.rotation_vector, self.translation_vector = cv2.solvePnP(
            self.face_3d, self.face_2d, self.camera_matrix, self.distortion_matrix
        )
        
        # Get rotation matrix
        self.rotation_matrix, jacobian_matrix = cv2.Rodrigues(self.rotation_vector)
        
        # Calculate the Euler angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(self.rotation_matrix)
        
        # Get the y rotation degree
        x_angle = angles[0] * 360
        y_angle = angles[1] * 360
        z_angle = angles[2] * 360
        
        return x_angle, y_angle, z_angle
            
    def extract_irises(self, landmarks):
        """
        Extracts irises information based on facial landmarks.
        Placeholder method - requires actual implementation.

        Args:
            landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
                Facial landmarks detected by MediaPipe.

        Returns:
            str: Information about the irises (placeholder value).
        """
        # Implement irises extraction using landmarks
        # Placeholder: return dummy value
        return "irises_info"

    def extract_mouth(self, landmarks):
        """
        Extracts mouth information based on facial landmarks.
        Placeholder method - requires actual implementation.

        Args:
            landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
                Facial landmarks detected by MediaPipe.

        Returns:
            str: Information about the mouth (placeholder value).
        """
        if landmarks:
            mouth_opening_distance = self.calculate_mouth_opening(landmarks)
            self.mouth_opening_distances.append(mouth_opening_distance)
            
            # Keep only the last N measurements
            if (len(self.mouth_opening_distances) > 10):
                self.mouth_opening_distances.pop(0)
                
            if self.is_talking():
                return "talking"
            
        return "not talking"
    
    def calculate_mouth_opening(self, landmarks):
        # Assuming landmarks are indexed or named appropriately
        upper_lip = landmarks[UPPER_LIP_INDEX]
        lower_lip = landmarks[LOWER_LIP_INDEX]
        
        # Calculate vertical distance
        mouth_opening_distance = np.abs(upper_lip.y - lower_lip.y)
        return mouth_opening_distance
    
    def is_talking(self):
        if len(self.mouth_opening_distances) < 5:
            return False
        
        # Calculate the variance or range of mouth openings
        mouth_opening_range = max(self.mouth_opening_distances) - min(self.mouth_opening_distances)
        
        # Determine if the examinee is talking
        return mouth_opening_range > self.TALKING_THRESHOLD
    
    def plot_head_posture(self, frame, head_pose_estimation):
        """
        Plots the head posture (pitch, yaw, roll) on the video frame.

        Args:
            frame (numpy.ndarray): The video frame where the posture will be plotted.
            head_pose_estimation (tuple): The head posture information (pitch, yaw, roll).
        """
        x, y, z = head_pose_estimation
        
        # See where the user's head is tilting
        if y < -5:
            text = "Looking Left"
        elif y > 5:
            text = "Looking Right"
        elif x < -5:
            text = "Looking Down"
        elif x > 5:
            text = "Looking Up"
        else:
            text = "Forward"
            
        # Display the nose direction
        nose_3d_projection, jacobian = cv2.projectPoints(
            self.nose_3d, self.rotation_vector, self.translation_vector, self.camera_matrix, self.distortion_matrix
        )
        
        # Get nose points
        nose_p1 = (int(self.nose_2d[0]), int(self.nose_2d[1]))
        nose_p2 = (int(self.nose_2d[0] + y * 30), int(self.nose_2d[1] - x * 30))  # x and y are reversed,and scaled
        
        # Draw the line
        cv2.line(frame, nose_p1, nose_p2, (0, 255, 255), 3)
        
        # Add text on the image
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(frame, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(frame, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(frame, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    def plot_mouth_status(self, frame, mouth):
        """
        Plots the mouth status on the video frame.

        Args:
            frame (numpy.ndarray): The video frame where the mouth status will be plotted.
            mouth (str): The mouth status information.
        """
        # Add text on the image
        cv2.putText(frame, f"Mouth: {mouth}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)


# Main flow
if __name__ == "__main__":
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Initialize the facial feature extractor
    feature_extractor = FacialFeatureExtractor()

    while cap.isOpened():
        success, frame = cap.read()

        start = time.time()
        
        if not success:
            continue
        
        # Process the frame
        processed_frame = feature_extractor.process_frame(frame)

        end = time.time()
        total_time = end - start
        fps = 1 / total_time
        
        cv2.putText(
            processed_frame,
            f"FPS: {int(fps)}",
            (20, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            2,
        )
        
        # Show the processed frame
        cv2.imshow('Facial Feature Extractor', processed_frame)
        if cv2.waitKey(5) & 0xFF == 27:  # Exit on ESC
            break

    cap.release()
