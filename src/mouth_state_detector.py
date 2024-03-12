import cv2
import numpy as np
import logging
from constants import LOWER_LIP_INDEX, UPPER_LIP_INDEX
from mediapipe_face_mesh_detector import MediapipeFaceMeshDetector

class MouthStateDetection:
    """
    Class for detecting whether a person is talking based on mouth openness. 
    """

    def __init__(self, talking_threshold=0.01):
        """
        Initializes the MouthStateDetection object.

        Args:
            talking_threshold (float): The minimum range of mouth opening distances 
                                       required to classify as "talking".
        """
        self.talking_threshold = talking_threshold
        self.recent_mouth_openings = []  # More descriptive name

    def extract_mouth_state(self, mediapipe_result):
        """
        Extracts mouth state information from MediaPipe facial landmarks.

        Args:
            mediapipe_result: The output from MediaPipe's face mesh detector.

        Returns:
            dict: A dictionary containing:
                - classify (str): "talking" or "not talking"
                - mouth_opening_distance (float): The current mouth opening distance.
        """

        output = {
            "classify": "not talking",
            "mouth_opening_distance": 0
        }

        if mediapipe_result.multi_face_landmarks is not None:
            for face_landmarks in mediapipe_result.multi_face_landmarks:
                mouth_opening_distance = self.calculate_mouth_opening(face_landmarks.landmark)
                self.recent_mouth_openings.append(mouth_opening_distance)

                # Keep only the last 10 measurements
                if len(self.recent_mouth_openings) > 15:
                    self.recent_mouth_openings.pop(0)

                if self._is_talking():
                    output["classify"] = "talking"

                output["mouth_opening_distance"] = mouth_opening_distance

        return output

    def _is_talking(self):
        """
        Determines if a person is talking based on recent mouth opening measurements.

        Returns:
            bool: True if the person is classified as talking, False otherwise.
        """
        if len(self.recent_mouth_openings) < 15:
            return False

        mouth_opening_range = max(self.recent_mouth_openings) - min(self.recent_mouth_openings)
        return mouth_opening_range > self.talking_threshold

    def calculate_mouth_opening(self, landmarks):
        """
        Calculates the vertical distance between the upper and lower lips.

        Args:
            landmarks: A list of facial landmarks extracted by MediaPipe.  

        Returns:
            float: The vertical distance between the upper and lower lips.
        """
        upper_lip = landmarks[UPPER_LIP_INDEX]
        lower_lip = landmarks[LOWER_LIP_INDEX]
        return np.abs(upper_lip.y - lower_lip.y)

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    # Create a face mesh detector object
    face_mesh_detector = MediapipeFaceMeshDetector()

    cap = cv2.VideoCapture(0)

    # Initialize the mouth state detector
    mouth_state_detector = MouthStateDetection()

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if not success:
            logging.info("Error reading from camera, Exiting...")
            break
        
        frame = cv2.flip(frame, 1)

        # Detect the face landmarks using the face mesh detector
        mediapipe_result = face_mesh_detector.detect_landmarks(frame)

        # Extract the mouth state
        output = mouth_state_detector.extract_mouth_state(mediapipe_result)

        # Display the mouth state on the frame
        cv2.putText(frame, output["classify"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Mouth Opening Distance: {output["mouth_opening_distance"]:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Frame", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()