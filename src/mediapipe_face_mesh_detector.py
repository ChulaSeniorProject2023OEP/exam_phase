import mediapipe as mp
import cv2

class MediapipeFaceMeshDetector:
    """Handles facial landmark detection using MediaPipe Face Mesh."""

    def __init__(self, 
                max_num_faces=1, 
                refine_landmarks=True,
                min_detection_confidence=0.5, 
                min_tracking_confidence=0.5):
        """
        Initializes the facial landmark detector.

        Args:
            max_num_faces (int): Maximum number of faces to detect.
            refine_landmarks (bool): Whether to refine landmarks for more precise iris location.
            min_detection_confidence (float): Minimum confidence for face detection.
            min_tracking_confidence (float): Minimum confidence for facial landmark tracking.
        """

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def detect_landmarks(self, image):
        """Detects facial landmarks from an image.

        Args:
            image (numpy.ndarray): Input image (BGR format).

        Returns:
            mediapipe.framework.formats.face_mesh_pb2.FaceMesh: MediaPipe results object, or None if no face is detected.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)

        return results
    
    def draw_landmarks(self, image, results):
        """Draws facial landmarks on an image.

        Args:
            image (numpy.ndarray): Input image (BGR format).
            results (mediapipe.framework.formats.face_mesh_pb2.FaceMesh): MediaPipe results object.
        
        Returns:
            numpy.ndarray: Image with drawn facial landmarks.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for face_landmarks in results.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            landmark_drawing_spec=self.drawing_spec,
            connection_drawing_spec=self.mp_drawing_styles
            .get_default_face_mesh_iris_connections_style()
        )
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    face_mesh_detector = MediapipeFaceMeshDetector()
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        mediapipe_results = face_mesh_detector.detect_landmarks(image)
        if mediapipe_results.multi_face_landmarks is not None:
            image = face_mesh_detector.draw_landmarks(image, mediapipe_results)
        cv2.imshow('MediaPipe Face Mesh', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
