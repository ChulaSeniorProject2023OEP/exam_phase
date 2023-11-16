import cv2
import numpy as np
import mediapipe as mp
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Initial threshold values
left_threshold = -5
right_threshold = 5

print("Press 'l' to set the current angle as left threshold, 'r' for right threshold. Press ESC to exit.")

# Load an image
image_path = 'path_to_your_image.jpg'  # Replace with your image path
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found.")
    exit()


# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Camera not found or could not be opened.")
#     exit()

# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         print("Error reading frame from camera.")
#         continue

# Get the dimensions of the image
img_h, img_w, img_c = image.shape

# Process the image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image.flags.writeable = False
results = face_mesh.process(image)
image.flags.writeable = True
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# start = time.time()
# image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

# image.setflags(write=False)
# results = face_mesh.process(image)

# image.setflags(write=True)
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# img_h, img_w, img_c = image.shape
# face_3d = []
# face_2d = []

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Prepare 2D and 3D points
        face_2d = []
        face_3d = []

        for idx, lm in enumerate(face_landmarks.landmark):
            x, y = int(lm.x * img_w), int(lm.y * img_h)

            # Select specific landmarks to infer the head pose
            # (You might need to adjust these landmarks to better suit your needs)
            if idx in [33, 263, 1, 61, 291, 199]:
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])  # Assuming Z coordinate as 0 for simplicity

        # Convert lists to numpy arrays
        face_2d = np.array(face_2d, dtype=np.float32)
        face_3d = np.array(face_3d, dtype=np.float32)

        # Camera matrix and distortion coefficients
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        cam_matrix = np.array([[focal_length, 0, center[0]],
                               [0, focal_length, center[1]],
                               [0, 0, 1]], dtype=np.float32)
        dist_matrix = np.zeros((4, 1), dtype=np.float32)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        # Convert rotation vector to rotation matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Calculate Euler angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Convert to degrees
        x_angle = angles[0] * 360
        y_angle = angles[1] * 360
        z_angle = angles[2] * 360

        # Determine the direction of the head tilt
        if y_angle < left_threshold:
            text = "Looking Left"
        elif y_angle > right_threshold:
            text = "Looking Right"
        elif x_angle < -2:
            text = "Looking Down"
        elif x_angle > 5:
            text = "Looking Up"
        else:
            text = "Forward"

        # Add text to the image
        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, "x: "+str(np.round(x_angle, 2)), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, "y: "+str(np.round(y_angle, 2)), (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, "z: "+str(np.round(z_angle, 2)), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    # end = time.time()
    # total_time = end - start
    # fps = 1/total_time
    #print("FPS: ", fps)

    # cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)



#     mp_drawing.draw_landmarks(
#         image = image,
#         landmark_list = face_landmarks,
#         connections = mp_face_mesh.FACEMESH_TESSELATION,
#         landmark_drawing_spec = drawing_spec,
#         connection_drawing_spec = drawing_spec
#     )
# cv2.imshow('Head Pose Estimation', image)

# Draw face mesh landmarks
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

# Display the image
cv2.imshow('Head Pose Estimation', image)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()

# key = cv2.waitKey(5)
# print(key)
# if key == ord('l'):
#     left_threshold = y
#     print(f"Left threshold set to {left_threshold}")
# elif key == ord('r'):
#     right_threshold = y
#     print(f"Right threshold set to {right_threshold}")
# elif key == 27:  # ESC key
#     break

# cap.release()