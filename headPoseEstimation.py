import cv2
import numpy as np
import mediapipe as mp
import time
from tensorflow.keras.models import load_model

# Load your cheating detection model
cheating_detection_model = load_model("BehaviorClassification_Unseen77%.h5")
# Buffer to store time steps
time_steps_buffer = []

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)


def compute_features_based_on_head_pose(image, face_mesh):
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    # Extract and flatten face landmarks
    keypoints = extract_face_keypoints(results)
    return keypoints


def extract_face_keypoints(results):
    if results.multi_face_landmarks:
        face = [[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark]
        return np.array(face).flatten()
    else:
        return np.zeros(468 * 3)  # Assuming 468 landmarks, each with x, y, z


if not cap.isOpened():
    print("Error: Camera not found or could not be opened.")
    exit()


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Error reading frame from camera.")
        continue
    # Compute the features for the current frame based on head pose
    current_features = compute_features_based_on_head_pose(image, face_mesh)
    time_steps_buffer.append(current_features)

    if len(time_steps_buffer) > 30:
        time_steps_buffer.pop(0)

    # Head Pose Estimation
    start = time.time()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.setflags(write=False)
    results = face_mesh.process(image)

    image.setflags(write=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if (
                    idx == 33
                    or idx == 263
                    or idx == 1
                    or idx == 61
                    or idx == 291
                    or idx == 199
                ):
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                x, y = int(lm.x * img_w), int(lm.y * img_h)

                # get the 2d coordinates
                face_2d.append([x, y])

                # get 3d coordinate
                face_3d.append([x, y, lm.z])
            # convert it to the numpy array
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array(
                [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
            )
            # distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix
            )

            # get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # see where the user's head tilting
            # if y < -5:
            #     text = "Looking Left"
            # elif y > 5:
            #     text = "Looking Right"
            # elif x < -3:
            #     text = "Looking Down"
            # elif x > 5:
            #     text = "Looking Up"
            # else:
            #     text = "Forward"

            # display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(
                nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix
            )

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Add text on the image
            # cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(
                image,
                "x: " + str(np.round(x, 2)),
                (500, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                image,
                "y: " + str(np.round(y, 2)),
                (500, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                image,
                "z: " + str(np.round(z, 2)),
                (500, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1,
            )

        end = time.time()
        total_time = end - start
        fps = 1 / total_time
        print("FPS: ", fps)

        cv2.putText(
            image,
            f"FPS: {int(fps)}",
            (20, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            2,
        )

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec,
        )

    # When buffer has 30 time steps, make a prediction
    if len(time_steps_buffer) == 30:
        model_input = np.array(time_steps_buffer).reshape(1, 30, 1404)
        cheating_prediction = cheating_detection_model.predict(model_input)

        # Correctly process the cheating_prediction
        print(f"cheating prediction results: ", cheating_prediction)
        predicted_class = np.argmax(cheating_prediction, axis=1)[
            0
        ]  # Get the predicted class index
        cheating_type = ""  # Initialize an empty string for cheating type

        if predicted_class == 1:
            cheating_text = "Cheating Type 1"
            cheating_type = "Type 1 (Looking)"
        # elif predicted_class == 2:
        #     cheating_text = "Cheating Type 2"
        #     cheating_type = "Type 2 (Talking)"
        else:
            cheating_text = "Not Cheating"
        # Display the cheating detection result
        cv2.putText(
            image, cheating_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        if cheating_type:
            cv2.putText(
                image,
                cheating_type,
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

    cv2.imshow("Head Pose Estimation", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()  # Close any OpenCV windows
