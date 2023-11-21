import cv2
import numpy as np
import mediapipe as mp
import os
import pandas as pd

# Create a directory to save the frames if it doesn't exist
frames_dir = "extracted_frames"
os.makedirs(frames_dir, exist_ok=True)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Define threshold values for yaw and pitch
yaw_threshold = 30  # degrees
pitch_threshold = 15  # degrees


def normalize_angle(angle):
    # Normalize angles to the range [0, 360] degrees
    return angle % 360


def estimate_head_pose(frame, img_w, img_h):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
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
            # face_2d = [
            #     (
            #         face_landmarks.landmark[i].x * img_w,
            #         face_landmarks.landmark[i].y * img_h,
            #     )
            #     # for i in [1, 152, 226, 446, 57, 287]
            #     for i in [33, 263, 1, 61, 291, 199]
            # ]
            # face_3d = [
            #     (0.0, 0.0, 0.0),
            #     (0.0, -330.0, -65.0),
            #     (-225.0, 170.0, -135.0),
            #     (225.0, 170.0, -135.0),
            #     (-150.0, -150.0, -125.0),
            #     (150.0, -150.0, -125.0),
            # ]
            # # face_3d = [[face_landmarks.landmark[i].x * img_w, face_landmarks.landmark[i].y * img_h,]]

            # # Convert lists to numpy arrays and reshape
            # face_2d = np.array(face_2d, dtype=np.float64)
            # face_3d = np.array(face_3d, dtype=np.float64).reshape(
            #     -1, 3
            # )  # Reshape to Nx3
            focal_length = 1 * img_w

            cam_matrix = np.array(
                [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
            )
            # cam_matrix = np.array(
            #     [[img_w, 0, img_h / 2], [0, img_w, img_w / 2], [0, 0, 1]],
            #     dtype=np.float64,
            # )
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix
            )
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # x_angle = normalize_angle(np.degrees(angles[0]))
            # y_angle = normalize_angle(np.degrees(angles[1]))
            # z_angle = normalize_angle(np.degrees(angles[2]))
            x_angle = angles[0] * 360
            y_angle = angles[1] * 360
            z_angle = angles[2] * 360

            # return {"yaw": y_angle, "pitch": x_angle, "roll": z_angle}
            return {"y": y_angle, "x": x_angle, "z": z_angle}

    return None


def extract_and_label_frames(video_path, interval=30, display_video=False):
    frames = []
    labels = []
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0

    while success:
        if count % interval == 0:
            img_h, img_w, _ = image.shape
            head_pose = estimate_head_pose(image, img_w, img_h)
            if head_pose:
                # Display video with head pose angles if enabled
                if display_video:
                    cv2.putText(
                        image,
                        f"y: {head_pose['y']:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        image,
                        f"x: {head_pose['x']:.2f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        image,
                        f"z: {head_pose['z']:.2f}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("Video", image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):  # Exit if 'q' is pressed
                        break

                frames.append(image)
                is_cheating = (head_pose["y"] < -1439 or head_pose["y"] > 2000) or (
                    head_pose["x"] < -9000 or head_pose["x"] > -7000
                )
                label = "cheating" if is_cheating else "not_cheating"
                labels.append(label)

        success, image = vidcap.read()
        count += 1

    vidcap.release()
    cv2.destroyAllWindows()
    return frames, labels


# Process videos and generate dataset
video_paths = ["Yousef1.mp4", "Jourabloo1.mp4", "XiYin1.mp4", "meowseph1.mp4"]
dataset = []
frame_id = 0

for video_path in video_paths:
    frames, labels = extract_and_label_frames(video_path, display_video=True)
    for frame, label in zip(frames, labels):
        frame_path = os.path.join(frames_dir, f"frame_{frame_id}.jpg")
        cv2.imwrite(frame_path, frame)
        dataset.append((frame_path, label))
        frame_id += 1

# Convert to DataFrame and save
df = pd.DataFrame(dataset, columns=["frame_path", "label"])
df.to_csv("dataset.csv", index=False)
