import cv2
import numpy as np
import pandas as pd
import os
import logging
import time
from head_pose_estimator import HeadPoseEstimator
from mediapipe_face_mesh_detector import MediapipeFaceMeshDetector

face_mesh_detector = MediapipeFaceMeshDetector()
head_pose_estimator = HeadPoseEstimator(vertical_threshold=0.5, horizontal_threshold=0.5)

def capture_baseline_head_pose(video_path):
    print("Capturing baseline head pose...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return None
    else:
        print("Video stream opened")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            cv2.imshow("Frame", frame)
            time.sleep(30)
    cap.release()
    cv2.destroyAllWindows()

VIDEO_PATH = "/Users/sarmkunatham/Desktop/1. Projects/GithubCodes/CU_Work/senior_project/exam_phase/data/raw/screen staring/right_upper_corner.mov"
poses = capture_baseline_head_pose(VIDEO_PATH)
print(f"Base line poses: {poses}")