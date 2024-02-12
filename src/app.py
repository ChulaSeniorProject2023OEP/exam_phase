import streamlit as st
import cv2
from facial_features_extractor import FacialFeatureExtractor

# Initialize the facial feature extractor
feature_extractor = FacialFeatureExtractor()

st.title("Real-time Facial Feature Extraction and Talking Detection")

# Placeholder for displaying the video frames
stframe = st.empty()

# Start capturing video from the first camera device
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video frame. Is your camera accessible?")
        break

    # Process the frame (implement your processing logic here)
    processed_frame = feature_extractor.process_frame(frame)

    # Convert the colors from BGR to RGB
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

    # Display the frame
    stframe.image(processed_frame)

    # Add a small delay to make the video display smoother
    cv2.waitKey(1)

cap.release()
