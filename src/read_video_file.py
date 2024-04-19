import cv2

def get_video_frames(video_path):
    """
    Generator that yields video frames from the specified video file.
    
    Args:
        video_path (str): Path to the video file.
    
    Yields:
        frame (np.array): The next video frame in BGR format.
        
    Exits:
        Yields None when the video ends or cannot be opened.
    """
    # Create video capture object
    capture = cv2.VideoCapture(video_path)
    
    # Check that the video capture has been successfully opened
    if not capture.isOpened():
        print("Error opening video file")
        yield None

    # Yield each frame of the video
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        yield frame
    
    # Release the video capture object
    capture.release()

if __name__ == "__main__":
    # Example usage of the generator
    video_path = '/Users/sarmkunatham/Desktop/1. Projects/GithubCodes/CU_Work/senior_project/exam_phase/data/raw/screen staring/right_upper_corner.mov'
    for frame in get_video_frames(video_path):
        if frame is not None:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) == 27:  # Exit on pressing the ESC key
                break
        else:
            break

    cv2.destroyAllWindows()
