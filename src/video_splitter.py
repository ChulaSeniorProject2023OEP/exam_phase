import cv2
import os

def convert_pair_to_seconds(pair):
    if len(pair) != 2 or not all(len(time_str) == 4 and time_str.isdigit() for time_str in pair):
        raise ValueError("Each pair must contain two 'MMSS' format strings")

    return tuple(int(time_str[:2]) * 60 + int(time_str[2:]) for time_str in pair)

def convert_list_of_pairs(time_pairs):
    return [convert_pair_to_seconds(pair) for pair in time_pairs]

# Example usage
time_pairs = [
("0027", "0036"),
("0039", "0050"),
("0249", "0309"),
("0628", "0644"),
("0843", "0844"),
("0848", "0849"),
("0854", "0903"),
("0923", "0937"),
("1415", "1416"),
("1418", "1419")
]
time_spans = convert_list_of_pairs(time_pairs)
print(time_spans)  # Output will be in the format [(seconds1, seconds2), ...]


def split_video_at_specific_times(video_path, time_spans, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)

    for i, (start_time, end_time) in enumerate(time_spans):
        # Calculate the start and end frame numbers
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Set up the video writer for this segment
        segment_filename = os.path.join(output_folder, f'segment_{i}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(segment_filename, fourcc, fps, 
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        # Set the current frame to the start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Read and write frames for this segment
        while True:
            ret, frame = cap.read()
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if not ret or current_frame > end_frame:
                break  # Exit if we've reached the end of the segment or video
            out.write(frame)

        out.release()

    cap.release()
    print(f"Video split into {len(time_spans)} segments.")

# Example usage
video_path = '24mustaffa1.mp4'
output_folder = 'output_segments'
split_video_at_specific_times(video_path, time_spans, output_folder)
