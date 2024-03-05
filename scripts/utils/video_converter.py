import argparse
from moviepy.editor import VideoFileClip
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def convert_video_to_mp4(input_file_path, output_file_path):
    '''
    Converts a video file to MP4 format using the moviepy library.
    Args:
        input_file_path(str): The path to the input video file.
        output_file_path(str): The path to the output video file.
    '''
    try:
        clip = VideoFileClip(input_file_path)
        clip.write_videofile(output_file_path, codec='libx264', audio_codec='aac')
        logging.info(f'Converted {input_file_path} to {output_file_path}')
    except IOError as e:
        logging.error(f'Error occrured: {e}. Check if the input file exists.')
    except Exception as e:
        logging.error(f'Error occured: {e}')
        sys.exit(1)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Convert videos to MP4 format')
    parser.add_argument('--i', dest='input_dir', type=str, required=True, help='Directory containing input video files to convert.')
    parser.add_argument('--o', dest='output_dir', type=str, required=True, help='Directory to save converted MP4 videos.')
    
    args = parser.parse_args()
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Process each file in the input directory
    for root, dirs, files in os.walk(args.input_dir):
        for filename in files:
            # Filtering out specific file extension if neccessary
            if not filename.endswith('.txt') and not filename.endswith('.wav'):
                input_file_path = os.path.join(root, filename)
                
                # Create corresponding subfolder structure in the output directory
                relative_path = os.path.relpath(root, args.input_dir)
                output_dir_path = os.path.join(args.output_dir, relative_path)
                if not os.path.exists(output_dir_path):
                    os.makedirs(output_dir_path)
                
                output_file_name = os.path.splitext(filename)[0] + '.mp4'
                output_file_path = os.path.join(output_dir_path, output_file_name)
                
                convert_video_to_mp4(input_file_path, output_file_path)

if __name__ == '__main__':
    main()