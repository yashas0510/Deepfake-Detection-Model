import cv2
import os

def extract_frames(video_dir, output_dir, label, frame_rate=1):
    """
    Extracts frames from videos and saves them to the specified directory.
    
    Parameters:
    - video_dir: Path to the folder containing videos.
    - output_dir: Path to the output folder where frames will be saved.
    - label: Subfolder name (e.g., 'real' or 'manipulated').
    - frame_rate: Number of frames to extract per second.
    """
    # Create output directory if it doesn't exist
    output_path = os.path.join(output_dir, label)
    os.makedirs(output_path, exist_ok=True)
    
    # Loop through all video files in the directory
    for video_file in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_file)
        if not video_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue  # Skip non-video files
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_file}")
            continue
        
        frame_id = 0
        video_name = os.path.splitext(video_file)[0]
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frames at the specified frame rate
            fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second of the video
            if frame_id % (fps // frame_rate) == 0:
                frame_filename = os.path.join(output_path, f"{video_name}_frame{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_count += 1
            
            frame_id += 1
        
        cap.release()
        print(f"Extracted {frame_count} frames from {video_file} to {output_path}.")

# Paths for your video folders
real_videos_path = "C:/Users/yashas/Downloads/archive/DFD_original sequences"
manipulated_videos_path = "C:/Users/yashas/Downloads/archive/DFD_manipulated_sequences/go"
frames_output_path = "C:/Users/yashas/Downloads/archive/frames"

# Extract frames for real and manipulated videos
extract_frames(real_videos_path, frames_output_path, label="real", frame_rate=1)
extract_frames(manipulated_videos_path, frames_output_path, label="manipulated", frame_rate=1)

print("Frame extraction complete.")
