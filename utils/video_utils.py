import cv2

def read_video(video_path):
    # Open the video file using OpenCV's VideoCapture
    cap = cv2.VideoCapture(video_path)
    
    # Initialize a list to store the frames of the video
    frames = []
    
    # Loop through each frame in the video
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        
        # If no more frames, break the loop
        if not ret:
            break
        
        # Append the frame to the list of frames
        frames.append(frame)
    
    # Return the list of frames
    return frames

def save_video(output_video_frames, output_video_path):
    # Define the codec and create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Get the frame size from the first frame (width, height)
    frame_size = (output_video_frames[0].shape[1], output_video_frames[0].shape[0])
    
    # Create the VideoWriter object with the output path, codec, frame rate (24 fps), and frame size
    out = cv2.VideoWriter(output_video_path, fourcc, 24, frame_size)
    
    # Loop through each frame in the output frames and write it to the video file
    for frame in output_video_frames:
        out.write(frame)
    
    # Release the VideoWriter object after finishing the video writing
    out.release()
