import pickle
import cv2
import numpy as np
import os
import sys 
sys.path.append('../')
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        # Set the minimum distance to consider camera movement significant
        self.minimum_distance = 5

        # Define parameters for the Lucas-Kanade optical flow method
        self.lk_params = dict(
            winSize=(15, 15),  # Size of the search window
            maxLevel=2,  # Maximum number of pyramid levels
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Termination criteria
        )

        # Convert the input frame to grayscale
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Create a mask to specify regions of interest for feature detection
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1  # Allow features in the left 20 pixels
        mask_features[:, 900:1050] = 1  # Allow features in a region from columns 900 to 1050

        # Define parameters for good feature tracking
        self.features = dict(
            maxCorners=100,  # Maximum number of corners to return
            qualityLevel=0.3,  # Minimum quality level for corners
            minDistance=3,  # Minimum distance between detected corners
            blockSize=7,  # Size of the averaging block for computing covariance
            mask=mask_features  # Apply the mask to limit feature detection
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        # Adjust the positions of tracked objects based on camera movement
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    # Retrieve the original position of the tracked object
                    position = track_info['position']
                    # Get the camera movement for the current frame
                    camera_movement = camera_movement_per_frame[frame_num]
                    # Adjust the position by subtracting the camera movement
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    # Update the tracks dictionary with the adjusted position
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read camera movement data from a stub file if specified
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Initialize camera movement list for each frame
        camera_movement = [[0, 0]] * len(frames)

        # Convert the first frame to grayscale and detect features to track
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Iterate over frames to calculate camera movement
        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            # Measure distance between new and old features
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()  # Flatten the new feature point
                old_features_point = old.ravel()  # Flatten the old feature point

                # Calculate the distance between the new and old features
                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    # Calculate the x and y movement based on distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point) 
            
            # Update camera movement if significant movement is detected
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                # Detect new features in the current frame
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            # Update old_gray to the current frame's grayscale image
            old_gray = frame_gray.copy()
        
        # Save the camera movement data to a stub file if specified
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            # Create an overlay for displaying movement information
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)  # White rectangle for background
            alpha = 0.6  # Transparency for overlay
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # Blend overlay with frame

            # Get the camera movement for the current frame
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            # Add text to the frame showing camera movement
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame) 

        return output_frames
