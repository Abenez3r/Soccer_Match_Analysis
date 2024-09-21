from utils import get_center_of_bbox, get_bbox_width, get_foot_position
from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import pickle
import os
import sys
import cv2

# Append the parent directory to the system path for module imports
sys.path.append('../')

""" A class to track players, referees, and the ball in soccer matches using YOLO for object detection and ByteTrack for tracking.
    Attributes:
        model_path (str): Path to the YOLO model for object detection.
        model (YOLO): Instance of the YOLO object detection model.
        tracker (sv.ByteTrack): Instance of the ByteTrack tracker. """
        
class Tracker:
    def __init__(self, model_path): #Initializes the Tracker with the specified YOLO model path. Parameters: model_path (str): The file path to the YOLO model.
        self.model = YOLO(model_path)
        # Initialize tracker to maintain unique IDs for detected objects, allowing for continuous tracking across frames.
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks): #Updates the tracks with the calculated positions of objects based on bounding box data. Parameters: tracks (dict): A dictionary containing tracked objects and their bounding boxes.
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    # Determine position based on object type (ball or player).
                    position = get_center_of_bbox(bbox) if object == 'ball' else get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions): #Fills in missing ball positions using interpolation. Parameters: ball_positions (list): A list of dictionaries with bounding box data for the ball. Returns: list: A list of ball positions with interpolated values.
        # Extract ball positions by track ID and convert to DataFrame for interpolation.
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values, filling edges as necessary.
        df_ball_positions = df_ball_positions.interpolate().bfill()
        # Reconstruct nested dictionary from interpolated DataFrame.
        return [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

    def detect_frames(self, frames): #Detects objects in the provided video frames using the YOLO model Parameters: frames (list): A list of video frames for processing. Returns: list: A list of detection results for each frame.
        batch_size = 20  # Process frames in batches for efficiency.
        detections = []

        # Process each batch of frames through the YOLO model.
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch  # Append current batch detections.

        return detections  # Return all detections.

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None): #Retrieves tracked objects in the frames, optionally using cached data from a stub file. Parameters: frames (list): A list of video frames for tracking. read_from_stub (bool): Whether to read existing tracks from a stub file. stub_path (str): Path to the stub file if reading from it. Returns: dict: A dictionary of tracked objects, including players, referees, and the ball.
        # Load tracking data from a stub file if specified.
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)

        # Initialize structure to hold tracking information.
        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert detections to a format usable for tracking.
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Correct class IDs for goalkeepers to players.
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Update object tracking with current detections./ Track objects using ByteTrack
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Initialize current frame's tracking dictionaries.
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                # Organize tracked objects by type.
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Track the ball detection.
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # Save tracking data if a stub path is provided.
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None): # Draws an ellipse around the bounding box on the frame. Parameters: frame (ndarray): The video frame to draw on. bbox (list): The bounding box coordinates [x1, y1, x2, y2]. color (tuple): The color of the ellipse (B, G, R). track_id (int, optional): The ID of the track for labeling. Defaults to None. Returns: ndarray: The modified frame with the drawn ellipse.
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Draw an ellipse to represent the object on the frame.
        cv2.ellipse(frame, center=(x_center, y2), axes=(int(width), int(0.35 * width)),
                    angle=0.0, startAngle=-45, endAngle=235, color=color, thickness=2, lineType=cv2.LINE_4)

        # Draw a rectangle for track ID display if provided.
        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15

            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame, f"{track_id}", (int(x1_text), int(y1_rect + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame

    def draw_triangle(self, frame, bbox, color): #   Draws a triangle above the bounding box to indicate ball possession. Parameters: frame (ndarray): The video frame to draw on. bbox (list): The bounding box coordinates [x1, y1, x2, y2]. color (tuple): The color of the triangle (B, G, R). Returns: ndarray: The modified frame with the drawn triangle.
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        # Define points for an upward-pointing triangle.
        triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])

        # Draw filled triangle and its outline on the frame.
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control): #Displays ball control statistics for both teams on the frame. Parameters: frame (ndarray): The video frame to annotate. frame_num (int): The current frame number for reference. team_ball_control (np.ndarray): An array indicating ball control for each frame. Returns: ndarray: The modified frame with ball control statistics drawn.    
        overlay = frame.copy()  # Create a semi-transparent overlay.
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)  # Blend overlay with original frame.

        # Calculate and display ball control percentages for each team.
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_frames = np.sum(team_ball_control_till_frame == 1)
        team_2_frames = np.sum(team_ball_control_till_frame == 2)
        total_frames = team_1_frames + team_2_frames

        if total_frames > 0:  # Prevent division by zero.
            team_1_percentage = team_1_frames / total_frames
            team_2_percentage = team_2_frames / total_frames

            cv2.putText(frame, f"Team 1 Ball Control: {team_1_percentage * 100:.2f}%",
                        (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, f"Team 2 Ball Control: {team_2_percentage * 100:.2f}%",
                        (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control): #Annotates video frames with player, referee, and ball information, as well as team ball control statistics. Parameters: video_frames (list): A list of video frames to annotate. tracks (dict): A dictionary of tracked objects including players, referees, and the ball. team_ball_control (np.ndarray): An array indicating ball control for each frame.  Returns: list: A list of annotated video frames.
        output_video_frames = []  # List to store annotated video frames.

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()  # Create a copy to avoid modifying the original frame.
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Annotate players on the frame
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))  # Default to red if no team color specified.
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get('has_ball', False):  # Draw triangle if player has the ball.
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Annotate referees on the frame
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Annotate the ball on the frame
            for track_id, ball in ball_dict.items():
                frame = self.draw_ellipse(frame, ball["bbox"], (255, 0, 0), track_id)

            # Draw team ball control statistics 
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)  # Append annotated frame to output list.

        return output_video_frames  # Return list of annotated frames.
