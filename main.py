from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():
    # Read Video: Load the input video from the specified file path.
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize the Tracker with the model for object detection/tracking.
    tracker = Tracker('models/best.pt')

    # Get object tracks from the video frames (either by generating or reading from a stub file).
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    # Add object positions to each track based on the video frames.
    tracker.add_position_to_tracks(tracks)

    # Initialize the camera movement estimator with the first frame of the video.
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    
    # Calculate camera movement across frames or load the precomputed data from a stub file.
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                               read_from_stub=True,
                                                                               stub_path='stubs/camera_movement_stub.pkl')

    # Adjust object positions based on the estimated camera movement to maintain consistency.
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Initialize the ViewTransformer to handle the view transformation of object positions.
    view_transformer = ViewTransformer()
    
    # Apply the view transformation to the object positions in the tracks.
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate missing ball positions to ensure smooth tracking of the ball across frames.
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Initialize the SpeedAndDistance_Estimator for calculating speed and distance of objects.
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    
    # Add speed and distance data to the object tracks.
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Initialize the TeamAssigner to detect and assign team colors to players.
    team_assigner = TeamAssigner()
    
    # Assign team colors based on the first frame and player positions.
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    # For each frame, loop through the player tracks and assign team colors to individual players.
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            # Determine the team for each player based on their bounding box and frame.
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            # Store the assigned team and team color in the track data.
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Initialize the PlayerBallAssigner to assign ball possession to players.
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    # For each frame, check which player has possession of the ball.
    for frame_num, player_track in enumerate(tracks['players']):
        # Get the bounding box of the ball in the current frame.
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        
        # Assign the ball to the closest player based on their position relative to the ball.
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            # If a player is assigned, mark that they have the ball and track their team's control.
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # If no player is assigned, the last team to control the ball retains control.
            team_ball_control.append(team_ball_control[-1])
    
    # Convert the team ball control list to a NumPy array for easier processing.
    team_ball_control = np.array(team_ball_control)

    # Draw object tracks, team colors, and ball possession on the video frames.
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Draw camera movement arrows or other annotations to indicate camera adjustments.
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Draw the speed and distance of players and objects on the video frames.
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save the annotated output video to the specified file path.
    save_video(output_video_frames, 'output_videos/output_video.avi')


# If the script is run directly (not imported), execute the main function.
if __name__ == '__main__':
    main()
