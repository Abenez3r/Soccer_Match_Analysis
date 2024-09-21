import sys 
# Adding the parent directory to the system path to import utils
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        # Setting maximum distance for player to ball assignment as 70 pixels
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self, players, ball_bbox):
        # Getting the center position of the ball bounding box
        ball_position = get_center_of_bbox(ball_bbox)

        # Initializing minimum distance to a large value
        minimum_distance = 99999
        assigned_player = -1  # Default value indicating no player assigned

        # Looping over each player to find the closest one to the ball
        for player_id, player in players.items():
            player_bbox = player['bbox']

            # Calculating distance from the player's left and right edges to the ball position
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            
            # Real distance is the minimum of the two distances calculated
            distance = min(distance_left, distance_right)

            # Checking if the distance is within the maximum allowed distance
            if distance < self.max_player_ball_distance:
                # Updating the closest player if a shorter distance is found
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player  # Returning the ID of the assigned player
