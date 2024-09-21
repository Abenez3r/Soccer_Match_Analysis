import numpy as np
import cv2

class ViewTransformer:
    def __init__(self): # Initializes the ViewTransformer with the court dimensions and sets up the perspective transformation matrix based on the pixel and target vertices.     
        court_width = 68  # Width of the football field (in meters)
        court_length = 23.32  # Length of the football field (in meters)

        # Vertices of the trapezoid (pixel coordinates) in the input video
        self.pixel_vertices = np.array([[110, 1035],  # Bottom-left corner in the image
                                        [265, 275],   # Top-left corner in the image
                                        [910, 260],   # Top-right corner in the image
                                        [1640, 915]]) # Bottom-right corner in the image

        # Vertices of the rectangle representing the court in real-world dimensions
        self.target_vertices = np.array([[0, court_width],  # Bottom-left corner in meters
                                         [0, 0],            # Top-left corner in meters
                                         [court_length, 0], # Top-right corner in meters
                                         [court_length, court_width]]) # Bottom-right corner in meters

        # Convert pixel_vertices and target_vertices to float32 for the transformation
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Calculate the perspective transformation matrix to map pixel_vertices to target_vertices
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point): # Transforms a point from the pixel space to the court space using the perspective transformation matrix. Returns None if the point is outside the defined polygon. Args: point (np.array): A point in pixel coordinates (x, y). Returns: np.array or None: The transformed point in court space (x, y) or None if the point is outside the court.
        # Convert the point to integer values for pixel coordinates
        p = (int(point[0]), int(point[1]))

        # Check if the point is inside the polygon defined by the pixel vertices
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0

        # If the point is outside the polygon, return None
        if not is_inside:
            return None

        # Reshape the point for perspective transformation (format required by OpenCV)
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)

        # Apply the perspective transformation to the point
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)

        # Return the transformed point, reshaped back to 2D coordinates
        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks): # Updates the provided tracks with the transformed positions. For each object and each track, it applies the perspective transformation and stores the transformed coordinates. Args: tracks (dict): A dictionary where the keys are object IDs, and the values are lists of tracks for each frame, containing position information. Modifies: tracks (dict): Updates the tracks with the transformed positions in court space.
        # Iterate through each object and its tracks
        for object_id, object_tracks in tracks.items():
            # Iterate through the frames in the object's tracks
            for frame_num, track in enumerate(object_tracks):
                # Iterate through each track entry in the current frame
                for track_id, track_info in track.items():
                    # Get the 'position_adjusted' value (position in pixel space)
                    position = track_info['position_adjusted']
                    
                    # Convert the position to a NumPy array for transformation
                    position = np.array(position)
                    
                    # Transform the position using the perspective transformer
                    position_transformed = self.transform_point(position)
                    
                    # If the position was successfully transformed, convert to list
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    
                    # Update the track with the transformed position
                    tracks[object_id][frame_num][track_id]['position_transformed'] = position_transformed
