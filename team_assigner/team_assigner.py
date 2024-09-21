from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}  # Dictionary to store team colors
        self.player_team_dict = {}  # Dictionary to associate players with their team IDs

    def get_clustering_model(self, image):
        # Reshape the image to a 2D array for clustering
        image_2d = image.reshape(-1, 3)

        # Perform K-means clustering with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)  # Fit the model to the image data

        return kmeans  # Return the trained K-means model

    def get_player_color(self, frame, bbox):
        # Crop the image based on the bounding box
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Extract the top half of the cropped image for color analysis
        top_half_image = image[0:int(image.shape[0] / 2), :]

        # Get the clustering model for the top half image
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel in the top half image
        labels = kmeans.labels_

        # Reshape the labels to match the shape of the top half image
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Identify the player cluster based on the corners of the clustered image
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1],
                           clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster  # Determine the player cluster

        # Get the color of the player from the K-means cluster centers
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color  # Return the detected player color

    def assign_team_color(self, frame, player_detections):
        player_colors = []  # List to store detected player colors

        # Loop through each detected player
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]  # Get the bounding box for the player
            player_color = self.get_player_color(frame, bbox)  # Get the player's color
            player_colors.append(player_color)  # Add the color to the list

        # Perform K-means clustering on the collected player colors
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)  # Fit the K-means model to the player colors

        self.kmeans = kmeans  # Save the trained K-means model

        # Store the colors for each team based on the cluster centers
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        # Check if the player's team ID is already known
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]  # Return the known team ID

        player_color = self.get_player_color(frame, player_bbox)  # Get the player's color

        # Predict the team ID based on the player's color
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  # Increment to match team ID indexing

        # Special case for goalkeeper
        if player_id == 86:
            team_id = 1  # Assign goalkeeper to team 1

        self.player_team_dict[player_id] = team_id  # Store the player ID and team ID association

        return team_id  # Return the assigned team ID
