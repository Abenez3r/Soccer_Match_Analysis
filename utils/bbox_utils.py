def get_center_of_bbox(bbox):
    # Extract the coordinates of the bounding box
    x1, y1, x2, y2 = bbox
    # Calculate and return the center point of the bounding box (x, y)
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox):
    # Calculate and return the width of the bounding box
    return bbox[2] - bbox[0]

def measure_distance(p1, p2):
    # Calculate and return the Euclidean distance between two points (p1, p2)
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def measure_xy_distance(p1, p2):
    # Calculate and return the difference in x and y coordinates between two points (p1, p2)
    return p1[0] - p2[0], p1[1] - p2[1]

def get_foot_position(bbox):
    # Extract the coordinates of the bounding box
    x1, y1, x2, y2 = bbox
    # Return the position of the bottom-center point of the bounding box (foot position)
    return int((x1 + x2) / 2), int(y2)
