import cv2

class UtilityClient:
    """
    Utility Client class
    ------------------------
    Class used for reusable helper functions regarding
    cropping, preprocessing, drawing landmarks and
    other utility based functions.
    """
    def __init__(self):
        pass

    def draw_points(self, frame, points, color=(0,255,0)):
        for (x, y) in points:
            cv2.circle(frame, (int(x), int(y)), 1, color, -1)