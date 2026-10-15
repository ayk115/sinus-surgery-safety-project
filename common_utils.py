import numpy as np

def angular_difference(angle1, angle2):
    """
    Calculates the shortest angular distance between two angles (in radians) on a circle.
    Returns the absolute difference.
    """
    diff = (angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi
    return np.abs(diff)
