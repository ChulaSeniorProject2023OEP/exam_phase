import cv2
import mediapipe as mp
import numpy as np

def distance(p1, p2):
    '''
    Calculate distance between two points
    params:
        p1: tuple (x, y)
        p2: tuple (x, y)
    return:
        distance: float
    '''
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

