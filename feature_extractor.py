import cv2
import numpy as np
from skimage.feature import hog

def detect_bruises(image):
    """Advanced bruise detection using HOG features"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fd = hog(gray, orientations=8, pixels_per_cell=(16,16),
             cells_per_block=(1,1), visualize=False)
    return round(np.mean(fd), 2)

def color_consistency(image):
    """Measure color uniformity"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    a_channel = lab[:,:,1]
    return round(np.std(a_channel), 2)