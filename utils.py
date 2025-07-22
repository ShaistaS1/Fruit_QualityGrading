import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_ripeness(image):
    """Calculate ripeness percentage (0-100) using HSV color space"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    return round((np.mean(hue) / 180) * 100, 2)

def extract_texture(image):
    """Analyze texture using Gray-Level Co-occurrence Matrix (GLCM)"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    return (
        round(graycoprops(glcm, 'contrast')[0, 0], 2),
        round(graycoprops(glcm, 'homogeneity')[0, 0], 2)
    )

def detect_defects(image):
    """Detect defects using Canny edge detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return round((np.sum(edges > 0) / edges.size) * 100, 2)