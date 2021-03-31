from cv2 import cv2
import numpy as np
haar_cascade = None
haar_cascade_loaded = False
try:
    haar_cascade = cv2.CascadeClassifier('./FaceDetectionAssets/haarcascade_frontalface_default.xml')
    haar_cascade_loaded = True
except:
    print('haar cascade could not be loaded')


def read_image(filename : str):
    result = cv2.imread(filename)
    return result, None

def write_image(image, filename: str):
    cv2.imwrite(filename, image)

def to_grayscale(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return result, None

def median_blur(image, ksize : int):
    if ksize < 1 or ksize%2 == 0:
        return image, "Kernel size should be positive odd number"
    result = cv2.medianBlur(image, ksize=ksize)
    return result, None

def gaussian_blur(image, ksize_x : int, ksize_y : int):
    if ksize_x < 1 or ksize_y < 1 or ksize_x%2 == 0 or ksize_y%2 == 0:
        return image, "Kernel sizes should be positive odd numbers"
    result = cv2.GaussianBlur(image, (ksize_x, ksize_y), 0)
    return result, None

def average_blur(image, ksize_x : int, ksize_y : int):
    if ksize_x < 1 or ksize_y < 1 or ksize_x%2 == 0 or ksize_y%2 == 0:
        return image, "Kernel sizes should be positive odd numbers"
    result = cv2.blur(image, (ksize_x, ksize_y))
    return result, None

def bilateral_filter(image, d : int, sigma : float):
    if d < 1:
        return image, "d should be positive"
    if sigma <= 0:
        return image, "Sigma should be positive"
    result = cv2.bilateralFilter(image, d, sigma, sigma)
    return result, None

def global_threshold(image, threshold : int, value : int):
    if threshold < 0 or threshold > 255:
        return image, "Threshold should be between 0 and 255"
    if value < 0 or value > 255:
        return image, "Value should be between 0 and 255"
    result, _ = to_grayscale(image)
    _, result = cv2.threshold(result, threshold, value, cv2.THRESH_BINARY)
    return result, None

def mean_threshold(image, blocksize : int, c : float, value : int):
    if blocksize < 1 or blocksize%2 == 0:
        return image, "Blocksize should be positive odd number"
    if value < 0 or value > 255:
        return image, "Value should be between 0 and 255"
    result, _ = to_grayscale(image)
    result = cv2.adaptiveThreshold(result, value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blocksize, c)
    return result, None
    
def gaussian_threshold(image, blocksize : int, c : float, value : int):
    if blocksize < 1 or blocksize%2 == 0:
        return image, "Blocksize should be positive odd number"
    if value < 0 or value > 255:
        return image, "Value should be between 0 and 255"
    result, _ = to_grayscale(image)
    result = cv2.adaptiveThreshold(result, value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, c)
    return result, None

def sobel(image, dx : int, dy : int, ksize : int, delta : float):
    if not (dx == 1 or dx == 0) or not (dy == 1 or dy == 0):
        return image, "dx and dy should be 1 or 0"
    if ksize < 1 or ksize%2 == 0:
        return image, "Kernel size should be positive odd number"
    result, _ = to_grayscale(image)
    result = cv2.Sobel(result, cv2.CV_64F, dx, dy, ksize=ksize, delta=delta)
    result = np.absolute(result)
    result = np.uint8(result)
    return result, None

def laplacian(image, ksize : int, delta : float):
    if ksize < 1 or ksize % 2 == 0:
        return image, "Kernel size should be positive odd number"
    result, _ = to_grayscale(image)
    result = cv2.Laplacian(result, cv2.CV_64F, ksize=ksize, delta=delta)
    result = np.absolute(result)
    result = np.uint8(result)
    return result, None

def canny_edge_detection(image, threshold1 : int, threshold2 : int):
    if threshold1 < 0 or threshold1 > 255 or threshold2 < 0 or threshold2 > 255:
        return image, "Thresholds should be between 0 and 255"
    result, _ = to_grayscale(image)
    result = cv2.Canny(result, threshold1, threshold2)
    return result, None

def haar_frontal_face_detection(image, min_neighbours : int, scale : float):
    if not haar_cascade_loaded:
        return image, "Server error, service not available"
    if min_neighbours < 1:
        return image, "min_neighbours should be positive number"
    if scale <= 1:
        return image, "scale should be greater then 1"
    gray, _ = to_grayscale(image)

    detected_faces = haar_cascade.detectMultiScale(gray, minNeighbors=min_neighbours, scaleFactor=scale)
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
    return image, None