from cv2 import cv2
import numpy as np

    
class ImageProcessor:
    def __init__(self):
        self.haar_cascade = None
        self.haar_cascade_loaded = False
        try:
            self.haar_cascade = cv2.CascadeClassifier('./FaceDetectionAssets/haarcascade_frontalface_default.xml')
            self.haar_cascade_loaded = True
        except:
            print('haar cascade could not be loaded')

    def read_image(self, filename : str):
        result = cv2.imread(filename)
        return result, None

    def bgr2rgb(self, image):
        return cv2.cvtColor(image,cv2.COLOR_BGR2RGB), None

    def write_image(self, image, filename: str):
        cv2.imwrite(filename, image)

    def to_grayscale(self, image):
        result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return result, None

    def median_blur(self, image, ksize : int):
        if ksize < 1 or ksize%2 == 0:
            return image, "Kernel size should be positive odd number"
        result = cv2.medianBlur(image, ksize=ksize)
        return result, None

    def gaussian_blur(self, image, ksize_x : int, ksize_y : int):
        if ksize_x < 1 or ksize_y < 1 or ksize_x%2 == 0 or ksize_y%2 == 0:
            return image, "Kernel sizes should be positive odd numbers"
        result = cv2.GaussianBlur(image, (ksize_x, ksize_y), 0)
        return result, None

    def average_blur(self, image, ksize_x : int, ksize_y : int):
        if ksize_x < 1 or ksize_y < 1 or ksize_x%2 == 0 or ksize_y%2 == 0:
            return image, "Kernel sizes should be positive odd numbers"
        result = cv2.blur(image, (ksize_x, ksize_y))
        return result, None

    def bilateral_filter(self, image, d : int, sigma : float):
        if d < 1:
            return image, "d should be positive"
        if sigma <= 0:
            return image, "Sigma should be positive"
        result = cv2.bilateralFilter(image, d, sigma, sigma)
        return result, None

    def global_threshold(self, image, threshold : int, value : int):
        if threshold < 0 or threshold > 255:
            return image, "Threshold should be between 0 and 255"
        if value < 0 or value > 255:
            return image, "Value should be between 0 and 255"
        result, _ = self.to_grayscale(image)
        _, result = cv2.threshold(result, threshold, value, cv2.THRESH_BINARY)
        return result, None

    def mean_threshold(self, image, blocksize : int, c : float, value : int):
        if blocksize < 1 or blocksize%2 == 0:
            return image, "Blocksize should be positive odd number"
        if value < 0 or value > 255:
            return image, "Value should be between 0 and 255"
        result, _ = self.to_grayscale(image)
        result = cv2.adaptiveThreshold(result, value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blocksize, c)
        return result, None
        
    def gaussian_threshold(self, image, blocksize : int, c : float, value : int):
        if blocksize < 1 or blocksize%2 == 0:
            return image, "Blocksize should be positive odd number"
        if value < 0 or value > 255:
            return image, "Value should be between 0 and 255"
        result, _ = self.to_grayscale(image)
        result = cv2.adaptiveThreshold(result, value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, c)
        return result, None

    def sobel(self, image, dx : int, dy : int, ksize : int, delta : float):
        if not (dx == 1 or dx == 0) or not (dy == 1 or dy == 0):
            return image, "dx and dy should be 1 or 0"
        if ksize < 1 or ksize%2 == 0:
            return image, "Kernel size should be positive odd number"
        result, _ = self.to_grayscale(image)
        result = cv2.Sobel(result, cv2.CV_64F, dx, dy, ksize=ksize, delta=delta)
        result = np.absolute(result)
        result = np.uint8(result)
        return result, None

    def laplacian(self, image, ksize : int, delta : float):
        if ksize < 1 or ksize % 2 == 0:
            return image, "Kernel size should be positive odd number"
        result, _ = self.to_grayscale(image)
        result = cv2.Laplacian(result, cv2.CV_64F, ksize=ksize, delta=delta)
        result = np.absolute(result)
        result = np.uint8(result)
        return result, None

    def canny_edge_detection(self, image, threshold1 : int, threshold2 : int):
        if threshold1 < 0 or threshold1 > 255 or threshold2 < 0 or threshold2 > 255:
            return image, "Thresholds should be between 0 and 255"
        result, _ = self.to_grayscale(image)
        result = cv2.Canny(result, threshold1, threshold2)
        return result, None

    def haar_frontal_face_detection(self, image, min_neighbours : int, scale : float):
        if not self.haar_cascade_loaded:
            return image, "Server error, service not available"
        if min_neighbours < 1:
            return image, "min_neighbours should be positive number"
        if scale <= 1:
            return image, "scale should be greater then 1"
        gray, _ = self.to_grayscale(image)

        detected_faces = self.haar_cascade.detectMultiScale(gray, minNeighbors=min_neighbours, scaleFactor=scale)
        for (x, y, w, h) in detected_faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        return image, None, detected_faces

    def naive_rotate(self, image, angle : int):
        angle = angle % 360
        (w, h) = image.shape[:2]
        center = (h // 2, w // 2)
        rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D(center, angle, 1.0), (w, h))
        return rotated, None

    def rotate(self, image, angle : int):
        angle = angle % 360
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        (cos, sin) = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_shape = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
        M[0, 2] += (new_shape[0] / 2) - center[0]
        M[1, 2] += (new_shape[1] / 2) - center[1]
        return cv2.warpAffine(image, M, (new_shape[0], new_shape[1])), None