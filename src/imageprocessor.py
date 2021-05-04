from cv2 import cv2
import numpy as np

    
class ImageProcessor:
    """
    Class containing methods used for image manipulation
    """

    def __init__(self):
        """
        Tries to load haar cascade from file and sets flag if succeeds
        """

        self.haar_cascade = None
        self.haar_cascade_loaded = False
        try:
            self.haar_cascade = cv2.CascadeClassifier('./FaceDetectionAssets/haarcascade_frontalface_default.xml')
            self.haar_cascade_loaded = True
        except:
            print('haar cascade could not be loaded')

    def read_image(self, filename : str):
        """
        Loads image from file

        :type filename: string
        :param filename: Path of image to load

        :rtype: CV_U8, None
        :return: Loaded image
        """
        result = cv2.imread(filename)
        return result, None

    def bgr2rgb(self, image):
        """
        Converts image from bgr to rgb

        :type image: CV_8U
        :param image: Image to convert

        :rtype: CV_U8, None
        :return: Converted image
        """

        return cv2.cvtColor(image,cv2.COLOR_BGR2RGB), None

    def write_image(self, image, filename: str):
        """
        Writes image to file

        :type image: CV_8U
        :param image: Image to write

        :type filename: string
        :param filename: Path of image to write
        """

        cv2.imwrite(filename, image)

    def to_grayscale(self, image):
        """
        Converts image from bgr to rgb

        :type image: CV_8U
        :param image: Image to convert

        :rtype: CV_U8, None
        :return: Converted image to grayscale
        """

        result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return result, None

    def median_blur(self, image, ksize : int):
        """
        Applies median blur to an image

        :type image: CV_8U
        :param image: Image to blur

        :type ksize: int
        :param ksize: Size of kernel. Must be positive odd number

        :rtype: CV_U8, string
        :return: Returns blured image and None. If ksize is not correct returns original image and error
        """

        if ksize < 1 or ksize%2 == 0:
            return image, "Kernel size should be positive odd number"
        result = cv2.medianBlur(image, ksize=ksize)
        return result, None

    def gaussian_blur(self, image, ksize_x : int, ksize_y : int):
        """
        Applies gaussian blur to an image

        :type image: CV_8U
        :param image: Image to blur

        :type ksize_x: int
        :param ksize_x: X size of kernel. Must be positive odd number

        :type ksize_y: int
        :param ksize_y: Y size of kernel. Must be positive odd number

        :rtype: CV_U8, string
        :return: Returns blured image and None. If either of ksizes is not correct returns original image and error
        """

        if ksize_x < 1 or ksize_y < 1 or ksize_x%2 == 0 or ksize_y%2 == 0:
            return image, "Kernel sizes should be positive odd numbers"
        result = cv2.GaussianBlur(image, (ksize_x, ksize_y), 0)
        return result, None

    def average_blur(self, image, ksize_x : int, ksize_y : int):
        """
        Applies averaging blur to an image

        :type image: CV_8U
        :param image: Image to blur

        :type ksize_x: int
        :param ksize_x: X size of kernel. Must be positive odd number

        :type ksize_y: int
        :param ksize_y: Y size of kernel. Must be positive odd number

        :rtype: CV_U8, string
        :return: Returns blured image and None. If either of ksizes is not correct returns original image and error
        """

        if ksize_x < 1 or ksize_y < 1 or ksize_x%2 == 0 or ksize_y%2 == 0:
            return image, "Kernel sizes should be positive odd numbers"
        result = cv2.blur(image, (ksize_x, ksize_y))
        return result, None

    def bilateral_filter(self, image, d : int, sigma : float):
        """
        Applies gaussian blur to an image

        :type image: CV_8U
        :param image: Image to blur

        :type d: int
        :param d: Diameter of each pixel neighborhood that is used during filtering. Must be positive.

        :type sigma: int
        :param sigma: Filter sigma in the color space. Must be positive number.

        :rtype: CV_U8, string
        :return: Returns filtered image and None. If either of params is not correct returns original image and error
        """

        if d < 1:
            return image, "d should be positive"
        if sigma <= 0:
            return image, "Sigma should be positive"
        result = cv2.bilateralFilter(image, d, sigma, sigma)
        return result, None

    def global_threshold(self, image, threshold : int, value : int):
        """
        Applies global thresholding to an image

        :type image: CV_8U
        :param image: Image to threshold

        :type threshold: int
        :param d: Threshold which determines whether pixcel changes value. 0 < Threshold <= 255

        :type value: int
        :param value: Value that pixels with values above threshold will get after thresholding. 0 < Value <= 255

        :rtype: CV_U8, string
        :return: Returns thresholded image and None. If either of params is not correct returns original image and error
        """

        if threshold <= 0 or threshold > 255:
            return image, "Threshold should be between 0 and 255"
        if value <= 0 or value > 255:
            return image, "Value should be between 0 and 255"
        result, _ = self.to_grayscale(image)
        _, result = cv2.threshold(result, threshold, value, cv2.THRESH_BINARY)
        return result, None

    def mean_threshold(self, image, blocksize : int, c : float, value : int):
        """
        Applies mean thresholding to an image

        :type image: CV_8U
        :param image: Image to threshold

        :type blocksize: int
        :param blocksize: Size of a pixel neighborhood that is used to calculate a threshold value for the pixel. Must be positive odd number

        :type c: int
        :param c: Constant subtracted from the mean

        :type value: int
        :param value: Value that pixels with values above threshold will get after thresholding. 0 < Value <= 255

        :rtype: CV_U8, string
        :return: Returns thresholded image and None. If either of params is not correct returns original image and error
        """

        if blocksize < 1 or blocksize%2 == 0:
            return image, "Blocksize should be positive odd number"
        if value <= 0 or value > 255:
            return image, "Value should be between 0 and 255"
        result, _ = self.to_grayscale(image)
        result = cv2.adaptiveThreshold(result, value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blocksize, c)
        return result, None
        
    def gaussian_threshold(self, image, blocksize : int, c : float, value : int):
        """
        Applies gaussian thresholding to an image

        :type image: CV_8U
        :param image: Image to threshold

        :type blocksize: int
        :param blocksize: Size of a pixel neighborhood that is used to calculate a threshold value for the pixel. Must be positive odd number

        :type c: int
        :param c: Constant subtracted from the mean

        :type value: int
        :param value: Value that pixels with values above threshold will get after thresholding. 0 < Value <= 255

        :rtype: CV_U8, string
        :return: Returns thresholded image and None. If either of params is not correct returns original image and error
        """

        if blocksize < 1 or blocksize%2 == 0:
            return image, "Blocksize should be positive odd number"
        if value <= 0 or value > 255:
            return image, "Value should be between 0 and 255"
        result, _ = self.to_grayscale(image)
        result = cv2.adaptiveThreshold(result, value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, c)
        return result, None

    def sobel(self, image, dx : int, dy : int, ksize : int, delta : float):
        """
        Applies sobel gradient to an image

        :type image: CV_8U
        :param image: Image to threshold

        :type dx: int
        :param dx: Order of the derivative x. Equal to 1 or 0

        :type dy: int
        :param dy: Order of the derivative y. Equal to 1 or 0

        :type ksize: int
        :param ksize: Size of kernel. Must be a positive odd number

        :type delta: float
        :param delta: Value that is added to the results

        :rtype: CV_U8, string
        :return: Returns an image with aplied sobel gradient and None. If either of params is not correct returns original image and error
        """

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
        """
        Applies laplacian gradient to an image

        :type image: CV_8U
        :param image: Image to threshold

        :type ksize: int
        :param ksize: Size of kernel. Must be a positive odd number

        :type delta: float
        :param delta: Value that is added to the results

        :rtype: CV_U8, string
        :return: Returns an image with aplied sobel gradient and None. If either of params is not correct returns original image and error
        """

        if ksize < 1 or ksize % 2 == 0:
            return image, "Kernel size should be positive odd number"
        result, _ = self.to_grayscale(image)
        result = cv2.Laplacian(result, cv2.CV_64F, ksize=ksize, delta=delta)
        result = np.absolute(result)
        result = np.uint8(result)
        return result, None

    def canny_edge_detection(self, image, threshold1 : int, threshold2 : int):
        """
        Applies canny edge detector to an image

        :type image: CV_8U
        :param image: Image to threshold

        :type threshold1: int
        :param ksize: First threshold for canny edge detection. 0 < threshold1 <= 255

        :type threshold2: int
        :param delta: First threshold for canny edge detection. 0 < threshold2 <= 255

        :rtype: CV_U8, string
        :return: Returns an image containing edges detected by canny edge detection algorithm and None. If either of params is not correct returns original image and error
        """

        if threshold1 <= 0 or threshold1 > 255 or threshold2 <= 0 or threshold2 > 255:
            return image, "Thresholds should be between 0 and 255"
        result, _ = self.to_grayscale(image)
        result = cv2.Canny(result, threshold1, threshold2)
        return result, None

    def haar_frontal_face_detection(self, image, min_neighbours : int, scale : float):
        """
        Detects human faces within an image and marks them.

        :type image: CV_8U
        :param image: Image to threshold

        :type min_neighbours: int
        :param min_neighbours: Parameter specifying how many neighbors each candidate rectangle should have to retain it. Must be greater or equal 1

        :type scale: float
        :param scale: Parameter specifying how much the image size is reduced at each image scale. Must be greater then 1

        :rtype: CV_U8, string, [(int, int, int, int)]
        :return: Returns an image with marked faces, None and coordinates of detected faces (top left x, top left y, width, height). If either of params is not correct or haar cascade is not loaded returns original image and error
        """

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
        """
        Rotates an image without changing size of an image.

        :type image: CV_8U
        :param image: Image to threshold

        :type angle: int
        :param angle: Angle in degrees

        :rtype: CV_U8, None
        :return: Returns rotated image and None
        """

        angle = angle % 360
        (h, w) = image.shape[:2]
        center = (h // 2, w // 2)
        rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D(center, angle, 1.0), (w, h))
        return rotated, None

    def rotate(self, image, angle : int):
        """
        Rotates an image and extends its size if it is necessary.

        :type image: CV_8U
        :param image: Image to threshold

        :type angle: int
        :param angle: Angle in degrees

        :rtype: CV_U8, None
        :return: Returns rotated image and None
        """

        angle = angle % 360
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        (cos, sin) = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_shape = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
        M[0, 2] += (new_shape[0] / 2) - center[0]
        M[1, 2] += (new_shape[1] / 2) - center[1]
        return cv2.warpAffine(image, M, (new_shape[0], new_shape[1])), None