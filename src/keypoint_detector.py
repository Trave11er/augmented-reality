import abc

import cv2

class KeypointDetector(abc.ABC):
    @abc.abstractmethod
    def detect_and_compute(image):
        pass

class OrbKeypointDetector(KeypointDetector):
    def __init__(self):
        self.orb = cv2.ORB_create()

    def detect_and_compute(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors

class SiftKeypointDetector(KeypointDetector):
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def detect_and_compute(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors

