import abc

import cv2

class KeypointMatcher(abc.ABC):
    @abc.abstractmethod
    def match(self, descriptors_1, descriptors_2):
        pass

class BfKeypointMatcher(KeypointMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match(self, descriptors_1, descriptors_2):
        matches = self.bf.match(descriptors_1, descriptors_2)
        # sort them in the order of their distance
        # the lower the distance, the better the match
        matches = sorted(matches, key=lambda match: match.distance)
        print(list(m.distance for m in matches))
        return matches


class FlannKeypointMatcher(KeypointMatcher):
    def __init__(self):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)   # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, descriptors_1, descriptors_2):
        matches = self.flann.knnMatch(descriptors_1, descriptors_2, k=2) 
        good = []
        for pair in matches:
            if len(pair) == 2 and pair[0].distance < 0.75*pair[1].distance:
              good.append(pair[0])

        return good
