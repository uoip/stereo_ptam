import numpy as np
import cv2

from collections import defaultdict
from numbers import Number

from threading import Thread, Lock 
from queue import Queue



class ImageFeature(object):
    def __init__(self, image, params):
        # TODO: pyramid representation
        self.image = image 
        self.height, self.width = image.shape[:2]

        self.keypoints = []      # list of cv2.KeyPoint
        self.descriptors = []    # numpy.ndarray

        self.detector = params.feature_detector
        self.extractor = params.descriptor_extractor
        self.matcher = params.descriptor_matcher

        self.cell_size = params.matching_cell_size
        self.distance = params.matching_distance
        self.neighborhood = (
            params.matching_cell_size * params.matching_neighborhood)

        self._lock = Lock()

    def extract(self):
        self.keypoints = self.detector.detect(self.image)
        self.keypoints, self.descriptors = self.extractor.compute(
            self.image, self.keypoints)

        self.unmatched = np.ones(len(self.keypoints), dtype=bool)

    def draw_keypoints(self, name='keypoints', delay=1):
        if self.image.ndim == 2:
            image = np.repeat(self.image[..., np.newaxis], 3, axis=2)
        else:
            image = self.image
        img = cv2.drawKeypoints(image, self.keypoints, None, flags=0)
        cv2.imshow(name, img);cv2.waitKey(delay)

    def find_matches(self, predictions, descriptors):
        matches = dict()
        distances = defaultdict(lambda: float('inf'))
        for m, query_idx, train_idx in self.matched_by(descriptors):
            if m.distance > min(distances[train_idx], self.distance):
                continue

            pt1 = predictions[query_idx]
            pt2 = self.keypoints[train_idx].pt
            dx = pt1[0] - pt2[0]
            dy = pt1[1] - pt2[1]
            if np.sqrt(dx*dx + dy*dy) > self.neighborhood:
                continue

            matches[train_idx] = query_idx
            distances[train_idx] = m.distance
        matches = [(i, j) for j, i in matches.items()]
        return matches

    def matched_by(self, descriptors):
        with self._lock:
            unmatched_descriptors = self.descriptors[self.unmatched]
            if len(unmatched_descriptors) == 0:
                return []
            lookup = dict(zip(
                range(len(unmatched_descriptors)), 
                np.where(self.unmatched)[0]))

        # TODO: reduce matched points by using predicted position
        matches = self.matcher.match(
            np.array(descriptors), unmatched_descriptors)
        return [(m, m.queryIdx, m.trainIdx) for m in matches]

    def row_match(self, *args, **kwargs):
        return row_match(self.matcher, *args, **kwargs)

    def circular_stereo_match(self, *args, **kwargs):
        return circular_stereo_match(self.matcher, *args, **kwargs)

    def get_keypoint(self, i):
        return self.keypoints[i]
    def get_descriptor(self, i):
        return self.descriptors[i]

    def get_color(self, pt):
        x = int(np.clip(pt[0], 0, self.width-1))
        y = int(np.clip(pt[1], 0, self.height-1))
        color = self.image[y, x]
        if isinstance(color, Number):
            color = np.array([color, color, color])
        return color[::-1] / 255.

    def set_matched(self, i):
        with self._lock:
            self.unmatched[i] = False

    def get_unmatched_keypoints(self):
        keypoints = []
        descriptors = []
        indices = []

        with self._lock:
            for i in np.where(self.unmatched)[0]:
                keypoints.append(self.keypoints[i])
                descriptors.append(self.descriptors[i])
                indices.append(i)

        return keypoints, descriptors, indices



# TODO: only match points in neighboring rows
def row_match(matcher, kps1, desps1, kps2, desps2,
        matching_distance=40, 
        max_row_distance=2.5, 
        max_disparity=100):

    matches = matcher.match(np.array(desps1), np.array(desps2))
    good = []
    for m in matches:
        pt1 = kps1[m.queryIdx].pt
        pt2 = kps2[m.trainIdx].pt
        if (m.distance < matching_distance and 
            abs(pt1[1] - pt2[1]) < max_row_distance and 
            abs(pt1[0] - pt2[0]) < max_disparity):   # epipolar constraint
            good.append(m)
    return good
    


def circular_stereo_match(
        matcher, 
        desps1, desps2, matches12,
        desps3, desps4, matches34,
        matching_distance=30, 
        min_matches=10, ratio=0.8):

    dict_m13 = dict()
    dict_m24 = dict()
    dict_m34 = dict((m.queryIdx, m) for m in matches34)

    ms13 = matcher.knnMatch(np.array(desps1), np.array(desps3), k=2)
    for (m, n) in ms13:
        if m.distance < min(matching_distance, n.distance * ratio):
            dict_m13[m.queryIdx] = m

    # to avoid unnecessary computation
    if len(dict_m13) < min_matches:
        return []
                
    ms24 = matcher.knnMatch(np.array(desps2), np.array(desps4), k=2)
    for (m, n) in ms24:
        if m.distance < min(matching_distance, n.distance * ratio):
            dict_m24[m.queryIdx] = m

    matches = []
    for m in matches12:
        shared13 = dict_m13.get(m.queryIdx, None)
        shared24 = dict_m24.get(m.trainIdx, None)

        if shared13 is not None and shared24 is not None:
            shared34 = dict_m34.get(shared13.trainIdx, None)
            if (shared34 is not None and 
                shared34.trainIdx == shared24.trainIdx):
                matches.append((shared13, shared24))
    return matches