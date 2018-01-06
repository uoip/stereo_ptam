import numpy as np
import cv2

import g2o
from g2o.contrib import SmoothEstimatePropagator

import time
from threading import Thread, Lock
from queue import Queue

from collections import defaultdict, namedtuple

from optimization import PoseGraphOptimization
from components import Measurement



# a very simple implementation
class LoopDetection(object):
    def __init__(self, params):
        self.params = params
        self.nns = NearestNeighbors()

    def add_keyframe(self, keyframe):
        embedding = keyframe.feature.descriptors.mean(axis=0)
        self.nns.add_item(embedding, keyframe)

    def detect(self, keyframe):
        embedding = keyframe.feature.descriptors.mean(axis=0)
        kfs, ds = self.nns.search(embedding, k=20)

        if len(kfs) > 0 and kfs[0] == keyframe:
            kfs, ds = kfs[1:], ds[1:]
        if len(kfs) == 0:
            return None

        min_d = np.min(ds)
        for kf, d in zip(kfs, ds):
            if abs(kf.id - keyframe.id) < self.params.lc_min_inbetween_frames:
                continue
            if (np.linalg.norm(kf.position - keyframe.position) > 
                self.params.lc_max_inbetween_distance):
                break
            if d > self.params.lc_embedding_distance or d > min_d * 1.5:
                break
            return kf
        return None



class LoopClosing(object):
    def __init__(self, system, params):
        self.system = system
        self.params = params

        self.loop_detector = LoopDetection(params)
        self.optimizer = PoseGraphOptimization()

        self.loops = []
        self.stopped = False

        self._queue = Queue()
        self.maintenance_thread = Thread(target=self.maintenance)
        self.maintenance_thread.start()

    def stop(self):
        self.stopped = True
        self._queue.put(None)
        self.maintenance_thread.join()
        print('loop closing stopped')

    def add_keyframe(self, keyframe):
        self._queue.put(keyframe)
        self.loop_detector.add_keyframe(keyframe)

    def add_keyframes(self, keyframes):
        for kf in keyframes:
            self.add_keyframe(kf)

    def maintenance(self):
        last_query_keyframe = None
        while not self.stopped:
            keyframe = self._queue.get()
            if keyframe is None or self.stopped:
                return

            # check if this keyframe share many mappoints with a loop keyframe
            covisible = sorted(
                keyframe.covisibility_keyframes().items(), 
                key=lambda _:_[1], reverse=True)
            if any([(keyframe.id - _[0].id) > 5 for _ in covisible[:2]]):
                continue

            if (last_query_keyframe is not None and 
                abs(last_query_keyframe.id - keyframe.id) < 3):
                continue

            detected = self.loop_detector.detect(keyframe)
            if detected is None:
                continue

            query_keyframe = keyframe
            match_keyframe = detected

            result = match_and_estimate(
                query_keyframe, match_keyframe, self.params)

            if result is None:
                continue
            if (result.n_inliers < max(self.params.lc_inliers_threshold, 
                result.n_matches * self.params.lc_inliers_ratio)):
                continue

            dist = result.correction.position()
            if self.params.ground:
                dist = dist[:2]
            if np.abs(dist).max() > self.params.lc_distance_threshold:
                continue

            self.loops.append(
                (match_keyframe, query_keyframe, result.constraint))
            query_keyframe.set_loop(match_keyframe, result.constraint)

            # We have to ensure that the mapping thread is on a safe part of code, 
            # before the selection of KFs to optimize
            safe_window = self.system.mapping.lock_window()
            safe_window.add(self.system.reference)
            for kf in self.system.reference.covisibility_keyframes():
                safe_window.add(kf)

            
            # The safe window established between the Local Mapping must be 
            # inside the considered KFs.
            considered_keyframes = self.system.graph.keyframes()

            self.optimizer.set_data(considered_keyframes, self.loops)

            before_lc = [
                g2o.Isometry3d(kf.orientation, kf.position) for kf in safe_window]

            # Propagate initial estimate through 10% of total keyframes 
            # (or at least 20 keyframes)
            d = max(20, len(considered_keyframes) * 0.1)
            propagator = SmoothEstimatePropagator(self.optimizer, d)
            propagator.propagate(self.optimizer.vertex(match_keyframe.id))

            # self.optimizer.set_verbose(True)
            self.optimizer.optimize(20)
            
            # Exclude KFs that may being use by the local BA.
            self.optimizer.update_poses_and_points(
                considered_keyframes, exclude=safe_window)

            self.system.stop_adding_keyframes()

            # Wait until mapper flushes everything to the map
            self.system.mapping.wait_until_empty_queue()
            while self.system.mapping.is_processing():
                time.sleep(1e-4)

            # Calculating optimization introduced by local mapping while loop was been closed
            for i, kf in enumerate(safe_window):
                after_lc = g2o.Isometry3d(kf.orientation, kf.position)
                corr = before_lc[i].inverse() * after_lc

                vertex = self.optimizer.vertex(kf.id)
                vertex.set_estimate(vertex.estimate() * corr)

            self.system.pause()

            for keyframe in considered_keyframes[::-1]:
                if keyframe in safe_window:
                    reference = keyframe
                    break
            uncorrected = g2o.Isometry3d(
                reference.orientation, 
                reference.position)
            corrected = self.optimizer.vertex(reference.id).estimate()
            T = uncorrected.inverse() * corrected   # close to result.correction

            # We need to wait for the end of the current frame tracking and ensure that we
            # won't interfere with the tracker.
            while self.system.is_tracking():
                time.sleep(1e-4)
            self.system.set_loop_correction(T)

            # Updating keyframes and map points on the lba zone
            self.optimizer.update_poses_and_points(safe_window)

            # keyframes after loop closing
            keyframes = self.system.graph.keyframes()
            if len(keyframes) > len(considered_keyframes):
                self.optimizer.update_poses_and_points(
                    keyframes[len(considered_keyframes) - len(keyframes):], 
                    correction=T)

            for m13, _ in result.stereo_matches:
                query_meas = result.query_stereo_measurements[m13.queryIdx]
                match_meas = result.match_stereo_measurements[m13.trainIdx]

                new_query_meas = Measurement(
                    Measurement.Type.STEREO,
                    Measurement.Source.REFIND,
                    query_meas.get_keypoints(),
                    query_meas.get_descriptors())
                self.system.graph.add_measurement(
                    query_keyframe, match_meas.mappoint, new_query_meas)
                
                new_match_meas = Measurement(
                    Measurement.Type.STEREO,
                    Measurement.Source.REFIND,
                    match_meas.get_keypoints(),
                    match_meas.get_descriptors())
                self.system.graph.add_measurement(
                    match_keyframe, query_meas.mappoint, new_match_meas)

            self.system.mapping.free_window()
            self.system.resume_adding_keyframes()
            self.system.unpause()

            while not self._queue.empty():
                keyframe = self._queue.get()
                if keyframe is None:
                    return
            last_query_keyframe = query_keyframe
        


def match_and_estimate(query_keyframe, match_keyframe, params):
    query = defaultdict(list)
    for m in query_keyframe.measurements():
        if m.from_triangulation():
            query['measurements'].append(m)
            query['kps1'].append(m.get_keypoint(0))
            query['kps2'].append(m.get_keypoint(1))
            query['desps1'].append(m.get_descriptor(0))
            query['desps2'].append(m.get_descriptor(1))
            n = len(query['matches'])
            query['matches'].append(cv2.DMatch(n, n, 0))

    match = defaultdict(list)
    for m in match_keyframe.measurements():
        if m.from_triangulation():
            match['measurements'].append(m)
            match['kps1'].append(m.get_keypoint(0))
            match['kps2'].append(m.get_keypoint(1))
            match['desps1'].append(m.get_descriptor(0))
            match['desps2'].append(m.get_descriptor(1))
            n = len(match['matches'])
            match['matches'].append(cv2.DMatch(n, n, 0))

    stereo_matches = query_keyframe.feature.circular_stereo_match(
        query['desps1'], query['desps2'], query['matches'],
        match['desps1'], match['desps2'], match['matches'],
        params.matching_distance,
        params.lc_inliers_threshold)

    n_matches = len(stereo_matches)
    if n_matches < params.lc_inliers_threshold:
        return None

    for m13, _ in stereo_matches:
        i, j = m13.queryIdx, m13.trainIdx
        query['px'].append(query['kps1'][i].pt)
        query['pt'].append(query['measurements'][i].view)
        match['px'].append(match['kps1'][j].pt)
        match['pt'].append(match['measurements'][j].view)

    # query_keyframe's pose in match_keyframe's coordinates frame
    T13, inliers13 = solve_pnp_ransac(
        query['pt'], match['px'], match_keyframe.cam.intrinsic)

    T31, inliers31 = solve_pnp_ransac(
        match['pt'], query['px'], query_keyframe.cam.intrinsic)

    if T13 is None or T13 is None:
        return None

    delta = T31 * T13
    if (g2o.AngleAxis(delta.rotation()).angle() > 0.1 or
        np.linalg.norm(delta.translation()) > 0.5):          # 5.7° or 0.5m
        return None

    n_inliers = len(set(inliers13) & set(inliers31))
    query_pose = g2o.Isometry3d(
        query_keyframe.orientation, query_keyframe.position)
    match_pose = g2o.Isometry3d(
        match_keyframe.orientation, match_keyframe.position)

    # TODO: combine T13 and T31
    constraint = T13
    estimated_pose = match_pose * constraint
    correction = query_pose.inverse() * estimated_pose

    return namedtuple('MatchEstimateResult',
        ['estimated_pose', 'constraint', 'correction', 'query_stereo_measurements', 
        'match_stereo_measurements', 'stereo_matches', 'n_matches', 'n_inliers'])(
        estimated_pose, constraint, correction, query['measurements'], 
        match['measurements'], stereo_matches, n_matches, n_inliers)


def solve_pnp_ransac(pts3d, pts, intrinsic_matrix):
    val, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.array(pts3d), np.array(pts), 
            intrinsic_matrix, None, None, None, 
            False, 50, 2.0, 0.99, None)
    if inliers is None or len(inliers) < 5:
        return None, None

    T = g2o.Isometry3d(cv2.Rodrigues(rvec)[0], tvec)
    return T, inliers.ravel()
    


class NearestNeighbors(object):
    def __init__(self, dim=None):
        self.n = 0
        self.dim = dim
        self.items = dict()
        self.data = []
        if dim is not None:
            self.data = np.zeros((1000, dim), dtype='float32')

    def add_item(self, vector, item):
        assert vector.ndim == 1
        if self.n >= len(self.data):
            if self.dim is None:
                self.dim = len(vector)
                self.data = np.zeros((1000, self.dim), dtype='float32')
            else:
                self.data.resize(
                    (2 * len(self.data), self.dim) , refcheck=False)
        self.items[self.n] = item
        self.data[self.n] = vector
        self.n += 1

    def search(self, query, k):  # searching from 100000 items consume 30ms
        if len(self.data) == 0:
            return [], []

        ds = np.linalg.norm(query[np.newaxis, :] - self.data[:self.n], axis=1)
        ns = np.argsort(ds)[:k]
        return [self.items[n] for n in ns], ds[ns]