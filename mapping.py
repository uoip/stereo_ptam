import numpy as np

from queue import Queue
from threading import Thread, Lock, Condition
import time

from itertools import chain
from collections import defaultdict

from optimization import LocalBA
from components import Measurement



class Mapping(object):
    def __init__(self, graph, params):
        self.graph = graph
        self.params = params
        self.local_keyframes = []

        self.optimizer = LocalBA()

    def add_keyframe(self, keyframe, measurements):
        self.graph.add_keyframe(keyframe)
        self.create_points(keyframe)

        for m in measurements:
            self.graph.add_measurement(keyframe, m.mappoint, m)

        self.local_keyframes.clear()
        self.local_keyframes.append(keyframe)

        self.fill(self.local_keyframes, keyframe)
        self.refind(self.local_keyframes, self.get_owned_points(keyframe))

        self.bundle_adjust(self.local_keyframes)
        self.points_culling(self.local_keyframes)

    def fill(self, keyframes, keyframe):
        covisible = sorted(
            keyframe.covisibility_keyframes().items(), 
            key=lambda _:_[1], reverse=True)

        for kf, n in covisible:
            if n > 0 and kf not in keyframes and self.is_safe(kf):
                keyframes.append(kf)
                if len(keyframes) >= self.params.local_window_size:
                    return

    def create_points(self, keyframe):
        mappoints, measurements = keyframe.triangulate()
        self.add_measurements(keyframe, mappoints, measurements)

    def add_measurements(self, keyframe, mappoints, measurements):
        for mappoint, measurement in zip(mappoints, measurements):
            self.graph.add_mappoint(mappoint)
            self.graph.add_measurement(keyframe, mappoint, measurement)
            mappoint.increase_measurement_count()

    def bundle_adjust(self, keyframes):
        adjust_keyframes = set()
        for kf in keyframes:
            if not kf.is_fixed():
                adjust_keyframes.add(kf)

        fixed_keyframes = set()
        for kf in adjust_keyframes:
            for ck, n in kf.covisibility_keyframes().items():
                if (n > 0 and ck not in adjust_keyframes 
                    and self.is_safe(ck) and ck < kf):
                    fixed_keyframes.add(ck)

        self.optimizer.set_data(adjust_keyframes, fixed_keyframes)
        completed = self.optimizer.optimize(self.params.ba_max_iterations)

        self.optimizer.update_poses()
        self.optimizer.update_points()

        if completed:
            self.remove_measurements(self.optimizer.get_bad_measurements())
        return completed

    def is_safe(self, keyframe):
        return True

    def get_owned_points(self, keyframe):
        owned = []
        for m in keyframe.measurements():
            if m.from_triangulation():
                owned.append(m.mappoint)
        return owned

    def filter_unmatched_points(self, keyframe, mappoints):
        filtered = []
        for i in np.where(keyframe.can_view(mappoints))[0]:
            pt = mappoints[i]
            if (not pt.is_bad() and 
                not self.graph.has_measurement(keyframe, pt)):
                filtered.append(pt)
        return filtered

    def refind(self, keyframes, new_mappoints):    # time consuming
        if len(new_mappoints) == 0:
            return
        for keyframe in keyframes:
            filtered = self.filter_unmatched_points(keyframe, new_mappoints)
            if len(filtered) == 0:
                continue
            for mappoint in filtered:
                mappoint.increase_projection_count()

            measuremets = keyframe.match_mappoints(filtered, Measurement.Source.REFIND)

            for m in measuremets:
                self.graph.add_measurement(keyframe, m.mappoint, m)
                m.mappoint.increase_measurement_count()

    def remove_measurements(self, measurements):
        for m in measurements:
            m.mappoint.increase_outlier_count()
            self.graph.remove_measurement(m)

    def points_culling(self, keyframes):    # Remove bad mappoints
        mappoints = set(chain(*[kf.mappoints() for kf in keyframes]))
        for pt in mappoints:
            if pt.is_bad():
                self.graph.remove_mappoint(pt)




class MappingThread(Mapping):
    def __init__(self, graph, params):
        super().__init__(graph, params)

        self._requests_cv = Condition()
        self._requests = [False, False]   # requests: [LOCKWINDOW_REQUEST, PROCESS_REQUEST]

        self._lock = Lock()
        self.locked_window = set()
        self.status = defaultdict(bool)
        
        self._queue = Queue()
        self.maintenance_thread = Thread(target=self.maintenance)
        self.maintenance_thread.start()

    def add_keyframe(self, keyframe, measurements): 
        self.graph.add_keyframe(keyframe)

        self.create_points(keyframe)
        for m in measurements:
            self.graph.add_measurement(keyframe, m.mappoint, m)

        self._queue.put(keyframe)
        with self._requests_cv:
            self._requests_cv.notify()
        
    def maintenance(self):
        stopped = False
        while not stopped:
            while not self._queue.empty():
                keyframe = self._queue.get()
                if keyframe is None:
                    stopped = True
                    self._requests[1] = True
                    break
                else:
                    self.local_keyframes.append(keyframe)
                    if len(self.local_keyframes) >= 5:
                        self._requests[1] = True
                        break

            with self._requests_cv:
                if self._requests.count(True) == 0:
                    self._requests_cv.wait()

                    while not self._queue.empty():
                        keyframe = self._queue.get()
                        if keyframe is None:
                            stopped = True
                            self._requests[1] = True
                            break
                        else:
                            self.local_keyframes.append(keyframe)
                            if len(self.local_keyframes) >= 5:
                                self._requests[1] = True

                requests = self._requests[:]
                self._requests[0] = False
                self._requests[1] = False

            self.status['processing'] = True

            if requests[1] and len(self.local_keyframes) > 0:
                self.fill(self.local_keyframes, self.local_keyframes[-1])

            if requests[0]:
                with self._lock:
                    for kf in self.local_keyframes:
                        self.locked_window.add(kf)
                        for ck, n in kf.covisibility_keyframes().items():
                            if n > 0:
                                self.locked_window.add(ck)
                    self.status['window_locked'] = True

            if requests[1] and len(self.local_keyframes) > 0:
                completed = self.bundle_adjust(self.local_keyframes)
                if completed:
                    self.points_culling(self.local_keyframes)
                self.local_keyframes.clear()

            self.status['processing'] = False

    def stop(self):
        with self._requests_cv:
            self._requests_cv.notify()

        while not self._queue.empty():
            time.sleep(1e-4)
        self._queue.put(None)   # sentinel value
        self.maintenance_thread.join()
        print('mapping stopped')

    def is_safe(self, keyframe):
        with self._lock:
            return not self.is_window_locked() or keyframe in self.locked_window

    def is_processing(self):
        return self.status['processing']

    def lock_window(self):
        with self._lock:
            self.status['window_locked'] = False
            self.locked_window.clear()

        with self._requests_cv:
            self._requests[0] = True
            self._requests_cv.notify()

        while not self.is_window_locked():
            time.sleep(1e-4)
        return self.locked_window

    def free_window(self):
        with self._lock:
            self.status['window_locked'] = False
            self.locked_window.clear()

    def is_window_locked(self):
        return self.status['window_locked']

    def wait_until_empty_queue(self):
        while not self._queue.empty():
            time.sleep(1e-4)

    def interrupt_ba(self):
        self.optimizer.abort()