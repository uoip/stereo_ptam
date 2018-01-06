import numpy as np
import cv2
import os
import time

from collections import defaultdict, namedtuple

from threading import Thread, Lock
from multiprocessing import Process, Queue



class ImageReader(object):
    def __init__(self, ids, timestamps, cam=None):
        self.ids = ids
        self.timestamps = timestamps
        self.cam = cam
        self.cache = dict()
        self.idx = 0

        self.ahead = 10      # 10 images ahead of current index
        self.waiting = 1.5   # waiting time

        self.preload_thread = Thread(target=self.preload)
        self.thread_started = False

    def read(self, path):
        img = cv2.imread(path, -1)
        if self.cam is None:
            return img
        else:
            return self.cam.rectify(img)
        
    def preload(self):
        idx = self.idx
        t = float('inf')
        while True:
            if time.time() - t > self.waiting:
                return
            if self.idx == idx:
                time.sleep(1e-2)
                continue
            
            for i in range(self.idx, self.idx + self.ahead):
                if i not in self.cache and i < len(self.ids):
                    self.cache[i] = self.read(self.ids[i])
            if self.idx + self.ahead > len(self.ids):
                return
            idx = self.idx
            t = time.time()
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        self.idx = idx
        # if not self.thread_started:
        #     self.thread_started = True
        #     self.preload_thread.start()

        if idx in self.cache:
            img = self.cache[idx]
            del self.cache[idx]
        else:   
            img = self.read(self.ids[idx])
        return img

    def __iter__(self):
        for i, timestamp in enumerate(self.timestamps):
            yield timestamp, self[i]

    @property
    def dtype(self):
        return self[0].dtype
    @property
    def shape(self):
        return self[0].shape




class KITTIOdometry(object):   # without lidar
    '''
    path example: 'path/to/your/KITTI odometry dataset/sequences/00'
    '''
    def __init__(self, path):
        Cam = namedtuple('cam', 'fx fy cx cy width height baseline')
        cam00_02 = Cam(718.856, 718.856, 607.1928, 185.2157, 1241, 376, 0.5371657)
        cam03 = Cam(721.5377, 721.5377, 609.5593, 172.854, 1241, 376, 0.53715)
        cam04_12 = Cam(707.0912, 707.0912, 601.8873, 183.1104, 1241, 376, 0.53715)

        path = os.path.expanduser(path)
        timestamps = np.loadtxt(os.path.join(path, 'times.txt'))
        self.left = ImageReader(self.listdir(os.path.join(path, 'image_2')), 
            timestamps)
        self.right = ImageReader(self.listdir(os.path.join(path, 'image_3')), 
            timestamps)

        assert len(self.left) == len(self.right)
        self.timestamps = self.left.timestamps

        sequence = int(path.strip(os.path.sep).split(os.path.sep)[-1])
        if sequence < 3:
            self.cam = cam00_02
        elif sequence == 3:
            self.cam = cam03
        elif sequence < 13:
            self.cam = cam04_12

    def sort(self, xs):
        return sorted(xs, key=lambda x:float(x[:-4]))

    def listdir(self, dir):
        files = [_ for _ in os.listdir(dir) if _.endswith('.png')]
        return [os.path.join(dir, _) for _ in self.sort(files)]

    def __len__(self):
        return len(self.left)






class Camera(object):
    def __init__(self, 
            width, height,
            intrinsic_matrix, 
            undistort_rectify=False,
            extrinsic_matrix=None,
            distortion_coeffs=None,
            rectification_matrix=None,
            projection_matrix=None):

        self.width = width
        self.height = height
        self.intrinsic_matrix = intrinsic_matrix
        self.extrinsic_matrix = extrinsic_matrix
        self.distortion_coeffs = distortion_coeffs
        self.rectification_matrix = rectification_matrix
        self.projection_matrix = projection_matrix
        self.undistort_rectify = undistort_rectify
        self.fx = intrinsic_matrix[0, 0]
        self.fy = intrinsic_matrix[1, 1]
        self.cx = intrinsic_matrix[0, 2]
        self.cy = intrinsic_matrix[1, 2]

        if undistort_rectify:
            self.remap = cv2.initUndistortRectifyMap(
                cameraMatrix=self.intrinsic_matrix,
                distCoeffs=self.distortion_coeffs,
                R=self.rectification_matrix,
                newCameraMatrix=self.projection_matrix,
                size=(width, height),
                m1type=cv2.CV_8U)
        else:
            self.remap = None

    def rectify(self, img):
        if self.remap is None:
            return img
        else:
            return cv2.remap(img, *self.remap, cv2.INTER_LINEAR)

class StereoCamera(object):
    def __init__(self, left_cam, right_cam):
        self.left_cam = left_cam
        self.right_cam = right_cam

        self.width = left_cam.width
        self.height = left_cam.height
        self.intrinsic_matrix = left_cam.intrinsic_matrix
        self.extrinsic_matrix = left_cam.extrinsic_matrix
        self.fx = left_cam.fx
        self.fy = left_cam.fy
        self.cx = left_cam.cx
        self.cy = left_cam.cy
        self.baseline = abs(right_cam.projection_matrix[0, 3] / 
            right_cam.projection_matrix[0, 0])
        self.focal_baseline = self.fx * self.baseline


class EuRoCDataset(object):   # Stereo + IMU
    '''
    path example: 'path/to/your/EuRoC Mav dataset/MH_01_easy'
    '''
    def __init__(self, path, rectify=True):
        self.left_cam = Camera(
            width=752, height=480,
            intrinsic_matrix = np.array([
                [458.654, 0.000000, 367.215], 
                [0.000000, 457.296, 248.375], 
                [0.000000, 0.000000, 1.000000]]),
            undistort_rectify=rectify,
            distortion_coeffs = np.array(
                [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.000000]),
            rectification_matrix = np.array([
                [0.999966347530033, -0.001422739138722922, 0.008079580483432283],
                [0.001365741834644127, 0.9999741760894847, 0.007055629199258132],
                [-0.008089410156878961, -0.007044357138835809, 0.9999424675829176]]),
            projection_matrix = np.array([
                [435.2046959714599, 0, 367.4517211914062, 0],
                [0, 435.2046959714599, 252.2008514404297, 0],
                [0., 0, 1, 0]]),
            extrinsic_matrix = np.array([
                [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                [0.0, 0.0, 0.0, 1.0]])
        )  
        self.right_cam = Camera(
            width=752, height=480,
            intrinsic_matrix = np.array([
                [457.587, 0.000000, 379.999], 
                [0.000000, 456.134, 255.238], 
                [0.000000, 0.000000, 1.000000]]),
            undistort_rectify=rectify,
            distortion_coeffs = np.array(
                [-0.28368365, 0.07451284, -0.00010473, -3.555907e-05, 0.0]),
            rectification_matrix = np.array([
                [0.9999633526194376, -0.003625811871560086, 0.007755443660172947],
                [0.003680398547259526, 0.9999684752771629, -0.007035845251224894],
                [-0.007729688520722713, 0.007064130529506649, 0.999945173484644]]),
            projection_matrix = np.array([
                [435.2046959714599, 0, 367.4517211914062, -47.90639384423901],
                [0, 435.2046959714599, 252.2008514404297, 0],
                [0, 0, 1, 0]]),
            extrinsic_matrix = np.array([
                [0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556],
                [0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024],
                [-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038],
                [0.0, 0.0, 0.0, 1.0]])
        ) 
        
        path = os.path.expanduser(path)
        self.left = ImageReader(
            *self.list_imgs(os.path.join(path, 'mav0', 'cam0', 'data')), 
            self.left_cam)
        self.right = ImageReader(
            *self.list_imgs(os.path.join(path, 'mav0', 'cam1', 'data')), 
            self.right_cam)
        assert len(self.left) == len(self.right)
        self.timestamps = self.left.timestamps

        self.cam = StereoCamera(self.left_cam, self.right_cam)

    def list_imgs(self, dir):
        xs = [_ for _ in os.listdir(dir) if _.endswith('.png')]
        xs = sorted(xs, key=lambda x:float(x[:-4]))
        timestamps = [float(_[:-4]) * 1e-9 for _ in xs]
        return [os.path.join(dir, _) for _ in xs], timestamps

    def __len__(self):
        return len(self.left)