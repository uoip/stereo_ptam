import numpy as np 
import g2o



class MotionModel(object):
    def __init__(self, 
            timestamp=None, 
            initial_position=np.zeros(3), 
            initial_orientation=g2o.Quaternion(), 
            initial_covariance=None):

        self.timestamp = timestamp
        self.position = initial_position
        self.orientation = initial_orientation
        self.covariance = initial_covariance    # pose covariance

        self.v_linear = np.zeros(3)    # linear velocity
        self.v_angular_angle = 0
        self.v_angular_axis = np.array([1, 0, 0])

        self.initialized = False
        # damping factor
        self.damp = 0.95

    def current_pose(self):
        '''
        Get the current camera pose.
        '''
        return (g2o.Isometry3d(self.orientation, self.position), 
            self.covariance)

    def predict_pose(self, timestamp):
        '''
        Predict the next camera pose.
        '''
        if not self.initialized:
            return (g2o.Isometry3d(self.orientation, self.position), 
                self.covariance)
        
        dt = timestamp - self.timestamp

        delta_angle = g2o.AngleAxis(
            self.v_angular_angle * dt * self.damp, 
            self.v_angular_axis)
        delta_orientation = g2o.Quaternion(delta_angle)

        position = self.position + self.v_linear * dt * self.damp
        orientation = self.orientation * delta_orientation

        return (g2o.Isometry3d(orientation, position), self.covariance)

    def update_pose(self, timestamp, 
            new_position, new_orientation, new_covariance=None):
        '''
        Update the motion model when given a new camera pose.
        '''
        if self.initialized:
            dt = timestamp - self.timestamp
            assert dt != 0

            v_linear = (new_position - self.position) / dt
            self.v_linear = v_linear

            delta_q = self.orientation.inverse() * new_orientation
            delta_q.normalize()

            delta_angle = g2o.AngleAxis(delta_q)
            angle = delta_angle.angle()
            axis = delta_angle.axis()

            if angle > np.pi:
                axis = axis * -1
                angle = 2 * np.pi - angle

            self.v_angular_axis = axis
            self.v_angular_angle = angle / dt
            
        self.timestamp = timestamp
        self.position = new_position
        self.orientation = new_orientation
        self.covariance = new_covariance
        self.initialized = True

    def apply_correction(self, correction):     # corr: g2o.Isometry3d or matrix44
        '''
        Reset the model given a new camera pose.
        Note: This method will be called when it happens an abrupt change in the pose (LoopClosing)
        '''
        if not isinstance(correction, g2o.Isometry3d):
            correction = g2o.Isometry3d(correction)

        current = g2o.Isometry3d(self.orientation, self.position)
        current = current * correction

        self.position = current.position()
        self.orientation = current.orientation()

        self.v_linear = (
            correction.inverse().orientation() * self.v_linear)
        self.v_angular_axis = (
            correction.inverse().orientation() * self.v_angular_axis)