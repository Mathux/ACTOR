# taken and adapted from https://github.com/JimmyZou/PolarHumanPoseShape/
import pickle
import numpy as np
import os


class Transform:
    def __init__(self, R=np.eye(3, dtype='float'), t=np.zeros(3, 'float'), s=np.ones(3, 'float')):
        self.R = R.copy()  # rotation
        self.t = t.reshape(-1).copy()  # translation
        self.s = s.copy()  # scale

    def __mul__(self, other):
        # combine two transformation together
        R = np.dot(self.R, other.R)
        t = np.dot(self.R, other.t * self.s) + self.t
        if not hasattr(other, 's'):
            other.s = np.ones(3, 'float').copy()
        s = other.s.copy()
        return Transform(R, t, s)

    def inv(self):
        # inverse the rigid tansformation
        R = self.R.T
        t = -np.dot(self.R.T, self.t)
        return Transform(R, t)

    def transform(self, xyz):
        # transform 3D point
        if not hasattr(self, 's'):
            self.s = np.ones(3, 'float').copy()
        assert xyz.shape[-1] == 3
        assert len(self.s) == 3
        return np.dot(xyz * self.s, self.R.T) + self.t

    def getmat4(self):
        # homogeneous transformation matrix
        M = np.eye(4)
        M[:3, :3] = self.R * self.s
        M[:3, 3] = self.t
        return M


def quat2R(quat):
    """
    Description
    ===========
    convert vector q to matrix R

    Parameters
    ==========
    :param quat: (4,) array

    Returns
    =======
    :return: (3,3) array
    """
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    n = w * w + x * x + y * y + z * z
    s = 2. / np.clip(n, 1e-7, 1e7)

    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z

    R = np.stack([1 - (yy + zz), xy - wz, xz + wy,
                  xy + wz, 1 - (xx + zz), yz - wx,
                  xz - wy, yz + wx, 1 - (xx + yy)])

    return R.reshape((3, 3))


def convert_param2tranform(param, scale=1):
    R = quat2R(param[0:4])
    t = param[4:7]
    s = scale * np.ones(3, 'float')
    return Transform(R, t, s)


class CameraParams:
    def __init__(self, cam_folder="data/phspdCameras"):
        
        # load camera params, save intrinsic and extrinsic camera parameters as a dictionary
        # intrinsic ['param_p', 'param_c1', 'param_d1', 'param_c2', 'param_d2', 'param_c3', 'param_d3']
        # extrinsic ['d1p', 'd2p', 'd3p', 'cd1', 'cd2', 'cd3']
        self.cam_params = []
        with open(os.path.join(cam_folder, "CamParams0906.pkl"), 'rb') as f:
            self.cam_params.append(pickle.load(f))
        with open(os.path.join(cam_folder, "CamParams0909.pkl"), 'rb') as f:
            self.cam_params.append(pickle.load(f))

        # corresponding cam params to each subject
        self.name_cam_params = {}  # {"name": 0 or 1}
        for name in ['subject06', 'subject09', 'subject11', 'subject05', 'subject12', 'subject04']:
            self.name_cam_params[name] = 0
        for name in ['subject03', 'subject01', 'subject02', 'subject10', 'subject07', 'subject08']:
            self.name_cam_params[name] = 1

        # corresponding cam params to each subject
        self.name_gender = {}  # {"name": 0 or 1}
        for name in ['subject02', 'subject03', 'subject04', 'subject05', 'subject06',
                     'subject08', 'subject09', 'subject11', 'subject12']:
            self.name_gender[name] = 0  # male
        for name in ['subject01', 'subject07', 'subject10']:
            self.name_gender[name] = 1  # female

    def get_intrinsic(self, cam_name, subject_no):
        """
        'p': polarization camera, color
        'c1': color camera for the 1st Kinect
        'd1': depth (ToF) camera for the 1st Kinect
        ...
        return
            (fx, fy, cx, cy)
        """
        assert cam_name in ['p', 'c1', 'd1', 'c2', 'd2', 'c3', 'd3']
        assert subject_no in ['subject06', 'subject09', 'subject11', 'subject05', 'subject12', 'subject04',
                              'subject03', 'subject01', 'subject02', 'subject10', 'subject07', 'subject08']
        fx, fy, cx, cy, _, _, _ = self.cam_params[self.name_cam_params[subject_no]]['param_%s' % cam_name]
        intrinsic = (fx, fy, cx, cy)
        return intrinsic

    def get_extrinsic(self, cams_name, subject_no):
        """
        The annotated poses and shapes are saved in polarization camera coordinate.
        'd1p': transform from polarization camera to 1st Kinect depth image
        'c1p': transform from polarization camera to 1st Kinect color image
        ...
        return
            transform class
        """
        assert cams_name in ['d1', 'd2', 'd3', 'c1', 'c2', 'c3']
        assert subject_no in ['subject06', 'subject09', 'subject11', 'subject05', 'subject12', 'subject04',
                              'subject03', 'subject01', 'subject02', 'subject10', 'subject07', 'subject08']

        if cams_name in ['d1p', 'd2p', 'd3p']:
            T = convert_param2tranform(self.cam_params[self.name_cam_params[subject_no]][cams_name])
        else:
            i = cams_name[1]
            T_dp = convert_param2tranform(self.cam_params[self.name_cam_params[subject_no]]['d%sp' % i])
            T_cd = convert_param2tranform(self.cam_params[self.name_cam_params[subject_no]]['cd%s' % i])
            T = T_cd * T_dp
        return T

    def get_gender(self, subject_no):
        return self.name_gender[subject_no]


if __name__ == '__main__':
    # test
    camera_params = CameraParams(data_dir='../..//data')
    T = camera_params.get_extrinsic('c2', 'subject01')
    print(T.getmat4())


