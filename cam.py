import numpy as np
import numpy.linalg as la

def points_depth(hp, R, C):
    """
    Point depths relative to camera. HZ2 6.15, p.162
    - hp is an array of homogeneous 2D points w(x, y, 1)
    """
    # If we are sure we are working with calibrated cameras and right-handed
    # rotations (la.det(R) == 1), we could simply return hp[:,2].
    # But we want to support view (uncalibrated camera) coordinates and left
    # handed cameras, we use the generic solution
    sdet = np.sign(la.det(R))
    m3 = R[2,:]
    depths = sdet * hp[:, 2] / la.norm(m3)
    return depths

def project_on_camera(cloud, K, R, C):
    """Project point cloud on camera K[R, -RC]"""
    hp = K.dot(R.dot(cloud[:, :3].T - C.reshape(-1, 1)))
    # Project points behind the camera to infinity
    depths = points_depth(hp.T, R, C)
    assert len(depths) == cloud.shape[0]
    hp[2, depths < 0] = 0
    return hp[:2,:] / hp[2]

