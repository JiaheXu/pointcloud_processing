import numpy as np
import scipy
import math

class frame:
    def __init__(self, R, t):
        self.R = R
        self.t = t
        r_t = np.concatenate((R, t), 1)
        bot = np.array([0, 0, 0, 1])
        self.F = np.concatenate((r_t, bot))
    def get_R(self):
        return self.R
    def get_t(self):
        return self.t
    def get_F(self):
        return self.F
#Get the transpose of Rotation matrix
def Ri(R):
    return np.transpose(R)

#get the Rotation matrix
def get_R(F):
    return F[0:3, 0:3]

#get the translation matrix
def get_t(F):
    return F[0:3, 3].reshape(3, 1)

#Determines whether the given matrix is a rotation matrix
def isRot(R):
    det = np.linalg.det(R)
    return (det == 0)

#Given Rotation and translation, return a homogeneous matrix
def concat_frame(R, t):
    t = t.reshape(3,1)
    r_t = np.concatenate((R, t), 1)
    bot = np.array([0, 0, 0, 1]).reshape(1,4)
    return np.concatenate((r_t, bot))
                          
#Get the inverse of a frame
def Fi(F):
    tmp_ri = Ri(get_R(F))
    tmp_ti = -tmp_ri @ get_t(F)
    return concat_frame(tmp_ri, tmp_ti)

# points: n*3  F:4*4
def points_transform(F,points):
    n = np.shape(points)[0]
    t = np.ones((n,1))
    t = t.reshape(n,1)
    points = np.concatenate((points,t) , 1)
    points = points.T
    points = F@points
    points = (points.T)[:,0:3]
    return points




