import numpy as np
from cartesian import *
######
# calculate the lsq transform between 2 sets 
# start and goal are a nx3 numpy array 
# return a 4*4 transform matrix F that F*start[i] = goal[i] in a lsq way 
######
def registration_3d(inpts, outpts):
    # R @ inpts + t = outpts
    # to_E_from
    # cam_R_base @ base_p + cam_T_base = CAM_p
    # inpts: base_p (points in base frame)
    # outpts: CAM_p (tag position in camera frame) 
    """
    Takes in two sets of corresponding points, returns the rigid transformation matrix from the first to the second.
    """
    assert inpts.shape == outpts.shape
    inpts, outpts = np.copy(inpts), np.copy(outpts)
    
    inpt_mean = inpts.mean(axis=0)
    outpt_mean = outpts.mean(axis=0)
    
    outpts -= outpt_mean
    inpts -= inpt_mean

    X = inpts.T
    Y = outpts.T
    
    covariance = np.dot(X, Y.T)
    
    U, s, V = np.linalg.svd(covariance)
    S = np.diag(s)
    assert np.allclose(covariance, np.dot(U, np.dot(S, V)))
    V = V.T
    idmatrix = np.identity(3)
    idmatrix[2, 2] = np.linalg.det(np.dot(V, U.T))
    
    R = np.dot(np.dot(V, idmatrix), U.T)
    
    t = outpt_mean.T - np.dot(R, inpt_mean)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T
