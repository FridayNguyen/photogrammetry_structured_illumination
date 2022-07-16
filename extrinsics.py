# THIS ENTIRE FILE IS JUST THE CODE TO CALCULATE EXTRINSIC PARAMETERS, FROM ASSIGNMENT 3

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import visutils
import cv2

def makerotation(rx,ry,rz):
    """
    Generate a rotation matrix    

    Parameters
    ----------
    rx,ry,rz : floats
        Amount to rotate around x, y and z axes in degrees

    Returns
    -------
    R : 2D numpy.array (dtype=float)
        Rotation matrix of shape (3,3)
    """
    x = np.deg2rad(rx)
    y = np.deg2rad(ry)
    z = np.deg2rad(rz)
    
    x_rotation = np.array([[1,0,0],[0,np.cos(x), -np.sin(x)],[0, np.sin(x), np.cos(x)]])
    y_rotation = np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    z_rotation = np.array([[np.cos(z), -np.sin(z), 0],[np.sin(z), np.cos(z),0],[0,0,1]])
    
    return  x_rotation @ y_rotation @ z_rotation

class Camera:
    """
    A simple data structure describing camera parameters 
    
    The parameters describing the camera
    cam.f : float   --- camera focal length (in units of pixels)
    cam.c : 2x1 vector  --- offset of principle point
    cam.R : 3x3 matrix --- camera rotation
    cam.t : 3x1 vector --- camera translation 
    
    """
    
    def __init__(self,f,c,R,t):
        self.f = f
        self.c = c
        self.R = R
        self.t = t

    def __str__(self):
        return f'Camera : \n f={self.f} \n c={self.c.T} \n R={self.R} \n t = {self.t.T}'
    
    def project(self,pts3):
        """
        Project the given 3D points in world coordinates into the specified camera    

        Parameters
        ----------
        pts3 : 2D numpy.array (dtype=float)
            Coordinates of N points stored in a array of shape (3,N)

        Returns
        -------
        pts2 : 2D numpy.array (dtype=float)
            Image coordinates of N points stored in an array of shape (2,N)

        """
        assert(pts3.shape[0]==3)

        inv_R = np.linalg.inv(self.R)
        inv_Rt = -inv_R @ self.t
        three_by_four = np.concatenate((inv_R,inv_Rt),axis=1)
        
        ones_row = np.ones((1,pts3.shape[1]))
        four_by_one = np.concatenate((pts3,ones_row),axis=0)
        
        P3 = three_by_four @ four_by_one
        
        x_row = P3[0] * self.f / P3[2] + self.c[0]
        y_row = P3[1] * self.f / P3[2] + self.c[1]
        
        pts2 = np.vstack((x_row,y_row))
        
        assert(pts2.shape[1]==pts3.shape[1])
        assert(pts2.shape[0]==2)
    
        return pts2

 
    def update_extrinsics(self,params):
        """
        Given a vector of extrinsic parameters, update the camera
        to use the provided parameters.
  
        Parameters
        ----------
        params : 1D numpy.array of shape (6,) (dtype=float)
            Camera parameters we are optimizing over stored in a vector
            params[:3] are the rotation angles, params[3:] are the translation

        """ 
        rx = params[0]
        ry = params[1]
        rz = params[2]
        
        tx = params[3]
        ty = params[4]
        tz = params[5]
        
        self.R = makerotation(rx,ry,rz)
        self.t = np.array([[tx],[ty],[tz]])

def residuals(pts3,pts2,cam,params):
    """
    Compute the difference between the projection of 3D points by the camera
    with the given parameters and the observed 2D locations

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N)

    params : 1D numpy.array (dtype=float)
        Camera parameters we are optimizing stored in a vector of shape (6,)

    Returns
    -------
    residual : 1D numpy.array (dtype=float)
        Vector of residual 2D projection errors of size 2*N
                
    """
    
    cam.update_extrinsics(params)
    projected = cam.project(pts3)
    return np.subtract(projected,pts2).flatten()

def wrapped_residuals(params,pts3,pts2,cam):
    return residuals(pts3,pts2,cam,params)

def calibratePose(pts3,pts2,cam,params_init):
    """
    Calibrate the provided camera by updating R,t so that pts3 projects
    as close as possible to pts2

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N)

    cam : Camera
        Initial estimate of camera
        
    params_init : 1D numpy.array (dtype=float)
        Initial estimate of camera extrinsic parameters ()
        params[0:2] are the rotation angles, params[2:5] are the translation

    Returns
    -------
    cam : Camera
        Refined estimate of camera with updated R,t parameters
        
    """
    estimate = scipy.optimize.leastsq(wrapped_residuals, x0=params_init, args=(pts3,pts2,cam))
    best_params = estimate[0]
    cam.update_extrinsics(best_params)
    return cam

def getCameras():
    # load in the intrinsic camera parameters from 'calibration.pickle'
    intrinsics = np.load("calibration.pickle", allow_pickle=True)

    # create Camera objects representing the left and right cameras
    # use the known intrinsic parameters you loaded in.
    avg_f = (intrinsics["fx"] + intrinsics["fy"]) / 2
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    camL = Camera(f=avg_f,c=np.array([[cx,cy]]).T,t=np.array([[0,0,0]]).T, R=makerotation(0,0,0))
    camR = Camera(f=avg_f,c=np.array([[cx,cy]]).T,t=np.array([[0,0,0]]).T, R=makerotation(0,0,0))

    # load in the left and right images and find the coordinates of
    # the chessboard corners using OpenCV
    imgL = plt.imread('./calib_jpg_u/frame_C0_01.jpg')
    ret, cornersL = cv2.findChessboardCorners(imgL, (8,6), None)
    pts2L = cornersL.squeeze().T

    imgR = plt.imread('./calib_jpg_u/frame_C1_01.jpg')
    ret, cornersR = cv2.findChessboardCorners(imgR, (8,6), None)
    pts2R = cornersR.squeeze().T

    # generate the known 3D point coordinates of points on the checkerboard in cm
    pts3 = np.zeros((3,6*8))
    yy,xx = np.meshgrid(np.arange(8),np.arange(6))
    pts3[0,:] = 2.8*xx.reshape(1,-1)
    pts3[1,:] = 2.8*yy.reshape(1,-1)


    # Now use your calibratePose function to get the extrinsic parameters
    # for the two images. You may need to experiment with the initialization
    # in order to get a good result

    params_init = np.array([0,0,-1,0,0,-1]) 

    camL = calibratePose(pts3,pts2L,camL,params_init)
    camR = calibratePose(pts3,pts2R,camR,params_init)

    return camL, camR