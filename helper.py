import cv2
import numpy as np

def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def warp(undist):
    if len(undist.shape)>2:
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    else:
        gray = undist
    # Grab the image shape
    img_size = (gray.shape[1], gray.shape[0])
    src = np.array([(580, 460), (205, 720), (1110, 720), (703, 460)],np.float32)
    dst = np.array([(320, 0), (320, 720), (960, 720), (960, 0)],np.float32)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, img_size)
    
    return warped,M,Minv