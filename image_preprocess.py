import cv2 as cv
import numpy as np


def imgPreprocess():
	
	mask_path = "muertos_mask.png"
	mask = cv.imread(mask_path, cv.IMREAD_UNCHANGED)
	
	# split mask into color and alpha channels
	b,g,r,a = cv.split(mask)
	mask = cv.merge((b,g,r))
	mask = np.float32(mask)/255
	
	alpha = cv.merge((a,a,a))
	alpha = np.float32(alpha)
	
	return (mask, alpha)


def pointsPreprocess():
	
	mask_pts_path = "muertos_points.txt"
	
	mask_pts = []
	lines = np.loadtxt(mask_pts_path, dtype='uint16')
	
	for p in lines:
		mask_pts.append((p[0], p[1]))
	
	return mask_pts
	
	



