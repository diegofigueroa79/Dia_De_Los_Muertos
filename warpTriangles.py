import cv2 as cv
import dlib
import numpy as np


def applyAffineTransform(src, srcTri, dstTri, size):

  # Given a pair of triangles, find the affine transform.
  warpMat = cv.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

  # Apply the Affine Transform just found to the src image
  dst = cv.warpAffine(src, warpMat, (size[0], size[1]), None,
             flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)

  return dst

def warp(profile, muertos_mask, tri1, tri2):
	r1 = cv.boundingRect(np.float32([tri1]))
	r2 = cv.boundingRect(np.float32([tri2]))
	
	t1Rect = []
	t2Rect = []
	t2RectInt = []
	
	for i in range(0,3):
		t1Rect.append(((tri1[i][0] - r1[0]), (tri1[i][1] - r1[1])))
		t2Rect.append(((tri2[i][0] - r2[0]), (tri2[i][1] - r2[1])))
		t2RectInt.append(((tri2[i][0] - r2[0]), (tri2[i][1] - r2[1])))
	
	mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
	cv.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)
	
	img1Rect = profile[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
	size = (r2[2], r2[3])
	
	img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
	img2Rect = img2Rect * mask
	muertos_mask[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = muertos_mask[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)
	muertos_mask[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = muertos_mask[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect
