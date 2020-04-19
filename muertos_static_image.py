import dlib
import cv2 as cv
import numpy as np
import warpTriangles


dlib_path = "shape_predictor_68_face_landmarks.dat"
ladnmark_detector = dlib.shape_predictor(dlib_path)
dlib_detector = dlib.get_frontal_face_detector()

profile = "profile.jpeg"
profile = cv.imread(profile)

mask_pts = "muertos_points.txt"
mask = "muertos_mask.png"
mask = cv.imread(mask, cv.IMREAD_UNCHANGED)

selected_points = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,
						18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,
						44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,
						60,61,62,63,64,65,66,67]

def get_saved_points(mask_points):
	points = []
	lines = np.loadtxt(mask_points, dtype='uint16')
	
	for p in lines:
		points.append((p[0], p[1]))
	
	return points

b,g,r,a = cv.split(mask)
mask = cv.merge((b,g,r))
mask = np.float32(mask)/255

alpha = cv.merge((a,a,a))
alpha = np.float32(alpha)

mask_feature_pts = get_saved_points(mask_pts)
size = mask.shape
rect = (0, 0, size[1], size[0])

subdiv = cv.Subdiv2D(rect)
for p in mask_feature_pts:
	subdiv.insert((p[0], p[1]))

triangleList = subdiv.getTriangleList()

def checkPoint(rect, point):
	if point[0] < rect[0]:
		return False
	elif point[1] < rect[1]:
		return False
	elif point[0] > rect[2]:
		return False
	elif point[1] > rect[3]:
		return False
	else: return True

delaunyTris = []

for triangle in triangleList:
	pts = []
	pts.append((triangle[0], triangle[1]))
	pts.append((triangle[2], triangle[3]))
	pts.append((triangle[4], triangle[5]))
	
	if checkPoint(rect, pts[0]) and checkPoint(rect, pts[1]) and checkPoint(rect, pts[2]):
		ind = []
		
		for j in range(0,3):
			for k in range(0, len(mask_feature_pts)):
				if(abs(pts[j][0] - mask_feature_pts[k][0]) < 1.0 and abs(pts[j][1] - mask_feature_pts[k][1]) < 1.0):
					ind.append(k)
	
	if len(ind) == 3:
		delaunyTris.append((ind[0], ind[1], ind[2]))


profileImage = np.float32(profile)/255
maskWarped = np.zeros(profile.shape)
alphaWarped = np.zeros(profile.shape)

faces = dlib_detector(profile, 0)
landmarks = ladnmark_detector(profile, faces[0])
landmark_points = [(p.x, p.y) for p in landmarks.parts()]

for i in range(0, len(delaunyTris)):
	t1 = []
	t2 = []
	
	for j in range(0,3):
		t1.append(mask_feature_pts[delaunyTris[i][j]])
		t2.append(landmark_points[delaunyTris[i][j]])
	
	warpTriangles.warp(mask, maskWarped, t1, t2)
	warpTriangles.warp(alpha, alphaWarped, t1, t2)

mask1 = alphaWarped/255
mask2 = 1.0 - mask1

temp1 = np.multiply(profileImage, mask2)
temp2 = np.multiply(maskWarped, mask1)

result = temp1 + temp2

cv.imshow('result', result)
cv.waitKey(0)
cv.destroyAllWindows()

	
	



