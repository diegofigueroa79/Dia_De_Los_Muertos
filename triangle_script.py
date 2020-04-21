import cv2 as cv
import numpy


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

def delaunyTriangles(rect, mask_points):
	
	subdiv = cv.Subdiv2D(rect)
	for p in mask_points:
		subdiv.insert((p[0], p[1]))
	
	triangleList = subdiv.getTriangleList()
	
	delaunyTris = []
	
	for triangle in triangleList:
		pts = []
		pts.append((triangle[0], triangle[1]))
		pts.append((triangle[2], triangle[3]))
		pts.append((triangle[4], triangle[5]))
		
		if checkPoint(rect, pts[0]) and checkPoint(rect, pts[1]) and checkPoint(rect, pts[2]):
			ind = []
			
			for j in range(0, 3):
				for k in range(0, len(mask_points)):
					if(abs(pts[j][0] - mask_points[k][0]) < 1.0 and abs(pts[j][1] - mask_points[k][1]) < 1.0):
						ind.append(k)
						
		if len(ind) == 3:
			delaunyTris.append((ind[0], ind[1], ind[2]))
	
	return delaunyTris
