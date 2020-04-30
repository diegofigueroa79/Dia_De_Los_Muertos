import cv2 as cv
import dlib
import numpy as np


pt_diff = 204-149

image = cv.imread("profile.jpeg")
notepad = "rose_points.txt"
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_detector = dlib.get_frontal_face_detector()

def get_saved_pts(notepad):
	crown_points = []
	lines = np.loadtxt(notepad, dtype='uint16')
	
	for p in lines:
		crown_points.append((p[0], p[1]))
	
	return crown_points

crown_points = get_saved_pts(notepad)

crown = cv.imread("rose_crown.png", cv.IMREAD_UNCHANGED)

faces = face_detector(image, 0)
landmarks = landmark_detector(image, faces[0])

landmark_pts = [(p.x, p.y) for p in landmarks.parts()]
bottom_left = landmark_pts[0]
bottom_right = landmark_pts[16]
top_left = (bottom_left[0], bottom_left[1] - pt_diff)
top_right = (bottom_right[0], bottom_right[1] - pt_diff)


crown_triangles = [[crown_points[0], crown_points[2], crown_points[3]], [crown_points[0], crown_points[1], crown_points[3]]]
face_triangles = [[bottom_left, top_left, top_right], [bottom_left, bottom_right, top_right]]


def show(image):
	cv.imshow('im', image)
	cv.waitKey(0)
	cv.destroyAllWindows()
	






