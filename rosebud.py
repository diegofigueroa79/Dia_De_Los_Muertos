import cv2 as cv
import dlib
import numpy as np


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

points = get_saved_pts(notepad)

crown = cv.imread("rose_crown.png", cv.IMREAD_UNCHANGED)

faces = face_detector(image, 0)
landmarks = landmark_detector(image, faces[0])

def show(image):
	cv.imshow('im', image)
	cv.waitKey(0)
	cv.destroyAllWindows()






