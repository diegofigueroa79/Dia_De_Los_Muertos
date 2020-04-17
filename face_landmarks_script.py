import cv2 as cv
import numpy as np
import dlib


img = "family.jpg"
cv_detector_path = "haarcascade_frontalface_default.xml"
cv_detector = cv.CascadeClassifier(cv_detector_path)
dlib_detector = dlib.get_frontal_face_detector()
landmark_detector_path = "shape_predictor_68_face_landmarks.dat"
landmark_detector = dlib.shape_predictor(landmark_detector_path)

im = cv.imread(img)
gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

dlib_faces = dlib_detector(gray, 0)

opencv_faces = cv_detector.detectMultiScale(gray, 1.3, 5)

dlib_landmarks = [landmark_detector(im, face) for face in dlib_faces]

for face in dlib_landmarks:
	points = []
	for i in range(0,16):
		point = [face.part(i).x, face.part(i).y]
		points.append(point)
	
	points = np.array(points, dtype=np.int32)
	cv.polylines(im, [points], False, (255, 200, 0), thickness=2, lineType=cv.LINE_8)

cv.imshow("im", im)
cv.waitKey(0)
cv.destroyAllWindows()