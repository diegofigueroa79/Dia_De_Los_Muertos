import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
haar_cascade_path = "haarcascade_frontalface_default.xml"
face_detector = cv.CascadeClassifier(haar_cascade_path)

if not cap.isOpened():
	print("Cannot open camera.")
	exit()

while(1):
	# ret is going to return a boolean
	# if the frame is not read correctly
	ret, frame = cap.read()
	if not ret:
		print("Can't receive frame.")
		break
	
	# our operations on the frame come here
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	faces = face_detector.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		frame = cv.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
	
	#Display the resulting frame
	cv.imshow('frame', frame)
	if cv.waitKey(1) == ord('q'):
		break

cap.release()
cv.destroyAllWindows()