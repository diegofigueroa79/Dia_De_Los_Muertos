import cv2 as cv
import numpy as np
import dlib
import image_preprocess
import triangle_script
import warpTriangles

SKIP_FRAMES =3
FRAME_COUNT = 0

if __name__ == '__main__':
	
	# load face and landmark detectors
	landmark_path = "shape_predictor_68_face_landmarks.dat"
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(landmark_path)
	
	# get mask image and alpha image
	# load mask landmark points from text file
	mask, alpha = image_preprocess.imgPreprocess()
	mask_pts = image_preprocess.pointsPreprocess()
	
	size = mask.shape
	rect = (0, 0, size[1], size[0])
	
	# get array of indices of delauny triangles
	# from mask landmark points
	delaunyTris = triangle_script.delaunyTriangles(rect, mask_pts)
	
	#load webcam
	cap = cv.VideoCapture(0)
	
	if not cap.isOpened():
		print("Cannot open camera.")
		exit()
	
	#create loop for video frames
	while(True):
		ret, frame = cap.read()
		
		if not ret:
			print("Cannot receive frame.")
			break
		
		# convert frame to grey
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		finalImage = np.float32(frame)/255
		maskWarped = np.zeros(frame.shape)
		alphaWarped = np.zeros(frame.shape)
		
		# detect first face and landmarks
		if (FRAME_COUNT % SKIP_FRAMES == 0):
			faces = detector(gray, 0)
		
		if faces:
			landmarks = predictor(gray, faces[0])
			landmark_points = [(p.x, p.y) for p in landmarks.parts()]
			
			for i in range(0, len(delaunyTris)):
				t1 = []
				t2 = []
				
				for j in range(0,3):
					t1.append(mask_pts[delaunyTris[i][j]])
					t2.append(landmark_points[delaunyTris[i][j]])
			
				warpTriangles.warp(mask, maskWarped, t1, t2)
				warpTriangles.warp(alpha, alphaWarped, t1, t2)
			
			mask1 = alphaWarped/255
			mask2 = 1.0 - mask1
			
			temp1 = np.multiply(finalImage, mask2)
			temp2 = np.multiply(maskWarped, mask1)
			
			frame = temp1 + temp2
			
		#Display the frame
		cv.imshow("frame", frame)
		if cv.waitKey(1) == ord('q'):
			break
		FRAME_COUNT += 1
	
	cap.release()
	cv.destroyAllWindows()
		

