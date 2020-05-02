import cv2 as cv
import numpy as np
import dlib
import image_preprocess
import triangle_script
import warpTriangles

SKIP_FRAMES =3
FRAME_COUNT = 0
CROWN_HT = 55

if __name__ == '__main__':
	
	# load face and landmark detectors
	landmark_path = "shape_predictor_68_face_landmarks.dat"
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(landmark_path)
	
	mask_path = "muertos_mask.png"
	mask_pts_path = "muertos_points.txt"
	
	crown_path = "rose_crown.png"
	crown_pts_path = "rose_points.txt"
	
	# get mask image and alpha image
	# load mask landmark points from text file
	mask, alpha = image_preprocess.imgPreprocess(mask_path)
	mask_pts = image_preprocess.pointsPreprocess(mask_pts_path)
	
	crown, crown_alpha = image_preprocess.imgPreprocess(crown_path)
	crown_points = image_preprocess.pointsPreprocess(crown_pts_path)
	
	size = mask.shape
	rect = (0, 0, size[1], size[0])
	
	# get array of indices of delauny triangles
	# from mask landmark points
	delaunyTris = triangle_script.delaunyTriangles(rect, mask_pts)
	
	# calculate crown triangles
	crown_triangles = [[crown_points[0], crown_points[2], crown_points[3]], [crown_points[0], crown_points[1], crown_points[3]]]
	
	
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
			
			# Calculate forehead triangles
			bottom_left = (landmark_points[0][0], landmark_points[0][1] - 20)
			bottom_right = (landmark_points[16][0], landmark_points[16][1] - 20)
			top_left = (bottom_left[0], bottom_left[1] - CROWN_HT)
			top_right = (bottom_right[0], bottom_right[1] - CROWN_HT)
			
			forehead_triangles = [[bottom_left, top_left, top_right], [bottom_left, bottom_right, top_right]]
			
			for i in range(0, len(delaunyTris)):
				t1 = []
				t2 = []
				
				for j in range(0,3):
					t1.append(mask_pts[delaunyTris[i][j]])
					t2.append(landmark_points[delaunyTris[i][j]])
			
				warpTriangles.warp(mask, maskWarped, t1, t2)
				warpTriangles.warp(alpha, alphaWarped, t1, t2)
			
			# warp crown triangles onto mask, and alpha
			warpTriangles.warp(crown, maskWarped, crown_triangles[0], forehead_triangles[0])
			warpTriangles.warp(crown, maskWarped, crown_triangles[1], forehead_triangles[1])
			warpTriangles.warp(crown_alpha, alphaWarped, crown_triangles[0], forehead_triangles[0])
			warpTriangles.warp(crown_alpha, alphaWarped, crown_triangles[1], forehead_triangles[1])
			
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
		

