import cv2
import dlib
import numpy as np

face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

def extract_face(img_path, path_to_save):
	file_name = img_path.split('/')[-1]
	img = cv2.imread(img_path)
	try:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	except Exception as e:
		return False
	faces = face_cascade.detectMultiScale(gray)

	if len(faces) != 1:
		return False

	(x, y, w, h) = faces[0]

	face_landmarks = predictor(gray, dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h))

	landmark_8 = face_landmarks.part(8)
	h_max = abs(y - landmark_8.y)
	max_d = max(w, h_max)
	if max_d == h_max:
		h = h_max
		aux = int(abs(w - h)/2)
		x -= aux
		w += 2*aux
		if x < 0 or x+w > img.shape[1]:
			(x, y, w, h) = faces[0]
			h = w
	else:
		h = w
	face_region = img[y:y + h, x:x + w]

	cv2.imwrite(path_to_save+file_name, face_region)
	return True


def face_detection(img_path):
	img = cv2.imread(img_path)
	if img is None:
		return False
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray)
	return len(faces) == 1