import tensorflow as tf
import analyse_photo as ap
import numpy as np
import shutil
import bmi_head
import re
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)

tmp_path = "./test/tmp/"
IMG_DIM = 128
pattern = r'\.jpg$'

if not os.path.exists(tmp_path):
	os.mkdir(tmp_path)

for filename in os.listdir("./test/"):
	match = re.search(pattern, filename, re.IGNORECASE)
	if not match:
		continue
	img_path =  "test/"+filename

	model = tf.keras.models.load_model("models/face_to_bmi.keras")

	face = ap.extract_face(img_path, "./test/tmp/")
	if face is True:
		img = tf.keras.preprocessing.image.load_img("test/tmp/"+filename, target_size=(IMG_DIM, IMG_DIM))
		img_array = tf.keras.preprocessing.image.img_to_array(img)
		img_array /= 255.0
		img_array = np.expand_dims(img_array, axis=0)

		predictions = model.predict(img_array)	
		bmi_predicted = predictions[0][0]

		print("BMI predetto per immagine '" + filename + "':", bmi_predicted)
	else:
		print("Non e' stato riconosciuto alcun volto oppure sono presenti piu' volti nella immagine '" + filename + "'.")
		print("Si prega di selezionare un'altra immagine.")
	print("\n")

shutil.rmtree(tmp_path)