import tensorflow as tf
import analyse_photo as ap
import numpy as np
import os
import shutil

file_name = "test5.jpg"
img_path =  "test/"+file_name
tmp_path = "./test/tmp/"
IMG_DIM = 128

if os.path.exists(tmp_path):
	shutil.rmtree(tmp_path)

model = tf.keras.models.load_model("models/face_to_bmi.keras")

os.mkdir(tmp_path)

face = ap.extract_face(img_path, "./test/tmp/")
if face is True:
	import subprocess
	os.chdir('face-parsing')
	subprocess.run('python main.py --image ../test/tmp --output ../test/tmp')
	os.chdir('..')

	img = tf.keras.preprocessing.image.load_img("test/tmp/"+file_name, target_size=(IMG_DIM, IMG_DIM), color_mode='grayscale')
	img_array = tf.keras.preprocessing.image.img_to_array(img)
	img_array /= 255.0
	img_array = np.expand_dims(img_array, axis=0)

	predictions = model.predict(img_array)	
	bmi_predicted = predictions[0][0]

	print('BMI predetto:', bmi_predicted)
	shutil.rmtree(tmp_path)
else:
	print("Non e' stato riconosciuto alcun volto oppure sono presenti piu' volti nella immagine.")
	print("Selezionare un'altra immagine.")