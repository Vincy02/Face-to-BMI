import os
import pandas as pd

data_path = "data/person.csv"
imgs_path = "data/img"
data = pd.read_csv(data_path, delimiter=';')

import feature_engineering as fe
new_data = fe.feature_engineering(data.copy())

import analyse_photo as ap
import time
import tqdm
ids = set(new_data['id'])
id_to_delete = []

with tqdm.tqdm(total=len(os.listdir(imgs_path)), desc="Analisi immagini") as pbar:
	for filename in os.listdir(imgs_path):
		id = filename.rsplit('.', 1)[0]
		if id in ids:
			img_path = imgs_path+"/"+filename
			face = ap.extract_face(img_path, "./data/new_img/")
			if face is False:
				id_to_delete.append(id)
			else:
				img_path = "data/new_img/"+filename
				if ap.face_detection(img_path) is False:
					id_to_delete.append(id)
					os.remove(img_path)
		pbar.update(1)

new_data = new_data[~new_data['id'].isin(id_to_delete)]

new_data.to_csv("data/new_data.csv", index=False)

import os
import subprocess

os.chdir('face-parsing')
subprocess.run('python main.py')
os.chdir('..')