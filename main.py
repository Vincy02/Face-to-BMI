import os
import pandas as pd
import time
import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/new_data.csv")
image_dir = 'data/new_img/'
IMG_DIM = 128

num_images = int(len(os.listdir(image_dir)) / 3)
X = np.empty((num_images, IMG_DIM, IMG_DIM, 3))
y = np.empty(num_images)

with tqdm.tqdm(total=num_images, desc="Carico immagini") as pbar:
	for index, row in df.iterrows():
		if index == num_images:
			break
		img_path = image_dir + str(row['id']) + '.jpg'
		img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_DIM, IMG_DIM))
		img_array = tf.keras.preprocessing.image.img_to_array(img)
		img_array /= 255.0

		X[index] = img_array
		y[index] = row['bmi']
		pbar.update(1)

print("preparo test - train set")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
del X
del y

print("creo il modello ...")

from bmi_head import BMIHead

def get_model():
	base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_DIM, IMG_DIM, 3))

	for layer in base_model.layers:
	    layer.trainable = False

	x = base_model.output
	x = tf.keras.layers.Flatten()(x)
	x = BMIHead()(x)

	model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
	return model

model = get_model()
print("modello creato!")

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

print("inizio addestramento")

history = model.fit(
	X_train, y_train,
	epochs=10,
	batch_size=32,
	validation_data=(X_test, y_test)
)

loss = model.evaluate(X_test, y_test)
print("MSE | MAE:", loss)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()

model.save("models/face_to_bmi.keras")
print("Modello salvato!")