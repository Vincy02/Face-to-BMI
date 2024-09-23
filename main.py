import os
import pandas as pd
import time
import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/new_data.csv")
image_dir = 'data/parsed_img/'
IMG_DIM = 128

def load_data(df):
	X = []
	y = []
	with tqdm.tqdm(total=len(os.listdir(image_dir)), desc="Carico immagini") as pbar:
		for index, row in df.iterrows():
			img_path = image_dir + str(row['id']) + '.jpg'
			img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_DIM, IMG_DIM), color_mode='grayscale')
			img_array = tf.keras.preprocessing.image.img_to_array(img)
			img_array /= 255.0
			X.append(img_array)
			y.append(row['bmi'])
			pbar.update(1)
	return np.array(X), np.array(y)

X, y = load_data(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
del X
del y

model = Sequential([
    Input((IMG_DIM, IMG_DIM, 1)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1)
])

from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

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