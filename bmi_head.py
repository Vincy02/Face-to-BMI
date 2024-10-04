import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class BMIHead(tf.keras.layers.Layer):
	def __init__(self, trainable=True, **kwargs):
		super(BMIHead, self).__init__(**kwargs)
		self.trainable = trainable
		self.layer1 = tf.keras.layers.Dense(128, activation='relu')
		self.layer2 = tf.keras.layers.Dense(64, activation='relu')
		self.layer3 = tf.keras.layers.Dense(32, activation='relu')
		self.layer4 = tf.keras.layers.Dense(16, activation='relu')
		self.layer_out = tf.keras.layers.Dense(1)

	def call(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		out = self.layer_out(x)
		return out