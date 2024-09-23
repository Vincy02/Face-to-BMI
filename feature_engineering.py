import pandas as pd

def feature_engineering(data):
	data = data[['id', 'weight', 'height']]

	data['weight'] = data['weight'] / 2.205
	data['height'] = data['height'] * 2.54
	data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)

	data = data.dropna(subset=['bmi'])
	data = data[data['bmi'] != 0]

	data = data[data['bmi'] > 15]
	data = data[data['bmi'] < 60]

	data['bmi'] = data['bmi'].round(1)
	return data