import pandas as pd
import matplotlib.pyplot as plt


def analyse_data(data):
	print(data.info())

	numerical_columns = ['weight', 'height', 'bmi']

	numerical_summary = data[numerical_columns].describe()
	print(numerical_summary)

	data[numerical_columns].hist(bins=15, figsize=(15, 6), layout=(2, 3))
	plt.suptitle('Distributions of Numerical Variables')
	plt.show()

data_path = "data/new_data.csv"
analyse_data(pd.read_csv(data_path))