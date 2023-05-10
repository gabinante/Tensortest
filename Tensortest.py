import logging
import csv
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger()

data_folder = 'dataset/'
data_file = data_folder + 'iris.csv'

print(data_file)

parsed_data_file = pd.read_csv(data_file)
parsed_data_file.head()

x_values = parsed_data_file.iloc[:,0:4].values
y_values = parsed_data_file.iloc[:,4].values

logger.debug('transforming_values')

encoder =  LabelEncoder()
transformed_values = encoder.fit_transform(y_values)
print(transformed_values)


dummy_values = pd.get_dummies(transformed_values).values
print(dummy_values[0:5])

logger.debug('splitting data into training set and testing set')

X_train, X_test, y_train, y_test = train_test_split(x_values, dummy_values, test_size=0.2, random_state=0)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])

logger.debug('compiling model...')

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=50, epochs=100)


y_prediction = model.predict(X_test)
y_prediction

actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_prediction,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
