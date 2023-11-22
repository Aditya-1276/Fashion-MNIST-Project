import pandas as pd # data manipulation in dataframes
import numpy as np # statistical analysis 
from sklearn.model_selection import train_test_split # to split train data into train and validation
import itertools
import random
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

fashion_train_df = pd.read_csv('../data/raw/fashion-mnist_train.csv',sep=',')
fashion_test_df = pd.read_csv('../data/raw/fashion-mnist_test.csv', sep = ',')

train_labels = fashion_train_df.iloc[:, 0].to_numpy()
train_features = fashion_train_df.iloc[:, 1:].to_numpy()
test_labels = fashion_test_df.iloc[:, 0].to_numpy()
test_features = fashion_test_df.iloc[:, 1:].to_numpy()

train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, test_size=10000, random_state=42)

train_features = train_features/255
test_features = test_features/255
val_features = val_features/255

train_features = train_features.reshape(train_features.shape[0], * (28, 28, 1))
test_features = test_features.reshape(test_features.shape[0], * (28, 28, 1))
val_features = val_features.reshape(val_features.shape[0], * (28, 28, 1))

cnn3 = Sequential()
cnn3.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))    # Output shape: (None, 26, 26, 32)
cnn3.add(BatchNormalization())
cnn3.add(Conv2D(64, (3, 3), activation='relu')) # (None, 24, 24, 64)
cnn3.add(BatchNormalization())
cnn3.add(MaxPooling2D((2, 2)))  # (None, 12, 12, 64)
cnn3.add(Dropout(0.25))
cnn3.add(Conv2D(64, (3, 3), activation='relu')) # (None, 10, 10, 64)
cnn3.add(BatchNormalization())
cnn3.add(MaxPooling2D(pool_size=(2, 2)))  # (None, 4, 4, 64)
cnn3.add(Dropout(0.25))

cnn3.add(Flatten()) # (None, 1024)
cnn3.add(Dense(64, activation='relu'))  # (None, 64)
cnn3.add(Dense(10, activation='softmax'))   # (None, 10)

METRICS = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

cnn3.compile(loss ='sparse_categorical_crossentropy', optimizer='adam' ,metrics=['accuracy'])

# Defining hyperparameters
epochs = 1
batch_size = 512

# training the model
history = cnn3.fit(
    train_features, train_labels, 
    batch_size=batch_size, 
    epochs=epochs,
    verbose=1, 
    validation_data=(val_features, val_labels)
)

# Evaluating the modell on test dataset

evaluation = cnn3.evaluate(test_features, test_labels)
print(f'Test Accuracy : {evaluation[1]:.3f}')