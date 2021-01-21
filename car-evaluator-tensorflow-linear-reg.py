import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# Check tensorflow version
print('TensorFlow Ver.. ', tf.__version__)

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()

# Read and return dataset
def read_data_set(url, fieldnames):
    raw_dataset = pd.read_csv(
        url, names=fieldnames, dtype='float32', na_values='?', skipinitialspace=True)
    return raw_dataset

def plot_mileage(x, y):
    plt.scatter(train_features['mileage'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('mileage')
    plt.ylabel('price')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    raw_dataset = read_data_set('./train-test-data/cars_train.csv',
                                ['make', 'model', 'year', 'mileage', 'fuelType', 'engineCapacity', 'cylinders', 'price'])

    dataset = raw_dataset.copy()
    
    # split data in train and test
    train_dataset = dataset.sample(frac=0.7, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # split train test features from labels
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    train_labels = train_features.pop('price')
    test_labels = test_features.pop('price')

    # add normalizer
    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(train_features))
    # print(normalizer.mean.numpy())

    linear_model = tf.keras.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])
    linear_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error'
    )
    history = linear_model.fit(
        train_features, train_labels, 
        epochs=1200,
        # suppress logging
        verbose=0,
        # Calculate validation results on 20% of the training data
        validation_split = 0.2
    )
    plot_loss(history)

    test_results = {}
    test_results['linear_model'] = linear_model.evaluate(
        test_features, test_labels, verbose=0
    )
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print('history tail \n', hist.tail())

print('Car Evaluator with Tensorflow finished')
