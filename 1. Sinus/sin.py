'''final version of the sine approximation by Ahmet Efe, Sven Leschber, Mika Rother'''

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# create an array with x values in [0,π], the sin of the x values and a random value
pi = np.pi
xs = np.arange(0, pi, 0.1)
sin = np.sin(xs)
rand = np.random.uniform(0, pi)

# create different values the model should predict later
x_pred = []
for k in range(len(xs)):
    if len(xs) == len(x_pred)+1:
        break
    val = (xs[k]+xs[k+1]) / 2
    x_pred.append(val)

# create a professional dataset for the sine model
# Options: <different split>
xs, sin = shuffle(xs, sin, random_state=0)
xs_train, xs_temp, sin_train, sin_temp = train_test_split(xs, sin, test_size = 0.3)
xs_val, xs_test, sin_val, sin_test = train_test_split(xs_temp, sin_temp, test_size = 0.5)

# now create a model for sine, using activation functions and different type of layer sizes
# Options: <number of layers> <size of layers> <activation function>
model = keras.Sequential([
    tf.keras.layers.Dense(512, activation='tanh', input_shape=(1,)),
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1)
    ])

# define callback functions that helpes the model to fit more precise
# and create filepaths, where the weights are saved
# Options: <adding and removing parameter of ES> <change parameter>
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500)
filepathSin = 'weights.best.hdf5'
mc = ModelCheckpoint(filepathSin, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# complie model, using an optimizer and a loss function
# Options: <optimizer> <loss function>
model.compile(optimizer = tf.optimizers.Adam(), loss='mean_squared_error')

# now fit the model for sine, using the train data, the validation data,
# a number of epochs, and the callback functions
# Options: <number of epochs> <callback>
model.fit(xs_train, sin_train, validation_data=(xs_val,sin_val), epochs=5000, callbacks=[es, mc])

# evaluate the sine model with the test data and print results
results = model.evaluate(xs_test, sin_test)
print('-----------------------------------------------------------')
print('Sin: test loss, test acc: ', results)
print('-----------------------------------------------------------')

# print a summary of the model
print('-----------------------------------------------------------')
print('Here is a summary of the sine model: ')
modelSin_data = model.summary()

# create a new model with the best weights for sine
bestModel = keras.Sequential([
    tf.keras.layers.Dense(512, activation='tanh', input_shape=(1,)),
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1)
    ])
# compile the new Model with the best weights, found during the training of the old Model
bestModel.load_weights('weights.best.hdf5')
bestModel.compile(optimizer = tf.optimizers.Adam(), loss='mean_squared_error')
resBestSin = bestModel.evaluate(xs_test, sin_test, verbose=0)

# now print the results of the new evaluation to compare
print('-----------------------------------------------------------')
print('Sin: test loss, test acc: ', resBestSin)

# print a summary of our both best models to compare
print('-----------------------------------------------------------')
print('Here is a summary of the sine model: ')
bestSin_data = bestModel.summary()
print('-----------------------------------------------------------')

# serialize the model to JSON
modelSin_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(modelSin_json)
print('Saved the model to disk')
print('-----------------------------------------------------------')

# here let the model predict different values between the given ones
plotValueSin=[]
for x in x_pred :
    plotValueSin.append(bestModel.predict([x]))

# now use a random value and compare the exact data to the approximation of our models
print('This is the random value')
print(rand)

print('This is the sin(rand):')
print(np.sin(rand))
print('This is what our model predict:')
print(bestModel.predict([rand]))