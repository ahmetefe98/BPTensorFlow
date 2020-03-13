'''final version of the unitCircle approximation by Ahmet Efe, Sven Leschber, Mika Rother'''

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# create arrays with x values in [0,2π] and a random value
# Options: <less/more values>
pi = np.pi
xs = np.arange(0,2*pi,0.15)
xc = np.arange(0,2*pi,0.15)
rand = np.random.uniform(0, 2*pi)

# create different values the model should predict later
x_pred = []
for k in range(len(xs)):
    if len(xs) == len(x_pred)+1:
        break
    val = (xs[k]+xs[k+1]) / 2
    x_pred.append(val)

# create sin and cos arrays of the x values
sin = np.sin(xs)
cos = np.cos(xc)

# create a professional dataset for the sine model
# Options: <different split>
xs, sin = shuffle(xs, sin, random_state=0)
xs_train, xs_temp, sin_train, sin_temp = train_test_split(xs, sin, test_size = 0.3)
xs_val, xs_test, sin_val, sin_test = train_test_split(xs_temp, sin_temp, test_size = 0.5)

# create a professional dataset for the cosine model
# Options: <different split>
xc, cos = shuffle(xc, cos, random_state=0)
xc_train, xc_temp, cos_train, cos_temp = train_test_split(xc, cos, test_size = 0.3)
xc_val, xc_test, cos_val, cos_test = train_test_split(xc_temp, cos_temp, test_size = 0.5)

# now create models for sine and cosine, using activation functions
# and different type of layer sizes
# Options: <number of layers> <size of layers> <activation function>
modelSin = keras.Sequential([
    tf.keras.layers.Dense(512, activation='tanh', input_shape=(1,)),
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1)
    ])

modelCos = keras.Sequential([
    tf.keras.layers.Dense(512, activation='tanh', input_shape=(1,)),
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1)
    ])

# define callback functions that helpes the model to fit more precise
# and create filepaths, where the weights are saved
# Options: <adding and removing parameter of ES> <change parameter>
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500)
filepathSin = 'weightsSin.best.hdf5'
mcSin = ModelCheckpoint(filepathSin, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
filepathCos = 'weightsCos.best.hdf5'
mcCos = ModelCheckpoint(filepathCos, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# complie both models, using an optimizer and a loss function
# Options: <optimizer> <loss function>
modelSin.compile(optimizer = tf.optimizers.Adam(), loss='mean_squared_error')
modelCos.compile(optimizer = tf.optimizers.Adam(), loss='mean_squared_error')

# now fit the models for sine and cosine, using the train data, the validation data,
# a number of epochs, and the callback functions
# Options: <number of epochs> <callback>
modelSin.fit(xs_train, sin_train, validation_data=(xs_val,sin_val), epochs=5000, callbacks=[es, mcSin])
modelCos.fit(xc_train, cos_train, validation_data=(xc_val,cos_val), epochs=5000, callbacks=[es, mcCos])

# evaluate both models with the test data
resultsSin = modelSin.evaluate(xs_test, sin_test)
resultsCos = modelCos.evaluate(xc_test, cos_test)

# now print the results of the evaluation
print('-----------------------------------------------------------')
print('Sin: test loss, test acc: ', resultsSin)
print('-----------------------------------------------------------')
print('Cos: test loss, test acc: ', resultsCos)
print('-----------------------------------------------------------')

# print a summary of our both models
print('-----------------------------------------------------------')
print('Here is a summary of the sine model: ')
modelSin_data = modelSin.summary()
print('-----------------------------------------------------------')
print('Here is a summary of the cosine model: ')
modelCos_data = modelCos.summary()
print('-----------------------------------------------------------')

# create a new model with the best weights for sine
bestSin = keras.Sequential([
    tf.keras.layers.Dense(512, activation='tanh', input_shape=(1,)),
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1)
    ])

bestSin.load_weights('weightsSin.best.hdf5')
bestSin.compile(optimizer = tf.optimizers.Adam(), loss='mean_squared_error')
resBestSin = bestSin.evaluate(xs_test, sin_test, verbose=0)

# create a new model with the best weights for cosine
bestCos = keras.Sequential([
    tf.keras.layers.Dense(512, activation='tanh', input_shape=(1,)),
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1)
    ])

bestCos.load_weights('weightsCos.best.hdf5')
bestCos.compile(optimizer = tf.optimizers.Adam(), loss='mean_squared_error')
resBestCos = bestCos.evaluate(xc_test, cos_test, verbose=0)

# now print the results of the new evaluation to compare
print('-----------------------------------------------------------')
print('Sin: test loss, test acc: ', resBestSin)
print('-----------------------------------------------------------')
print('Cos: test loss, test acc: ', resBestCos)
print('-----------------------------------------------------------')

# print a summary of our both best models to compare
print('-----------------------------------------------------------')
print('Here is a summary of the sine model: ')
bestSin_data = bestSin.summary()
print('-----------------------------------------------------------')
print('Here is a summary of the cosine model: ')
bestCos_data = bestCos.summary()
print('-----------------------------------------------------------')

# serialize both models to JSON
modelSin_json = modelSin.to_json()
with open('modelSin.json', 'w') as json_file:
    json_file.write(modelSin_json)
modelCos_json = modelCos.to_json()
with open('modelCos.json', 'w') as json_file:
    json_file.write(modelCos_json)
print('Saved both models to disk')
print('-----------------------------------------------------------')

# here let the model predict different values between the given ones
plotValueSin=[]
for x in x_pred :
    plotValueSin.append(bestSin.predict([x]))

plotValueCos=[]
for x in x_pred:
    plotValueCos.append(bestCos.predict([x]))

# now use a random value and compare the exact data to the approximation of our models
print('This is the random value')
print(rand)
print('This is the sin(rand):')
print(np.sin(rand))
print('This is what our model predict:')
print(bestSin.predict([rand]))
print('This is the cos(rand):')
print(np.cos(rand))
print('This is what our model predict:')
print(bestCos.predict([rand]))
print('-----------------------------------------------------------')