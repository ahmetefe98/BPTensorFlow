'''final version of the unitCircleReader by Ahmet Efe, Sven Leschber, Mika Rother'''

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import json

# create arrays with x values in [0,2π] and a random value
# Options: <less/more values>
pi = np.pi
xs = np.arange(0,2*pi,0.15)
xc = np.arange(0,2*pi,0.15)
rand = np.random.uniform(0, 2*pi)

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

# load sine model from JSON file and create sine model with best weights
json_file = open('modelSin.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
bestSin = tf.keras.models.model_from_json(loaded_model_json)
bestSin.load_weights('weightsSin.best.hdf5')
bestSin.compile(optimizer = tf.optimizers.Adam(), loss='mean_squared_error')
scoreSin = bestSin.evaluate(xs_test, sin_test, verbose=0)

# load cosine model from JSON file and create cosine model with best weights
json_file = open('modelCos.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
bestCos = tf.keras.models.model_from_json(loaded_model_json)
bestCos.load_weights('weightsCos.best.hdf5')
bestCos.compile(optimizer = tf.optimizers.Adam(), loss='mean_squared_error')
scoreCos = bestCos.evaluate(xc_test, cos_test, verbose=0)

# print some results
print('-----------------------------------------------------------')
print('This is the random value')
print(rand)
print('This is the sin(rand):')
print(np.sin(rand))
print('This is what our model predict:')
print(bestSin.predict([rand]))
print("Difference:")
print(abs(np.sin(rand)-bestSin.predict([rand])))
print('This is the cos(rand):')
print(np.cos(rand))
print('This is what our model predict:')
print(bestCos.predict([rand]))
print("Difference:")
print(abs(np.cos(rand)-bestCos.predict([rand])))
print('-----------------------------------------------------------')
print('Sin: test loss, test acc: ', scoreSin)
print('-----------------------------------------------------------')
print('Cos: test loss, test acc: ', scoreCos)
print('-----------------------------------------------------------')

# plot the difference between the true values and the approximation
plotX = np.arange(0.05, 2*pi, 0.15)
plotSin = []
plotValueSin = []
for x in plotX:
    predict = bestSin.predict([x])
    plotValueSin.append(predict)
    plotSin.append(abs(predict-np.sin(x)))
plotCos = []
plotValueCos = []
for x in plotX:
    predict = bestCos.predict([x])
    plotValueCos.append(predict)
    plotCos.append(abs(predict-np.cos(x)))

sinDiff=plt.scatter(plotX,plotSin)
cosDiff=plt.scatter(plotX,plotCos)
plt.legend((sinDiff, cosDiff),
        ('Sine Difference', 'Cosine Difference'),
        scatterpoints=1,
        loc='lower left',
        ncol=1,
        fontsize=8)
plt.xlabel('x')
h=plt.ylabel('y')
h.set_rotation(0)
plt.title("Difference approximaton to real sine and cosine values")
plt.show()

# Plot the curves to see how good the approximation is
sinModel=plt.scatter(plotX,plotValueSin)
sinReal=plt.scatter(plotX,np.sin(plotX))
cosModel=plt.scatter(plotX,plotValueCos)
cosReal=plt.scatter(plotX,np.cos(plotX))
plt.legend((sinModel, sinReal, cosModel,cosReal),
        ('Sine Approximation', 'Real Sine curve', 'Cosine Approximation',
        'Real Cosine curve'),
        scatterpoints=1,
        loc='lower left',
        ncol=1,
        fontsize=6)
plt.xlabel('x')
h=plt.ylabel('y')
h.set_rotation(0)
plt.title("Training Data")
plt.show()