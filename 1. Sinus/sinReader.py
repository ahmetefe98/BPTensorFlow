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
from keras.models import model_from_json
import json

# create an array with x values in [0,π], the sin of the x values and a random value
pi = np.pi
xs = np.arange(0, pi, 0.1)
sin = np.sin(xs)
rand = np.random.uniform(0, pi)

# create a professional dataset for the sine model
# Options: <different split>
xs, sin = shuffle(xs, sin, random_state=0)
xs_train, xs_temp, sin_train, sin_temp = train_test_split(xs, sin, test_size = 0.3)
xs_val, xs_test, sin_val, sin_test = train_test_split(xs_temp, sin_temp, test_size = 0.5)

# load sine model from JSON file and create sine model with best weights
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
bestSin = tf.keras.models.model_from_json(loaded_model_json)
bestSin.load_weights('weights.best.hdf5')
bestSin.compile(optimizer = tf.optimizers.Adam(), loss='mean_squared_error')
scoreSin = bestSin.evaluate(xs_test, sin_test, verbose=0)

# print some results
print('-----------------------------------------------------------')
print('This is the random value:')
print(rand)
print('This is the sin(rand):')
print(np.sin(rand))
print('This is what our model predict:')
print(bestSin.predict([rand]))
print("Difference:")
print(abs(np.sin(rand)-bestSin.predict([rand])))
print('-----------------------------------------------------------')
print('Sin: test loss, test acc: ', scoreSin)
print('-----------------------------------------------------------')

# plot the difference between the true values and the approximation
plotX = np.arange(0.05, pi, 0.1)
plotY = []
plotValueSin = []
for x in plotX:
    predict = bestSin.predict([x])
    plotValueSin.append(predict)
    plotY.append(abs(predict-np.sin(x)))

plt.scatter(plotX,plotY)
plt.xlabel('x')
h=plt.ylabel('y')
h.set_rotation(0)
plt.title("Difference approximaton to real sine values")
plt.show()


# plot the curve to see how good the sine approximation is
sinModel=plt.scatter(plotX,plotValueSin)
sinReal=plt.scatter(plotX,np.sin(plotX))
plt.legend((sinModel, sinReal,),
        ('Sine Approximation', 'Real Sine curve'),
        scatterpoints=1,
        loc='lower center',
        ncol=1,
        fontsize=6)
plt.xlabel('x')
h=plt.ylabel('y')
h.set_rotation(0)
plt.title("Training Data")
plt.show()