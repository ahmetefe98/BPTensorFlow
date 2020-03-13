'''final version of the torus approximation by Ahmet Efe, Sven Leschber, Mika Rother'''

import math
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from keras.callbacks import ModelCheckpoint, EarlyStopping

"""
R = Abstand zwischen (0,0,0) und Mitte vom Torus
r = Radius vom Kreis
p = Bogenmaß im Torus
t = Bogenmaß innen
"""
# a function to calculate the values of torus
def torus(R, r, p, t):
    x = (R+r*math.cos(p))*math.cos(t)
    y = (R+r*math.cos(p))*math.sin(t)
    z = r*math.sin(p)
    return x,y,z

# create three arrays with x, y, z coordinates of the torus
input = []
output = []
for i in np.arange(0,math.pi,0.1):
    for j in np.arange(0,2*math.pi,0.1):
        input.append([i,j])
        x,y,z = torus(2,1,i,j)
        output.append([x,y,z])

# create a professional dataset for the torus model
# Options: <different split>
input, output = shuffle(input, output, random_state=0)
i_train,i_temp,o_train,o_temp = train_test_split(input, output,test_size = 0.3)
i_val, i_test, o_val, o_test = train_test_split(i_temp,o_temp,test_size = 0.5)

#save the x,y,z values to a csv file
np.savetxt('i_test.csv', i_test, delimiter=',')
np.savetxt('o_test.csv', o_test, delimiter=',')

# now create a model for torus, using activation functions and different type of layer sizes
# Options: <number of layers> <size of layers> <activation function>
model = keras.Sequential([
    tf.keras.layers.Dense(2048, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3)
    ])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500)
filepath = 'weightsTorus.best.hdf5'
mc = ModelCheckpoint(filepath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# complie model, using an optimizer and a loss function
# Options: <optimizer> <loss function>
model.compile(optimizer = tf.optimizers.Adamax(), loss='logcosh')

# now fit the model for torus, using the train data, the validation data,
# a number of epochs
# Options: <number of epochs>
model.fit(i_train, o_train, epochs=5000, validation_data=(i_val,o_val), callbacks=[es, mc])

# evaluate the torus model with the test data and print results
results = model.evaluate(i_test, o_test)
print ('test loss, test acc:', results)

# create a new torus model with the best weights
bestTorus = keras.Sequential([
    tf.keras.layers.Dense(2048, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3)
    ])

bestTorus.load_weights('weightsTorus.best.hdf5')
bestTorus.compile(optimizer=tf.optimizers.Adamax(), loss='logcosh')
resBestTorus = bestTorus.evaluate(i_test, o_test, verbose=0)

# now print the results of the new evaluation to compare
print('-----------------------------------------------------------')
print('Torus: test loss, test acc: ', resBestTorus)

# print a summary of our best model to compare
print('-----------------------------------------------------------')
print('Here is a summary of the torus model: ')
bestTorus_data = bestTorus.summary()
print('-----------------------------------------------------------')

# serialize torus model to JSON
torus_json = bestTorus.to_json()
with open('bestTorus.json', 'w') as json_file:
    json_file.write(torus_json)
print('Saved the model to disk')
print('-----------------------------------------------------------')