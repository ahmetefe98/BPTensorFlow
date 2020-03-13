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
xt = []
yt = []
zt = []
for i in np.arange(0,math.pi,0.1):
    for j in np.arange(0,2*math.pi,0.1):
        x,y,z = torus(2,1,i,j)
        xt.append(x)
        yt.append(y)
        zt.append(z)

# create a professional dataset for the torus model
# Options: <different split>
xt, yt, zt = shuffle(xt, yt, zt, random_state=0)
x_train,x_temp,y_train,y_temp, z_train, z_temp = train_test_split(xt,yt,zt,test_size = 0.3)
x_val, x_test, y_val, y_test, z_val, z_test = train_test_split(x_temp,y_temp,z_temp,test_size = 0.5)

# combine the x and y array
xy_train = []
xy_val = []
xy_test = []
for i in range(len(x_train)):
    xy_train.append([x_train[i], y_train[i]])
for i in range(len(x_val)):
    xy_val.append([x_val[i], y_val[i]])
for i in range(len(x_test)):
    xy_test.append([x_test[i], y_test[i]])

#save the x,y,z values to a csv file
np.savetxt('xy_test.csv', xy_test, delimiter=',')
np.savetxt('z_test.csv', z_test, delimiter=',')

# now create a model for torus, using activation functions and different type of layer sizes
# Options: <number of layers> <size of layers> <activation function>
model = keras.Sequential([
    tf.keras.layers.Dense(2048, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1)
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
model.fit(xy_train, z_train, epochs=5000, validation_data=(xy_val,z_val), callbacks=[es, mc])

# evaluate the torus model with the test data and print results
results = model.evaluate(xy_test, z_test)
print ('test loss, test acc:', results)

# create a new torus model with the best weights
bestTorus = keras.Sequential([
    tf.keras.layers.Dense(2048, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1)
    ])

bestTorus.load_weights('weightsTorus.best.hdf5')
bestTorus.compile(optimizer=tf.optimizers.Adamax(), loss='logcosh')
resBestTorus = bestTorus.evaluate(xy_test, z_test, verbose=0)

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